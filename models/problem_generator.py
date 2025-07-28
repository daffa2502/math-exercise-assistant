import json
import logging
import random
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.settings import CACHE_DIR, PROBLEM_GENERATOR_MODEL

logger = logging.getLogger(__name__)


class MathProblemGenerator:
    """
    A class to generate math problems, solutions, explanations, and multiple-choice options
    using Qwen3-1.7B model.
    """

    def __init__(
        self,
        model_name: str = PROBLEM_GENERATOR_MODEL,
        device: Optional[str] = None,
        mock_mode: bool = False,
    ):
        """
        Initialize the problem generator with the specified model.

        Args:
            model_name: Hugging Face model name/path or local path
            device: Device to run the model on ('cuda', 'cpu', etc.)
            mock_mode: If True, skip model loading and use fallback templates
        """
        self.mock_mode = mock_mode

        if mock_mode:
            logger.info("Running in mock mode - no models will be loaded")
            self.tokenizer = None
            self.model = None
            return

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading problem generator model {model_name} on {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=CACHE_DIR
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16
                if torch.cuda.is_available()
                else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                cache_dir=CACHE_DIR,
            )

            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Falling back to mock mode")
            self.mock_mode = True
            self.tokenizer = None
            self.model = None

    def generate_problem(self, topic: str, difficulty: str) -> Dict[str, Any]:
        """
        Generate a math problem, solution, explanation, and multiple choice options.

        Args:
            topic: Math topic (e.g., "algebra", "geometry")
            difficulty: Problem difficulty (e.g., "easy", "medium", "hard")

        Returns:
            Dictionary containing the problem, solution, explanation, and options
        """
        if self.mock_mode:
            return self._create_fallback_problem(topic, difficulty)

        # Create the prompt for Qwen
        prompt = self._create_qwen_prompt(topic, difficulty)

        try:
            # Generate output from the model
            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            logger.info(
                f"Generating problem for topic: {topic}, difficulty: {difficulty}"
            )
            with torch.no_grad():
                output_sequences = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode and parse the output
            output_text = self.tokenizer.decode(
                output_sequences[0], skip_special_tokens=True
            )

            # Remove the prompt from the output
            generated_text = output_text[len(prompt) :].strip()

            # Parse the JSON output
            problem_data = self._parse_qwen_output(generated_text)
            # Add metadata
            problem_data["metadata"] = {"topic": topic, "difficulty": difficulty}
            return problem_data

        except Exception as e:
            logger.error(f"Error generating problem with Qwen: {e}")
            # Return fallback problem if generation fails
            return self._create_fallback_problem(topic, difficulty)

    def _create_qwen_prompt(self, topic: str, difficulty: str) -> str:
        """Create the prompt for Qwen model"""
        return f"""Create a {difficulty} level {topic} math problem. Provide the response as a JSON object with these exact fields:
- "problem": the math question
- "solution": the correct answer
- "explanation": step-by-step solution explanation
- "options": array of 4 multiple choice options in format ["A. option1", "B. option2", "C. option3", "D. option4"]
- "correct_option": the letter of the correct answer (A, B, C, or D)

Example format:
{{"problem": "What is 2 + 2?", "solution": "4", "explanation": "Adding 2 + 2 equals 4", "options": ["A. 3", "B. 4", "C. 5", "D. 6"], "correct_option": "B"}}

Generate one {difficulty} level {topic} math problem. Response should be exactly one valid JSON object without any additional text or formatting.
"""

    def _parse_qwen_output(self, output_text: str) -> Dict[str, Any]:
        """Parse the Qwen model output into a structured format"""
        try:
            print(f"Raw output from model: {output_text}")

            # Clean the output text - remove markdown code blocks if present
            cleaned_text = output_text.strip()

            # Remove markdown code block markers
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]  # Remove "```json"
            if cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]  # Remove "```"
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]  # Remove trailing "```"

            cleaned_text = cleaned_text.strip()

            # Find all potential JSON objects in the text
            json_objects = []
            brace_count = 0
            start_idx = -1

            for i, char in enumerate(cleaned_text):
                if char == "{":
                    if brace_count == 0:
                        start_idx = i
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0 and start_idx != -1:
                        # Found a complete JSON object
                        json_candidate = cleaned_text[start_idx : i + 1]
                        json_objects.insert(0, json_candidate)

            if not json_objects:
                raise ValueError("No valid JSON objects found in model output")

            # Try to parse each JSON object, starting with the largest one
            json_objects.sort(key=len, reverse=True)

            for json_text in json_objects:
                try:
                    print(f"Trying to parse JSON: {json_text}")

                    # Clean up common issues in generated JSON
                    json_text = json_text.replace(
                        "'", '"'
                    )  # Replace single quotes with double quotes
                    json_text = json_text.replace('\\"', '"')  # Fix escaped quotes

                    # Parse the JSON
                    problem_data = json.loads(json_text)

                    # Validate required fields
                    required_fields = [
                        "problem",
                        "solution",
                        "explanation",
                        "options",
                        "correct_option",
                    ]

                    if not all(field in problem_data for field in required_fields):
                        print("Missing required fields, trying next JSON object...")
                        continue

                    # Validate options format
                    if (
                        not isinstance(problem_data["options"], list)
                        or len(problem_data["options"]) != 4
                    ):
                        print("Invalid options format, trying next JSON object...")
                        continue

                    # Validate correct_option
                    if problem_data["correct_option"] not in ["A", "B", "C", "D"]:
                        print("Invalid correct_option, trying next JSON object...")
                        continue

                    print(f"Successfully parsed JSON: {problem_data}")
                    return problem_data

                except json.JSONDecodeError as je:
                    print(f"JSON decode error for candidate: {je}")
                    continue

            # If we get here, none of the JSON objects were valid
            raise ValueError("No valid JSON objects could be parsed from model output")

        except Exception as e:
            logger.error(f"Error parsing JSON from Qwen output: {e}")
            logger.debug(f"Raw output: {output_text}")
            raise

    def _create_fallback_problem(self, topic: str, difficulty: str) -> Dict[str, Any]:
        """Create a fallback problem with comprehensive templates"""
        templates = {
            "algebra": {
                "easy": [
                    {
                        "problem": "Solve for x: 2x + 4 = 12",
                        "solution": "x = 4",
                        "explanation": "To solve for x, first subtract 4 from both sides: 2x = 8. Then divide both sides by 2: x = 4.",
                        "options": ["A. x = 2", "B. x = 4", "C. x = 6", "D. x = 8"],
                        "correct_option": "B",
                    },
                    {
                        "problem": "What is 3x - 7 when x = 5?",
                        "solution": "8",
                        "explanation": "Substitute x = 5 into the expression: 3(5) - 7 = 15 - 7 = 8.",
                        "options": ["A. 8", "B. 22", "C. 2", "D. 15"],
                        "correct_option": "A",
                    },
                ],
                "medium": [
                    {
                        "problem": "Solve the quadratic equation: x² - 5x + 6 = 0",
                        "solution": "x = 2 or x = 3",
                        "explanation": "Factor the quadratic: (x - 2)(x - 3) = 0. Setting each factor to zero: x - 2 = 0 or x - 3 = 0, so x = 2 or x = 3.",
                        "options": [
                            "A. x = 1, 6",
                            "B. x = 2, 3",
                            "C. x = -2, -3",
                            "D. x = 0, 5",
                        ],
                        "correct_option": "B",
                    },
                    {
                        "problem": "Simplify: (2x + 3)(x - 1)",
                        "solution": "2x² + x - 3",
                        "explanation": "Use FOIL method: (2x)(x) + (2x)(-1) + (3)(x) + (3)(-1) = 2x² - 2x + 3x - 3 = 2x² + x - 3.",
                        "options": [
                            "A. 2x² + x - 3",
                            "B. 2x² - x + 3",
                            "C. 2x² + 5x - 3",
                            "D. x² + x - 3",
                        ],
                        "correct_option": "A",
                    },
                ],
                "hard": [
                    {
                        "problem": "Find the discriminant of 3x² - 7x + 2 = 0",
                        "solution": "Discriminant = 25",
                        "explanation": "For ax² + bx + c = 0, discriminant = b² - 4ac. Here a=3, b=-7, c=2. So discriminant = (-7)² - 4(3)(2) = 49 - 24 = 25.",
                        "options": ["A. 25", "B. 23", "C. 27", "D. 21"],
                        "correct_option": "A",
                    }
                ],
            },
            "geometry": {
                "easy": [
                    {
                        "problem": "Find the area of a rectangle with length 8 cm and width 5 cm.",
                        "solution": "40 cm²",
                        "explanation": "Area of rectangle = length × width = 8 × 5 = 40 cm².",
                        "options": ["A. 26 cm²", "B. 40 cm²", "C. 13 cm²", "D. 80 cm²"],
                        "correct_option": "B",
                    },
                    {
                        "problem": "What is the perimeter of a square with side length 6 cm?",
                        "solution": "24 cm",
                        "explanation": "Perimeter of square = 4 × side length = 4 × 6 = 24 cm.",
                        "options": ["A. 12 cm", "B. 24 cm", "C. 36 cm", "D. 18 cm"],
                        "correct_option": "B",
                    },
                ],
                "medium": [
                    {
                        "problem": "Find the area of a triangle with base 10 cm and height 6 cm.",
                        "solution": "30 cm²",
                        "explanation": "Area of triangle = (1/2) × base × height = (1/2) × 10 × 6 = 30 cm².",
                        "options": ["A. 60 cm²", "B. 30 cm²", "C. 16 cm²", "D. 20 cm²"],
                        "correct_option": "B",
                    }
                ],
                "hard": [
                    {
                        "problem": "Find the volume of a sphere with radius 3 cm (use π ≈ 3.14).",
                        "solution": "113.04 cm³",
                        "explanation": "Volume of sphere = (4/3)πr³ = (4/3) × 3.14 × 3³ = (4/3) × 3.14 × 27 = 113.04 cm³.",
                        "options": [
                            "A. 113.04 cm³",
                            "B. 84.78 cm³",
                            "C. 56.52 cm³",
                            "D. 28.26 cm³",
                        ],
                        "correct_option": "A",
                    }
                ],
            },
            "trigonometry": {
                "easy": [
                    {
                        "problem": "What is sin(30°)?",
                        "solution": "1/2",
                        "explanation": "sin(30°) is a standard trigonometric value equal to 1/2 or 0.5.",
                        "options": ["A. 1/2", "B. √3/2", "C. √2/2", "D. 1"],
                        "correct_option": "A",
                    }
                ],
                "medium": [
                    {
                        "problem": "If cos(θ) = 3/5, find sin(θ) for θ in the first quadrant.",
                        "solution": "4/5",
                        "explanation": "Using the Pythagorean identity: sin²(θ) + cos²(θ) = 1. So sin²(θ) = 1 - (3/5)² = 1 - 9/25 = 16/25. Therefore sin(θ) = 4/5 (positive in first quadrant).",
                        "options": ["A. 4/5", "B. 3/4", "C. 5/4", "D. 2/3"],
                        "correct_option": "A",
                    }
                ],
                "hard": [
                    {
                        "problem": "Solve for x: 2sin²(x) - sin(x) - 1 = 0, where 0 ≤ x ≤ 2π",
                        "solution": "x = π/2, 7π/6, 11π/6",
                        "explanation": "Let y = sin(x). Then 2y² - y - 1 = 0. Factoring: (2y + 1)(y - 1) = 0. So y = -1/2 or y = 1. When sin(x) = 1, x = π/2. When sin(x) = -1/2, x = 7π/6 or 11π/6.",
                        "options": [
                            "A. π/2, 7π/6, 11π/6",
                            "B. π/6, π/2, 5π/6",
                            "C. π/3, 2π/3, π",
                            "D. 0, π/2, π",
                        ],
                        "correct_option": "A",
                    }
                ],
            },
            "calculus": {
                "easy": [
                    {
                        "problem": "Find the derivative of f(x) = 3x² + 2x + 1",
                        "solution": "f'(x) = 6x + 2",
                        "explanation": "Using the power rule: d/dx(3x²) = 6x, d/dx(2x) = 2, d/dx(1) = 0. So f'(x) = 6x + 2.",
                        "options": [
                            "A. 6x + 2",
                            "B. 3x + 2",
                            "C. 6x + 1",
                            "D. 3x² + 2",
                        ],
                        "correct_option": "A",
                    }
                ],
                "medium": [
                    {
                        "problem": "Find ∫(2x + 3)dx",
                        "solution": "x² + 3x + C",
                        "explanation": "∫2x dx = x² and ∫3 dx = 3x. Adding the constant of integration: x² + 3x + C.",
                        "options": [
                            "A. x² + 3x + C",
                            "B. 2x² + 3x + C",
                            "C. x² + 3 + C",
                            "D. 2x + 3x + C",
                        ],
                        "correct_option": "A",
                    }
                ],
                "hard": [
                    {
                        "problem": "Find the limit: lim(x→0) (sin x)/x",
                        "solution": "1",
                        "explanation": "This is a standard limit in calculus. Using L'Hôpital's rule or the squeeze theorem, lim(x→0) (sin x)/x = 1.",
                        "options": ["A. 1", "B. 0", "C. ∞", "D. -1"],
                        "correct_option": "A",
                    }
                ],
            },
            "statistics": {
                "easy": [
                    {
                        "problem": "Find the mean of the data set: 2, 4, 6, 8, 10",
                        "solution": "6",
                        "explanation": "Mean = (2 + 4 + 6 + 8 + 10) ÷ 5 = 30 ÷ 5 = 6.",
                        "options": ["A. 5", "B. 6", "C. 7", "D. 8"],
                        "correct_option": "B",
                    }
                ],
                "medium": [
                    {
                        "problem": "Find the standard deviation of: 1, 2, 3, 4, 5",
                        "solution": "√2 ≈ 1.41",
                        "explanation": "Mean = 3. Variance = [(1-3)² + (2-3)² + (3-3)² + (4-3)² + (5-3)²]/5 = [4+1+0+1+4]/5 = 2. Standard deviation = √2 ≈ 1.41.",
                        "options": ["A. √2", "B. 2", "C. √5", "D. 1"],
                        "correct_option": "A",
                    }
                ],
                "hard": [
                    {
                        "problem": "In a normal distribution with μ = 50 and σ = 10, what percentage of data falls within 2 standard deviations?",
                        "solution": "95%",
                        "explanation": "In a normal distribution, approximately 95% of data falls within 2 standard deviations of the mean (between μ - 2σ and μ + 2σ).",
                        "options": ["A. 68%", "B. 95%", "C. 99.7%", "D. 50%"],
                        "correct_option": "B",
                    }
                ],
            },
            "probability": {
                "easy": [
                    {
                        "problem": "What is the probability of getting heads when flipping a fair coin?",
                        "solution": "1/2",
                        "explanation": "A fair coin has two equally likely outcomes: heads or tails. So P(heads) = 1/2 = 0.5.",
                        "options": ["A. 1/4", "B. 1/3", "C. 1/2", "D. 2/3"],
                        "correct_option": "C",
                    }
                ],
                "medium": [
                    {
                        "problem": "A bag contains 3 red balls and 2 blue balls. What's the probability of drawing a red ball?",
                        "solution": "3/5",
                        "explanation": "Total balls = 3 + 2 = 5. Red balls = 3. P(red) = 3/5 = 0.6.",
                        "options": ["A. 2/5", "B. 3/5", "C. 1/2", "D. 3/2"],
                        "correct_option": "B",
                    }
                ],
                "hard": [
                    {
                        "problem": "Two dice are rolled. What's the probability that their sum is 7?",
                        "solution": "1/6",
                        "explanation": "Favorable outcomes for sum = 7: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1) = 6 outcomes. Total outcomes = 36. P(sum=7) = 6/36 = 1/6.",
                        "options": ["A. 1/6", "B. 1/12", "C. 7/36", "D. 1/4"],
                        "correct_option": "A",
                    }
                ],
            },
        }

        # Get the appropriate template
        topic_templates = templates.get(topic, templates["algebra"])
        difficulty_templates = topic_templates.get(
            difficulty, topic_templates["medium"]
        )

        # Randomly select a template
        selected_template = random.choice(difficulty_templates)

        # Add metadata
        problem_data = selected_template.copy()
        problem_data["metadata"] = {"topic": topic, "difficulty": difficulty}

        return problem_data

    def generate_batch(
        self, topic: str, difficulty: str, count: int
    ) -> List[Dict[str, Any]]:
        """Generate multiple problems at once"""
        return [self.generate_problem(topic, difficulty) for _ in range(count)]
