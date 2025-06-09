import json
import logging
import random
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config.settings import CACHE_DIR, PROBLEM_GENERATOR_MODEL

logger = logging.getLogger(__name__)


class MathProblemGenerator:
    """
    A class to generate math problems, solutions, explanations, and multiple-choice options
    using a fine-tuned language model.
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
            model_name: Hugging Face model name/path
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

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, cache_dir=CACHE_DIR
        )
        self.model.to(self.device)

        logger.info("Model loaded successfully")

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

        # Create the prompt for the model
        prompt = self._create_prompt(topic, difficulty)

        # Generate output from the model
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_sequences = self.model.generate(
                **inputs,
                max_length=1024,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )

        # Decode and parse the output
        output_text = self.tokenizer.decode(
            output_sequences[0], skip_special_tokens=True
        )

        try:
            # Parse the JSON output
            problem_data = self._parse_output(output_text)
            # Add metadata
            problem_data["metadata"] = {"topic": topic, "difficulty": difficulty}
            return problem_data
        except Exception as e:
            logger.error(f"Error parsing model output: {e}")
            # Return fallback problem if parsing fails
            return self._create_fallback_problem(topic, difficulty)

    def _create_prompt(self, topic: str, difficulty: str) -> str:
        """Create the prompt for the model"""
        return (
            f"Generate a {difficulty} level {topic} math problem with an explanation, solution, "
            f"and four multiple-choice options (A, B, C, D), where only one is correct. "
            f"Format the output as valid JSON with the following fields: "
            f"'problem', 'solution', 'explanation', 'options', and 'correct_option'. "
            f"Make the explanation clear and educational, walking through the solution step-by-step."
        )

    def _parse_output(self, output_text: str) -> Dict[str, Any]:
        """Parse the model output into a structured format"""
        # Extract JSON from the output text
        json_start = output_text.find("{")
        json_end = output_text.rfind("}")

        if json_start == -1 or json_end == -1:
            raise ValueError("No valid JSON found in model output")

        json_text = output_text[json_start : json_end + 1]

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
            missing = [field for field in required_fields if field not in problem_data]
            raise ValueError(f"Missing required fields in model output: {missing}")

        return problem_data

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
