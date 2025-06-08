import json
import logging
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
        self, model_name: str = PROBLEM_GENERATOR_MODEL, device: Optional[str] = None
    ):
        """
        Initialize the problem generator with the specified model.

        Args:
            model_name: Hugging Face model name/path
            device: Device to run the model on ('cuda', 'cpu', etc.)
        """
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
        """Create a fallback problem in case the model output parsing fails"""
        if topic == "algebra":
            return {
                "problem": "Solve for x: 2x + 3 = 9",
                "solution": "x = 3",
                "explanation": "To solve for x, we need to isolate x on one side of the equation. First, subtract 3 from both sides: 2x + 3 - 3 = 9 - 3, which gives 2x = 6. Then divide both sides by 2: 2x/2 = 6/2, which gives x = 3.",
                "options": ["A. x = 2", "B. x = 3", "C. x = 4", "D. x = 6"],
                "correct_option": "B",
                "metadata": {"topic": topic, "difficulty": difficulty},
            }
        else:
            return {
                "problem": f"This is a fallback {difficulty} {topic} problem.",
                "solution": "Solution to the fallback problem",
                "explanation": "This is a fallback explanation as the model output could not be parsed.",
                "options": ["A. Option 1", "B. Option 2", "C. Option 3", "D. Option 4"],
                "correct_option": "A",
                "metadata": {"topic": topic, "difficulty": difficulty},
            }

    def generate_batch(
        self, topic: str, difficulty: str, count: int
    ) -> List[Dict[str, Any]]:
        """Generate multiple problems at once"""
        return [self.generate_problem(topic, difficulty) for _ in range(count)]
