import logging
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.settings import CACHE_DIR, EXPLANATION_MODEL

logger = logging.getLogger(__name__)


class MathExplainer:
    """
    A class to provide additional explanations for math problems
    using a language model.
    """

    def __init__(
        self,
        model_name: str = EXPLANATION_MODEL,
        device: Optional[str] = None,
        mock_mode: bool = False,
    ):
        """
        Initialize the math explainer with the specified model.

        Args:
            model_name: Hugging Face model name/path
            device: Device to run the model on ('cuda', 'cpu', etc.)
            mock_mode: If True, skip model loading and use fallback explanations
        """
        self.mock_mode = mock_mode

        if mock_mode:
            logger.info(
                "Running explanation model in mock mode - no models will be loaded"
            )
            self.tokenizer = None
            self.model = None
            return

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading explanation model {model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir=CACHE_DIR
        )
        self.model.to(self.device)

        logger.info("Model loaded successfully")

    def generate_explanation(
        self, problem_data: Dict[str, Any], user_query: str
    ) -> str:
        """
        Generate a detailed explanation based on the user's query about a specific math problem.

        Args:
            problem_data: Dictionary containing problem information
            user_query: User's question about the problem

        Returns:
            A detailed explanation addressing the user's query
        """
        if self.mock_mode:
            return self._create_fallback_explanation(problem_data, user_query)

        # Extract problem content
        problem = problem_data["problem"]
        original_explanation = problem_data["explanation"]
        solution = problem_data["solution"]

        # Create prompt for the model
        prompt = self._create_prompt(
            problem, original_explanation, solution, user_query
        )

        # Generate output from the model
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_sequences = self.model.generate(
                **inputs,
                max_length=1024,
                num_return_sequences=1,
                temperature=0.3,  # Lower temperature for more focused explanations
                do_sample=True,
                top_p=0.95,
            )

        # Decode the output
        output_text = self.tokenizer.decode(
            output_sequences[0], skip_special_tokens=True
        )

        # Clean up the output if needed
        explanation = self._clean_output(output_text)

        return explanation

    def _create_prompt(
        self, problem: str, original_explanation: str, solution: str, user_query: str
    ) -> str:
        """Create the prompt for the model"""
        return (
            f"Math Problem: {problem}\n\n"
            f"Solution: {solution}\n\n"
            f"Original Explanation: {original_explanation}\n\n"
            f"Student Question: {user_query}\n\n"
            f"Please provide a more detailed explanation that addresses the student's question:"
        )

    def _clean_output(self, output_text: str) -> str:
        """Clean up the model output if necessary"""
        # Remove any potential prefixes like "Here's a detailed explanation:"
        lines = output_text.split("\n")
        cleaned_lines = []

        # Skip potential prefixes in the first few lines
        started_content = False
        for line in lines:
            if started_content or not (
                line.startswith("Here")
                or line.startswith("Let me")
                or line.startswith("I'll")
                or line.strip() == ""
            ):
                started_content = True
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines) if cleaned_lines else output_text

    def _create_fallback_explanation(
        self, problem_data: Dict[str, Any], user_query: str
    ) -> str:
        """Create a fallback explanation when in mock mode"""
        problem = problem_data["problem"]
        solution = problem_data["solution"]
        topic = problem_data.get("metadata", {}).get("topic", "math")

        return f"""**Additional Explanation (Mock Mode)**

You asked: "{user_query}"

Here's a more detailed breakdown of the problem: {problem}

The solution is: {solution}

This is a {topic} problem. In mock mode, I can provide some general guidance:

**Key steps for solving {topic} problems:**
- Always read the problem carefully and identify what you're looking for
- Write down what information is given
- Apply the appropriate formulas or methods for this topic
- Show your work step by step
- Check your answer by substituting back or using alternative methods

**Common mistakes to avoid:**
- Rushing through calculations
- Forgetting to apply order of operations
- Not checking units in word problems
- Making sign errors in algebra

This is a generated explanation in mock mode. In the full version, the AI model would provide a more personalized and detailed explanation based on your specific question.
"""
