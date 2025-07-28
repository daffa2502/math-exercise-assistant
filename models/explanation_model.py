import logging
from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config.settings import CACHE_DIR, EXPLANATION_MODEL

logger = logging.getLogger(__name__)


class MathExplainer:
    """
    A class to provide additional explanations for math problems
    using Qwen3-1.7B model.
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
            model_name: Hugging Face model name/path or local path
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

            logger.info("Explanation model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading explanation model: {e}")
            logger.info("Falling back to mock mode")
            self.mock_mode = True
            self.tokenizer = None
            self.model = None

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

        # Create prompt for Qwen
        prompt = self._create_qwen_prompt(
            problem, original_explanation, solution, user_query
        )

        try:
            # Generate output from the model
            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_sequences = self.model.generate(
                    **inputs,
                    max_new_tokens=400,
                    num_return_sequences=1,
                    temperature=0.3,  # Lower temperature for more focused explanations
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode the output
            output_text = self.tokenizer.decode(
                output_sequences[0], skip_special_tokens=True
            )

            # Remove the prompt from the output
            explanation = output_text[len(prompt) :].strip()

            # Clean up the output if needed
            explanation = self._clean_qwen_output(explanation)

            return explanation

        except Exception as e:
            logger.error(f"Error generating explanation with Qwen: {e}")
            return self._create_fallback_explanation(problem_data, user_query)

    def _create_qwen_prompt(
        self, problem: str, original_explanation: str, solution: str, user_query: str
    ) -> str:
        """Create the prompt for Qwen model"""
        return f"""You are a helpful math tutor. A student has asked for additional explanation about a math problem.

Math Problem: {problem}

Solution: {solution}

Original Explanation: {original_explanation}

Student Question: {user_query}

Please provide a clear, detailed explanation that addresses the student's specific question. Focus on helping them understand the concept better.

"""

    def _clean_qwen_output(self, output_text: str) -> str:
        """Clean up the Qwen model output if necessary"""
        # Remove any potential prefixes or suffixes
        print("Raw output from Qwen:", output_text)
        lines = output_text.split("\n")
        cleaned_lines = []

        # Skip potential prefixes and empty lines at the beginning
        started_content = False
        for line in lines:
            line = line.strip()
            if started_content or (
                line and not line.startswith(("Here", "Let me", "I'll", "Sure"))
            ):
                started_content = True
                if line:  # Only add non-empty lines
                    cleaned_lines.append(line)

        result = "\n".join(cleaned_lines) if cleaned_lines else output_text

        # Limit length to avoid overly long responses
        if len(result) > 1000:
            result = result[:1000] + "..."

        return result

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
