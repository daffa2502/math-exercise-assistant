# Math Exercise Assistant

A personal AI project that works as a math exercise assistant. This application uses open-source language models to generate customized math problems, solutions, multiple-choice options, and explanations.

## Features

- Generate math problems with adjustable topics and difficulty levels
- Multiple-choice format with explanations
- Timed exercises with progress tracking
- Problem flagging for review
- Detailed results and performance analytics
- Ask for additional explanations after completing exercises
- Dockerized for easy deployment

## Topics & Difficulty Levels

### Topics
- Algebra
- Geometry
- Trigonometry
- Calculus
- Statistics
- Probability

### Difficulty Levels
- Easy
- Medium
- Hard

## Tech Stack

- **Frontend/UI**: Streamlit
- **Backend**: Python
- **AI Models**: Hugging Face Transformers (Flan-T5 models)
- **Containerization**: Docker & Docker Compose

## Installation & Setup

### Prerequisites

- Python 3.8+ (if running locally)
- Docker and Docker Compose (if using containerized setup)
- 4GB+ RAM for running the models

### Option 1: Local Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd math-exercise-assistant
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   streamlit run app/main.py
   ```

### Option 2: Docker Setup (Recommended)

1. Clone the repository:
   ```
   git clone <repository-url>
   cd math-exercise-assistant
   ```

2. Build and start the Docker containers:
   ```
   docker-compose up -d
   ```

3. Access the application at: `http://localhost:8501`

## Usage Guide

1. **Configure Exercise Session**:
   - Select the number of problems
   - Choose a math topic
   - Set the difficulty level
   - Specify the time limit in minutes

2. **Start the Exercise**:
   - Click "Start Exercise" to begin
   - The system will generate the requested number of problems

3. **Answer Questions**:
   - Read each problem carefully
   - Select your answer from the multiple-choice options
   - Use navigation buttons to move between problems
   - Flag questions for later review if needed
   - Monitor your remaining time

4. **Review Results**:
   - After completing the exercise or when time runs out, view your results
   - See your score and detailed performance
   - Review explanations for each problem
   - Ask for additional explanations if needed

5. **Save Results** (optional):
   - Save your results for future reference

## Customizing Models

You can customize the models used for problem generation and explanation by setting environment variables:

- `PROBLEM_GENERATOR_MODEL`: Model used for generating math problems
- `EXPLANATION_MODEL`: Model used for providing additional explanations

Default models:
- Problem Generator: `google/flan-t5-base`
- Explanation: `google/flan-t5-large`

These can be changed in the `.env` file or in the `docker-compose.yml` file.

## Fine-tuning Your Own Model

For optimal results, you may want to fine-tune the problem generator model on a dataset of math problems with explanations. The application is designed to work with any model that can generate JSON-formatted outputs with the following structure:

```json
{
  "problem": "What is the solution to the equation 2x + 3 = 9?",
  "solution": "x = 3",
  "explanation": "To solve for x, subtract 3 from both sides: 2x = 6. Then divide both sides by 2: x = 3.",
  "options": ["A. x = 2", "B. x = 3", "C. x = 4", "D. x = 6"],
  "correct_option": "B"
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Hugging Face](https://huggingface.co/) for providing open-source models
- [Streamlit](https://streamlit.io/) for the simple yet powerful UI framework