import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class Timer:
    """Timer class for tracking exam duration"""

    def __init__(self, duration_minutes: int):
        """
        Initialize a timer with the specified duration

        Args:
            duration_minutes: Duration of the timer in minutes
        """
        self.duration_seconds = duration_minutes * 60
        self.start_time = None
        self.end_time = None
        self.paused_time = 0
        self.is_paused = False
        self.pause_start = None

    def start(self):
        """Start the timer"""
        self.start_time = time.time()
        self.end_time = self.start_time + self.duration_seconds
        self.is_paused = False
        logger.info(f"Timer started for {self.duration_seconds / 60} minutes")

    def pause(self):
        """Pause the timer"""
        if not self.is_paused and self.start_time is not None:
            self.pause_start = time.time()
            self.is_paused = True
            logger.info("Timer paused")

    def resume(self):
        """Resume the timer"""
        if self.is_paused and self.pause_start is not None:
            pause_duration = time.time() - self.pause_start
            self.paused_time += pause_duration
            self.is_paused = False
            logger.info(f"Timer resumed after {pause_duration:.1f} seconds")

    def get_remaining_seconds(self) -> int:
        """Get remaining seconds on the timer"""
        if self.start_time is None:
            return self.duration_seconds

        if self.is_paused:
            elapsed = self.pause_start - self.start_time - self.paused_time
        else:
            elapsed = time.time() - self.start_time - self.paused_time

        remaining = self.duration_seconds - elapsed
        return max(0, int(remaining))

    def get_formatted_time(self) -> str:
        """Get formatted remaining time as MM:SS"""
        seconds = self.get_remaining_seconds()
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{minutes:02d}:{remaining_seconds:02d}"

    def is_time_up(self) -> bool:
        """Check if the time is up"""
        return self.get_remaining_seconds() <= 0


class ExamSession:
    """Class to manage a math exam session"""

    def __init__(self, problems: List[Dict[str, Any]], duration_minutes: int):
        """
        Initialize an exam session

        Args:
            problems: List of problem dictionaries
            duration_minutes: Duration of the exam in minutes
        """
        self.problems = problems
        self.num_problems = len(problems)
        self.timer = Timer(duration_minutes)
        self.current_index = 0
        self.user_answers = [None] * self.num_problems
        self.flagged_problems = [False] * self.num_problems
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start the exam session"""
        self.timer.start()
        self.start_time = datetime.now()
        logger.info(f"Exam started at {self.start_time}")

    def end(self):
        """End the exam session"""
        self.end_time = datetime.now()
        logger.info(f"Exam ended at {self.end_time}")

    def submit_answer(self, problem_index: int, answer: str):
        """Submit an answer for a specific problem"""
        if 0 <= problem_index < self.num_problems:
            self.user_answers[problem_index] = answer
            logger.info(f"Answer submitted for problem {problem_index + 1}: {answer}")
        else:
            logger.error(f"Invalid problem index: {problem_index}")

    def toggle_flag(self, problem_index: int):
        """Toggle the flag status of a problem"""
        if 0 <= problem_index < self.num_problems:
            self.flagged_problems[problem_index] = not self.flagged_problems[
                problem_index
            ]
            logger.info(
                f"Problem {problem_index + 1} flag toggled to {self.flagged_problems[problem_index]}"
            )
        else:
            logger.error(f"Invalid problem index: {problem_index}")

    def move_to_next(self) -> bool:
        """Move to the next problem if possible"""
        if self.current_index < self.num_problems - 1:
            self.current_index += 1
            return True
        return False

    def move_to_prev(self) -> bool:
        """Move to the previous problem if possible"""
        if self.current_index > 0:
            self.current_index -= 1
            return True
        return False

    def move_to_problem(self, problem_index: int) -> bool:
        """Move to a specific problem by index"""
        if 0 <= problem_index < self.num_problems:
            self.current_index = problem_index
            return True
        return False

    def get_current_problem(self) -> Dict[str, Any]:
        """Get the current problem"""
        return self.problems[self.current_index]

    def get_exam_summary(self) -> Dict[str, Any]:
        """Get a summary of the exam session"""
        correct_count = 0
        for i, (problem, user_answer) in enumerate(
            zip(self.problems, self.user_answers)
        ):
            if user_answer == problem["correct_option"]:
                correct_count += 1

        return {
            "total_problems": self.num_problems,
            "answered_problems": sum(1 for a in self.user_answers if a is not None),
            "correct_answers": correct_count,
            "score_percentage": (correct_count / self.num_problems) * 100
            if self.num_problems > 0
            else 0,
            "time_taken": (self.end_time - self.start_time).total_seconds()
            if self.end_time and self.start_time
            else None,
            "detailed_results": [
                {
                    "problem_index": i,
                    "user_answer": user_answer,
                    "correct_answer": problem["correct_option"],
                    "is_correct": user_answer == problem["correct_option"],
                    "was_flagged": self.flagged_problems[i],
                }
                for i, (problem, user_answer) in enumerate(
                    zip(self.problems, self.user_answers)
                )
            ],
        }


def save_exam_results(session: ExamSession, filepath: Optional[str] = None) -> str:
    """Save exam results to a JSON file"""
    if filepath is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"exam_results_{timestamp}.json"

    summary = session.get_exam_summary()

    # Add more details to the export
    export_data = {
        "summary": summary,
        "problems": [
            {
                "problem_index": i,
                "problem_text": problem["problem"],
                "user_answer": session.user_answers[i],
                "correct_answer": problem["correct_option"],
                "options": problem["options"],
                "explanation": problem["explanation"],
                "topic": problem.get("metadata", {}).get("topic", "unknown"),
                "difficulty": problem.get("metadata", {}).get("difficulty", "unknown"),
            }
            for i, problem in enumerate(session.problems)
        ],
        "exam_metadata": {
            "start_time": session.start_time.isoformat()
            if session.start_time
            else None,
            "end_time": session.end_time.isoformat() if session.end_time else None,
            "duration_minutes": session.timer.duration_seconds // 60,
        },
    }

    with open(filepath, "w") as f:
        json.dump(export_data, f, indent=2)

    logger.info(f"Exam results saved to {filepath}")
    return filepath
