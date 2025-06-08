import streamlit as st

from config.settings import (
    AVAILABLE_TOPICS,
    DEFAULT_DIFFICULTY,
    DEFAULT_NUM_PROBLEMS,
    DEFAULT_TIME_MINUTES,
    DEFAULT_TOPIC,
    DIFFICULTY_LEVELS,
)
from models.explanation_model import MathExplainer
from models.problem_generator import MathProblemGenerator
from utils.helpers import ExamSession, save_exam_results


def setup_page():
    """Set up the Streamlit page with custom styling"""
    st.set_page_config(
        page_title="Math Exercise Assistant",
        page_icon="🧮",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown(
        """
    <style>
        .title {
            font-size: 42px;
            font-weight: bold;
            color: #1E88E5;
            margin-bottom: 20px;
        }
        .subtitle {
            font-size: 24px;
            color: #424242;
            margin-bottom: 30px;
        }
        .timer {
            font-size: 32px;
            font-weight: bold;
            color: #E53935;
            background-color: #F5F5F5;
            padding: 10px;
            border-radius: 5px;
            text-align: center;
        }
        .question-nav {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin: 10px 0;
        }
        .problem-card {
            background-color: #F5F5F5;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }
        .explanation-box {
            background-color: #E3F2FD;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }
        .correct-answer {
            color: #2E7D32;
            font-weight: bold;
        }
        .incorrect-answer {
            color: #C62828;
            font-weight: bold;
        }
        .stButton>button {
            width: 100%;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


def home_page():
    """Render the home page with exam configuration options"""
    st.markdown(
        '<div class="title">Math Exercise Assistant</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">Configure your math practice session</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        st.header("Exercise Settings")

        num_problems = st.number_input(
            "Number of problems",
            min_value=1,
            max_value=50,
            value=DEFAULT_NUM_PROBLEMS,
            step=1,
        )

        topic = st.selectbox(
            "Topic",
            options=AVAILABLE_TOPICS,
            index=AVAILABLE_TOPICS.index(DEFAULT_TOPIC)
            if DEFAULT_TOPIC in AVAILABLE_TOPICS
            else 0,
        )

        difficulty = st.selectbox(
            "Difficulty",
            options=DIFFICULTY_LEVELS,
            index=DIFFICULTY_LEVELS.index(DEFAULT_DIFFICULTY)
            if DEFAULT_DIFFICULTY in DIFFICULTY_LEVELS
            else 1,
        )

        time_minutes = st.number_input(
            "Time (minutes)",
            min_value=1,
            max_value=120,
            value=DEFAULT_TIME_MINUTES,
            step=1,
        )

    with col2:
        st.header("Information")
        st.info(
            """
            **Math Exercise Assistant** helps you practice math problems 
            with AI-generated exercises and explanations.
            
            1. Configure your exercise session
            2. Start the practice
            3. Answer the questions
            4. Review your results and explanations
            5. Ask for additional explanations if needed
            
            All problems are generated using AI, with detailed 
            step-by-step explanations.
            """
        )

        # Loading state placeholder
        loading_placeholder = st.empty()

        if st.button("Start Exercise", type="primary", key="start_button"):
            with loading_placeholder.container():
                with st.spinner("Generating math problems..."):
                    # Generate problems
                    try:
                        problem_generator = MathProblemGenerator()
                        problems = problem_generator.generate_batch(
                            topic, difficulty, int(num_problems)
                        )

                        # Create a new exam session
                        session = ExamSession(problems, int(time_minutes))

                        # Save to session state
                        st.session_state.exam_session = session
                        st.session_state.page = "exercise"
                        st.session_state.current_problem_index = 0
                        st.session_state.show_results = False

                        # Force page refresh to show the exercise page
                        st.experimental_rerun()

                    except Exception as e:
                        st.error(f"Error generating problems: {str(e)}")


def exercise_page():
    """Render the exercise page with problems and timer"""
    session = st.session_state.exam_session

    if session.start_time is None:
        session.start()

    # Display header
    col1, col2, col3 = st.columns([2, 6, 2])

    with col1:
        if st.button("⏹️ End Exercise"):
            session.end()
            st.session_state.show_results = True
            st.experimental_rerun()

    with col2:
        st.markdown(
            '<div class="title" style="text-align: center;">Math Exercise</div>',
            unsafe_allow_html=True,
        )

    with col3:
        # Display timer
        timer_display = session.timer.get_formatted_time()
        st.markdown(f'<div class="timer">{timer_display}</div>', unsafe_allow_html=True)

    # Problem navigation
    problem_nav()

    # Display current problem
    current_index = session.current_index
    problem = session.get_current_problem()

    with st.container():
        st.markdown(
            f'<div class="problem-card"><h3>Problem {current_index + 1}</h3>{problem["problem"]}</div>',
            unsafe_allow_html=True,
        )

        # Display options and get user answer
        options = problem.get("options", [])
        user_answer = session.user_answers[current_index]

        for i, option in enumerate(options):
            option_letter = option[0]  # Assuming format "A. Option text"
            is_selected = user_answer == option_letter

            if st.button(
                option,
                key=f"option_{current_index}_{i}",
                type="primary" if is_selected else "secondary",
            ):
                session.submit_answer(current_index, option_letter)
                st.experimental_rerun()

        # Problem actions
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("⬅️ Previous", disabled=current_index == 0):
                session.move_to_prev()
                st.experimental_rerun()

        with col2:
            flag_status = session.flagged_problems[current_index]
            flag_text = "🚩 Unflag" if flag_status else "🚩 Flag for Review"

            if st.button(flag_text):
                session.toggle_flag(current_index)
                st.experimental_rerun()

        with col3:
            if st.button("➡️ Next", disabled=current_index == session.num_problems - 1):
                session.move_to_next()
                st.experimental_rerun()

    # Check if time's up
    if session.timer.is_time_up():
        session.end()
        st.session_state.show_results = True
        st.warning("Time's up! Your answers have been submitted.")
        st.experimental_rerun()


def problem_nav():
    """Render the problem navigation bar"""
    session = st.session_state.exam_session
    num_cols = min(10, session.num_problems)

    st.write("Jump to problem:")

    cols = st.columns(num_cols)

    for i in range(session.num_problems):
        col_idx = i % num_cols

        with cols[col_idx]:
            # Determine button style
            button_type = "primary"
            if session.current_index == i:
                button_type = "primary"
            elif session.user_answers[i] is not None:
                button_type = "secondary"
            else:
                button_type = "secondary"

            # Add flag indicator
            button_text = str(i + 1)
            if session.flagged_problems[i]:
                button_text = f"{button_text} 🚩"

            if st.button(button_text, key=f"nav_{i}", type=button_type):
                session.move_to_problem(i)
                st.experimental_rerun()


def results_page():
    """Render the results page with explanations"""
    session = st.session_state.exam_session
    summary = session.get_exam_summary()

    # Header
    st.markdown('<div class="title">Exercise Results</div>', unsafe_allow_html=True)

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Problems", summary["total_problems"])

    with col2:
        st.metric("Answered", summary["answered_problems"])

    with col3:
        st.metric("Correct", summary["correct_answers"])

    with col4:
        st.metric("Score", f"{summary['score_percentage']:.1f}%")

    # Save results button
    if st.button("💾 Save Results"):
        filepath = save_exam_results(session)
        st.success(f"Results saved to {filepath}")

    # Back to home button
    if st.button("🏠 Back to Home"):
        # Reset session state
        for key in list(st.session_state.keys()):
            if key != "page":
                del st.session_state[key]
        st.session_state.page = "home"
        st.experimental_rerun()

    # Detailed results for each problem
    st.markdown("## Problem Review")

    for i, problem in enumerate(session.problems):
        with st.expander(f"Problem {i + 1}"):
            user_answer = session.user_answers[i]
            correct_answer = problem["correct_option"]
            is_correct = user_answer == correct_answer

            st.markdown(f"**Question:** {problem['problem']}")

            # Display options
            st.markdown("**Options:**")
            for option in problem["options"]:
                option_letter = option[0]
                if option_letter == correct_answer:
                    st.markdown(f"- {option} ✅")
                elif option_letter == user_answer:
                    st.markdown(f"- {option} ❌")
                else:
                    st.markdown(f"- {option}")

            # Display correctness
            if is_correct:
                st.markdown(
                    '<p class="correct-answer">Your answer is correct!</p>',
                    unsafe_allow_html=True,
                )
            else:
                if user_answer:
                    st.markdown(
                        '<p class="incorrect-answer">Your answer is incorrect.</p>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<p class="incorrect-answer">You did not answer this question.</p>',
                        unsafe_allow_html=True,
                    )

            # Explanation
            st.markdown('<div class="explanation-box">', unsafe_allow_html=True)
            st.markdown("**Explanation:**")
            st.markdown(problem["explanation"])
            st.markdown("</div>", unsafe_allow_html=True)

            # Ask for more explanation
            if "explanation_model" not in st.session_state:
                st.session_state.explanation_model = MathExplainer()

            user_query = st.text_input(
                "Ask for more explanation:",
                key=f"query_{i}",
                placeholder="E.g., 'Can you explain the first step more clearly?'",
            )

            if st.button("Get Explanation", key=f"explain_{i}"):
                if user_query:
                    with st.spinner("Generating additional explanation..."):
                        try:
                            additional_explanation = (
                                st.session_state.explanation_model.generate_explanation(
                                    problem, user_query
                                )
                            )
                            st.markdown(
                                '<div class="explanation-box">', unsafe_allow_html=True
                            )
                            st.markdown("**Additional Explanation:**")
                            st.markdown(additional_explanation)
                            st.markdown("</div>", unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error generating explanation: {str(e)}")
                else:
                    st.warning("Please enter your question first.")


def main():
    """Main function to run the Streamlit app"""
    setup_page()

    # Initialize session state
    if "page" not in st.session_state:
        st.session_state.page = "home"

    # Show appropriate page
    current_page = st.session_state.page

    if current_page == "home":
        home_page()
    elif current_page == "exercise":
        if st.session_state.show_results:
            results_page()
        else:
            exercise_page()
    else:
        st.error("Invalid page state")
        st.session_state.page = "home"
        st.experimental_rerun()


if __name__ == "__main__":
    main()
