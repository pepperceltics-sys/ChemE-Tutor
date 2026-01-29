# app.py
# Week 1: Streamlit skeleton + Assignment -> Problem navigation
# - Loads assignments from: data/assignments.json
# - Loads problems from:     data/problems/<PROBLEM_ID>.json
# - If those files don't exist, auto-creates a demo dataset with 10 problems.
#
# Run:
#   pip install streamlit
#   streamlit run app.py

import json
from pathlib import Path
import streamlit as st

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROBLEMS_DIR = DATA_DIR / "problems"
ASSIGNMENTS_FILE = DATA_DIR / "assignments.json"


# -----------------------------
# Demo data (optional)
# -----------------------------
def ensure_demo_data(num_problems: int = 10) -> None:
    """
    Creates a minimal Week-1 dataset if you haven't added your own files yet.
    This is safe to delete later once you replace with real problems.

    Creates:
      data/assignments.json
      data/problems/MEB_001.json ... MEB_<num_problems>.json
    """
    PROBLEMS_DIR.mkdir(parents=True, exist_ok=True)

    # Create demo problems if missing
    demo_problem_ids = []
    for i in range(1, num_problems + 1):
        pid = f"MEB_{i:03d}"
        demo_problem_ids.append(pid)

        pfile = PROBLEMS_DIR / f"{pid}.json"
        if not pfile.exists():
            problem = {
                "problem_id": pid,
                "title": f"MEB Concept Check #{i:02d}",
                "statement": (
                    f"**{pid} (Demo Problem)**\n\n"
                    "This is a placeholder MEB problem for Week 1.\n\n"
                    "Prompt:\n"
                    "- Identify the system/control volume.\n"
                    "- List assumptions (steady/unsteady, adiabatic, negligible KE/PE).\n"
                    "- Write the appropriate balance form (no solving required).\n\n"
                    "_Week 2 will add answer checking (tolerance + units) and attempt tracking._"
                ),
                "inputs_expected": {
                    "answer_type": "conceptual",
                    "notes": "Auto-generated placeholder for Week 1."
                }
            }
            pfile.write_text(json.dumps(problem, indent=2), encoding="utf-8")

    # Create assignments file if missing
    if not ASSIGNMENTS_FILE.exists():
        assignments = {
            "Assignment 1": demo_problem_ids
        }
        ASSIGNMENTS_FILE.write_text(json.dumps(assignments, indent=2), encoding="utf-8")


# -----------------------------
# Load helpers
# -----------------------------
@st.cache_data
def load_assignments() -> dict:
    """Load assignment -> [problem_ids] mapping."""
    return json.loads(ASSIGNMENTS_FILE.read_text(encoding="utf-8"))


@st.cache_data
def load_problem(problem_id: str) -> dict:
    """Load a single problem JSON."""
    path = PROBLEMS_DIR / f"{problem_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def init_session_state() -> None:
    """Initialize navigation selections."""
    st.session_state.setdefault("selected_assignment", None)
    st.session_state.setdefault("selected_problem_id", None)


# -----------------------------
# UI helpers
# -----------------------------
def render_sidebar(assignments: dict) -> None:
    """Sidebar navigation: Assignment -> Problem."""
    st.sidebar.title("Navigation")

    assignment_names = list(assignments.keys())
    if not assignment_names:
        st.sidebar.error("No assignments found in data/assignments.json.")
        st.stop()

    # Default assignment
    default_a = 0
    if st.session_state["selected_assignment"] in assignment_names:
        default_a = assignment_names.index(st.session_state["selected_assignment"])

    selected_assignment = st.sidebar.selectbox(
        "Select Assignment",
        assignment_names,
        index=default_a
    )
    st.session_state["selected_assignment"] = selected_assignment

    # Problems for this assignment
    problem_ids = assignments.get(selected_assignment, [])
    if not problem_ids:
        st.sidebar.warning("No problems listed for this assignment.")
        st.stop()

    # Default problem
    default_p = 0
    if st.session_state["selected_problem_id"] in problem_ids:
        default_p = problem_ids.index(st.session_state["selected_problem_id"])

    selected_problem_id = st.sidebar.selectbox(
        "Select Problem",
        problem_ids,
        index=default_p,
        help="Students choose from a list to avoid mistyping IDs."
    )
    st.session_state["selected_problem_id"] = selected_problem_id

    st.sidebar.divider()
    st.sidebar.caption("Week 1 prototype: navigation + display only.")


def render_problem(problem: dict) -> None:
    """Main panel problem display."""
    title = problem.get("title", problem.get("problem_id", "Problem"))
    pid = problem.get("problem_id", "")

    st.markdown(f"## {title}")
    st.markdown(f"**Problem ID:** `{pid}`")

    statement = problem.get("statement", "_No statement provided._")
    st.markdown(statement)

    # Student answer placeholder (Week 1)
    st.divider()
    st.subheader("Answer (placeholder — not graded in Week 1)")
    st.text_input("Enter your answer (Week 2 will check it):", key="student_answer")
    st.button("Submit (disabled in Week 1)", disabled=True)

    # Show student-safe metadata if present
    inputs_expected = problem.get("inputs_expected", {})
    if inputs_expected:
        with st.expander("Problem metadata (student-safe)", expanded=False):
            st.json(inputs_expected)


def render_dev_notes() -> None:
    with st.expander("Developer notes (file structure)", expanded=False):
        st.code(
            "meb_tutor_app/\n"
            "  app.py\n"
            "  data/\n"
            "    assignments.json\n"
            "    problems/\n"
            "      MEB_001.json\n"
            "      MEB_002.json\n"
            "      ...\n",
            language="text"
        )
        st.write(
            "Replace demo problems by editing files in data/problems/.\n"
            "In Week 2 you'll add an answer key file and grading logic."
        )


# -----------------------------
# App entry
# -----------------------------
st.set_page_config(page_title="MEB Tutor — Week 1", layout="wide")
init_session_state()

# Auto-create demo dataset if needed
ensure_demo_data(num_problems=10)

st.title("MEB Homework Tutor (Week 1)")
st.caption("Skeleton app: Assignment → Problem navigation + problem display.")

# Load data
try:
    assignments = load_assignments()
except Exception as e:
    st.error(
        f"Could not load assignments. Expected file at {ASSIGNMENTS_FILE}.\n\n"
        f"Error: {e}"
    )
    st.stop()

# Sidebar navigation
render_sidebar(assignments)

# Load and show selected problem
try:
    problem = load_problem(st.session_state["selected_problem_id"])
except Exception as e:
    st.error(
        f"Could not load problem '{st.session_state['selected_problem_id']}'.\n\n"
        f"Expected file at {PROBLEMS_DIR}/{st.session_state['selected_problem_id']}.json\n\n"
        f"Error: {e}"
    )
    st.stop()

render_problem(problem)
render_dev_notes()
