# app.py
# Streamlit skeleton + Assignment -> Problem navigation + per-part numeric answer boxes
#
# Supports problem JSON files with optional "parts" (numeric only).
# - If "parts" exists, it renders one numeric input box per part.
# - If no "parts", it renders a single numeric input box.
#
# Optional (Week 2-ready): If data/answer_key.csv exists, it will grade answers with tolerance.
#
# File structure:
# meb_tutor_app/
#   app.py
#   data/
#     assignments.json
#     answer_key.csv               (optional)
#     problems/
#       MEB_001.json
#       MEB_002.json
#
# Run:
#   pip install streamlit
#   streamlit run app.py

import csv
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import streamlit as st

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROBLEMS_DIR = DATA_DIR / "problems"
ASSIGNMENTS_FILE = DATA_DIR / "assignments.json"
ANSWER_KEY_FILE = DATA_DIR / "answer_key.csv"


# -----------------------------
# Demo data (optional)
# -----------------------------
def ensure_demo_data(num_problems: int = 2) -> None:
    """
    Creates minimal demo data if files do not exist yet.
    Safe to delete once you add real problems.
    """
    PROBLEMS_DIR.mkdir(parents=True, exist_ok=True)

    # Demo assignment
    if not ASSIGNMENTS_FILE.exists():
        demo_ids = [f"MEB_{i:03d}" for i in range(1, num_problems + 1)]
        assignments = {"Assignment 1": demo_ids}
        ASSIGNMENTS_FILE.write_text(json.dumps(assignments, indent=2), encoding="utf-8")

    # Demo problems (simple)
    for i in range(1, num_problems + 1):
        pid = f"MEB_{i:03d}"
        pfile = PROBLEMS_DIR / f"{pid}.json"
        if not pfile.exists():
            problem = {
                "problem_id": pid,
                "title": f"Demo Problem {pid}",
                "statement": (
                    f"This is a demo problem for {pid}.\n\n"
                    "Week 1/2: navigation + numeric answer boxes.\n"
                    "Week 2: add answer_key.csv for grading.\n"
                ),
                "parts": [
                    {
                        "part_id": "a",
                        "prompt": "Enter a numeric answer for part (a).",
                        "input_type": "numeric",
                        "expected_output": {"name": "ans_a", "units": "unitless"},
                    }
                ],
            }
            pfile.write_text(json.dumps(problem, indent=2), encoding="utf-8")


# -----------------------------
# Load helpers
# -----------------------------
@st.cache_data
def load_assignments() -> dict:
    """Load assignment -> [problem_ids] mapping."""
    if not ASSIGNMENTS_FILE.exists():
        raise FileNotFoundError(f"Missing {ASSIGNMENTS_FILE}")
    return json.loads(ASSIGNMENTS_FILE.read_text(encoding="utf-8"))


@st.cache_data
def load_problem(problem_id: str) -> dict:
    """Load a single problem JSON."""
    path = PROBLEMS_DIR / f"{problem_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing problem file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_answer_key() -> Dict[Tuple[str, str], Dict[str, str]]:
    """
    Load answer keys from CSV, keyed by (problem_id, part_id).

    Expected columns:
      problem_id, part_id, answer_value, answer_units, tolerance_type, tolerance_value
    Example row:
      MEB_002,a,40,mol/min,absolute,1
    """
    key: Dict[Tuple[str, str], Dict[str, str]] = {}
    if not ANSWER_KEY_FILE.exists():
        return key

    with ANSWER_KEY_FILE.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"problem_id", "part_id", "answer_value", "answer_units", "tolerance_type", "tolerance_value"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"{ANSWER_KEY_FILE} must include columns: {', '.join(sorted(required))}"
            )

        for row in reader:
            pid = (row.get("problem_id") or "").strip()
            part_id = (row.get("part_id") or "").strip()
            if not pid or not part_id:
                continue
            key[(pid, part_id)] = {k: (v or "").strip() for k, v in row.items()}

    return key


def init_session_state() -> None:
    st.session_state.setdefault("selected_assignment", None)
    st.session_state.setdefault("selected_problem_id", None)


# -----------------------------
# Grading helpers (Week 2-ready)
# -----------------------------
def parse_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def within_tolerance(student_val: float, answer_val: float, tol_type: str, tol_value: float) -> bool:
    tol_type = (tol_type or "").lower().strip()
    if tol_type == "relative":
        # relative tolerance: |x - a| <= tol * |a|
        return abs(student_val - answer_val) <= tol_value * abs(answer_val)
    # default absolute
    return abs(student_val - answer_val) <= tol_value


def grade_part(
    problem_id: str,
    part_id: str,
    student_text: str,
    answer_key: Dict[Tuple[str, str], Dict[str, str]],
) -> Tuple[Optional[bool], str]:
    """
    Returns (is_correct, message). is_correct None means "not gradable" (no key or invalid input).
    """
    k = answer_key.get((problem_id, part_id))
    if not k:
        return None, "No answer key for this part (not graded yet)."

    student_val = parse_float(student_text.strip())
    if student_val is None:
        return False, "Please enter a numeric value."

    ans_val = parse_float(k.get("answer_value", ""))
    tol_val = parse_float(k.get("tolerance_value", ""))
    tol_type = k.get("tolerance_type", "absolute")
    ans_units = k.get("answer_units", "")

    if ans_val is None or tol_val is None:
        return None, "Answer key row is invalid (ask developer to fix CSV)."

    ok = within_tolerance(student_val, ans_val, tol_type, tol_val)
    if ok:
        return True, f"Correct (within {tol_type} tolerance). Expected units: {ans_units}"
    return False, f"Incorrect. Expected units: {ans_units}"


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

    default_a = 0
    if st.session_state["selected_assignment"] in assignment_names:
        default_a = assignment_names.index(st.session_state["selected_assignment"])

    selected_assignment = st.sidebar.selectbox(
        "Select Assignment",
        assignment_names,
        index=default_a,
    )
    st.session_state["selected_assignment"] = selected_assignment

    problem_ids = assignments.get(selected_assignment, [])
    if not problem_ids:
        st.sidebar.warning("No problems listed for this assignment.")
        st.stop()

    default_p = 0
    if st.session_state["selected_problem_id"] in problem_ids:
        default_p = problem_ids.index(st.session_state["selected_problem_id"])

    selected_problem_id = st.sidebar.selectbox(
        "Select Problem",
        problem_ids,
        index=default_p,
        help="Students choose from a list to avoid mistyping IDs.",
    )
    st.session_state["selected_problem_id"] = selected_problem_id

    st.sidebar.divider()
    st.sidebar.caption("Week 1/2: navigation + numeric answers. Week 3+: uploads + AI feedback.")


def render_problem(problem: Dict[str, Any], answer_key: Dict[Tuple[str, str], Dict[str, str]]) -> None:
    """Main panel: display problem + numeric inputs per part."""
    pid = problem.get("problem_id", "")
    title = problem.get("title", pid or "Problem")

    st.markdown(f"## {title}")
    st.markdown(f"**Problem ID:** `{pid}`")
    st.markdown(problem.get("statement", "_No statement provided._").replace("\n", "  \n"))

    parts = problem.get("parts", [])
    st.divider()
    st.subheader("Submit Answers")

    responses: Dict[str, str] = {}

    if not parts:
        # Single-answer fallback
        st.write("Enter your numeric answer below:")
        responses["answer"] = st.text_input("Answer", key=f"{pid}_single_answer")
        submitted = st.button("Submit", key=f"{pid}_submit_single")

        if submitted:
            st.info("Submitted. (Add parts + answer_key.csv for grading.)")
        return

    # Render each numeric part
    for p in parts:
        part_id = str(p.get("part_id", "")).strip() or "?"
        prompt = p.get("prompt", "")
        expected = p.get("expected_output", {}) or {}
        units = expected.get("units", "")

        st.markdown(f"### Part ({part_id})")
        if prompt:
            st.write(prompt)

        label = "Answer"
        if units:
            label = f"Answer ({units})"

        responses[part_id] = st.text_input(label, key=f"{pid}_part_{part_id}_answer")

    submitted = st.button("Submit", key=f"{pid}_submit_parts")

    if submitted:
        st.success("Submission received.")
        # Grade if answer_key.csv exists and has rows for these parts
        st.subheader("Results")
        any_graded = False

        for p in parts:
            part_id = str(p.get("part_id", "")).strip() or "?"
            student_text = responses.get(part_id, "")

            is_correct, msg = grade_part(pid, part_id, student_text, answer_key)

            # Display result row
            if is_correct is None:
                st.info(f"Part ({part_id}): {msg}")
            elif is_correct:
                any_graded = True
                st.success(f"Part ({part_id}): {msg}")
            else:
                any_graded = True
                st.error(f"Part ({part_id}): {msg}")

        if not any_graded and not ANSWER_KEY_FILE.exists():
            st.caption("Tip: Add data/answer_key.csv to enable grading next week.")


def render_dev_notes() -> None:
    with st.expander("Developer notes (expected repo structure)", expanded=False):
        st.code(
            "meb_tutor_app/\n"
            "  app.py\n"
            "  data/\n"
            "    assignments.json\n"
            "    answer_key.csv            (optional)\n"
            "    problems/\n"
            "      MEB_001.json\n"
            "      MEB_002.json\n",
            language="text",
        )
        st.write(
            "• problems/*.json are student-facing problem prompts.\n"
            "• assignments.json maps assignment name -> list of problem IDs.\n"
            "• answer_key.csv (optional) enables grading by part_id.\n"
        )


# -----------------------------
# App entry
# -----------------------------
st.set_page_config(page_title="MEB Tutor", layout="wide")
init_session_state()

# If you already have real data committed, you can delete this line:
ensure_demo_data(num_problems=2)

st.title("MEB Homework Tutor")
st.caption("Assignment → Problem navigation + numeric answers per part.")

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

# Load selected problem
try:
    problem = load_problem(st.session_state["selected_problem_id"])
except Exception as e:
    st.error(
        f"Could not load problem '{st.session_state['selected_problem_id']}'.\n\n"
        f"Expected file at {PROBLEMS_DIR}/{st.session_state['selected_problem_id']}.json\n\n"
        f"Error: {e}"
    )
    st.stop()

# Load answer key (optional)
try:
    answer_key = load_answer_key()
except Exception as e:
    st.warning(f"Answer key file exists but could not be loaded: {e}")
    answer_key = {}

render_problem(problem, answer_key)
render_dev_notes()
