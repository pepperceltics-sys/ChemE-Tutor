# app.py
# Instructor-demo-ready build:
# - Assignment -> Problem navigation
# - Numeric answer entry per part
# - CSV answer key grading (tolerance)
# - Attempt logging (SQLite)
# - Upload workflow (PDF) shown only when incorrect
# - Fallback form if PDF can't be read / to provide extra info
#
# Expected file structure:
# meb_tutor_app/
#   app.py
#   data/
#     assignments.json
#     answer_key.csv            (optional but recommended)
#     problems/
#       MEB_001.json
#       MEB_002.json
#     uploads/                  (created if missing)
#     logs/                     (created if missing)

import csv
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import streamlit as st

# Optional PDF extraction (typed PDFs). If unavailable, fallback form still works.
try:
    import PyPDF2  # type: ignore
    HAS_PYPDF2 = True
except Exception:
    HAS_PYPDF2 = False


# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROBLEMS_DIR = DATA_DIR / "problems"
ASSIGNMENTS_FILE = DATA_DIR / "assignments.json"
ANSWER_KEY_FILE = DATA_DIR / "answer_key.csv"

UPLOADS_DIR = DATA_DIR / "uploads"
LOGS_DIR = DATA_DIR / "logs"
DB_FILE = LOGS_DIR / "attempts.sqlite3"


# -----------------------------
# Utilities
# -----------------------------
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def within_tolerance(student_val: float, answer_val: float, tol_type: str, tol_value: float) -> bool:
    tol_type = (tol_type or "").lower().strip()
    if tol_type == "relative":
        return abs(student_val - answer_val) <= tol_value * abs(answer_val)
    return abs(student_val - answer_val) <= tol_value


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_assignments() -> dict:
    if not ASSIGNMENTS_FILE.exists():
        raise FileNotFoundError(f"Missing {ASSIGNMENTS_FILE}")
    return json.loads(ASSIGNMENTS_FILE.read_text(encoding="utf-8"))


@st.cache_data
def load_problem(problem_id: str) -> dict:
    path = PROBLEMS_DIR / f"{problem_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing problem file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_answer_key() -> Dict[Tuple[str, str], Dict[str, str]]:
    """
    Loads answer keys keyed by (problem_id, part_id).

    CSV columns required:
      problem_id, part_id, answer_value, answer_units, tolerance_type, tolerance_value
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


def grade_part(problem_id: str, part_id: str, student_text: str,
               answer_key: Dict[Tuple[str, str], Dict[str, str]]) -> Tuple[Optional[bool], str]:
    """
    Returns (is_correct, message). is_correct None means "not gradable" (no key).
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
        return None, "Answer key row invalid (check CSV)."

    ok = within_tolerance(student_val, ans_val, tol_type, tol_val)
    if ok:
        return True, f"Correct (within {tol_type} tolerance). Expected units: {ans_units}"
    return False, f"Incorrect. Expected units: {ans_units}"


# -----------------------------
# Database (SQLite) logging
# -----------------------------
def db_connect() -> sqlite3.Connection:
    safe_mkdir(LOGS_DIR)
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def db_init() -> None:
    safe_mkdir(LOGS_DIR)
    with db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY,
                created_utc TEXT NOT NULL,
                assignment TEXT NOT NULL,
                problem_id TEXT NOT NULL,
                was_submitted INTEGER NOT NULL,
                any_graded INTEGER NOT NULL,
                any_incorrect INTEGER NOT NULL
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attempt_parts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt_id TEXT NOT NULL,
                part_id TEXT NOT NULL,
                student_answer TEXT NOT NULL,
                is_correct INTEGER,
                message TEXT,
                FOREIGN KEY(attempt_id) REFERENCES attempts(attempt_id) ON DELETE CASCADE
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                extracted_text_len INTEGER NOT NULL,
                readable INTEGER NOT NULL,
                created_utc TEXT NOT NULL,
                FOREIGN KEY(attempt_id) REFERENCES attempts(attempt_id) ON DELETE CASCADE
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fallback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt_id TEXT NOT NULL,
                balance_equations TEXT,
                notes TEXT,
                created_utc TEXT NOT NULL,
                FOREIGN KEY(attempt_id) REFERENCES attempts(attempt_id) ON DELETE CASCADE
            );
        """)
        conn.commit()


def log_attempt(assignment: str, problem_id: str, any_graded: bool, any_incorrect: bool) -> str:
    attempt_id = str(uuid.uuid4())
    with db_connect() as conn:
        conn.execute("""
            INSERT INTO attempts (attempt_id, created_utc, assignment, problem_id, was_submitted, any_graded, any_incorrect)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (attempt_id, utc_now_iso(), assignment, problem_id, 1, int(any_graded), int(any_incorrect)))
        conn.commit()
    return attempt_id


def log_attempt_part(attempt_id: str, part_id: str, student_answer: str, is_correct: Optional[bool], message: str) -> None:
    with db_connect() as conn:
        conn.execute("""
            INSERT INTO attempt_parts (attempt_id, part_id, student_answer, is_correct, message)
            VALUES (?, ?, ?, ?, ?)
        """, (attempt_id, part_id, student_answer, None if is_correct is None else int(is_correct), message))
        conn.commit()


def list_attempts_for_problem(problem_id: str, limit: int = 10) -> List[Tuple[str, str, int, int]]:
    """
    Returns rows: (created_utc, attempt_id, any_incorrect, any_graded)
    """
    with db_connect() as conn:
        cur = conn.execute("""
            SELECT created_utc, attempt_id, any_incorrect, any_graded
            FROM attempts
            WHERE problem_id = ?
            ORDER BY created_utc DESC
            LIMIT ?
        """, (problem_id, limit))
        return list(cur.fetchall())


def get_attempt_parts(attempt_id: str) -> List[Tuple[str, str, Optional[int], str]]:
    """
    Returns rows: (part_id, student_answer, is_correct, message)
    """
    with db_connect() as conn:
        cur = conn.execute("""
            SELECT part_id, student_answer, is_correct, message
            FROM attempt_parts
            WHERE attempt_id = ?
            ORDER BY part_id ASC
        """, (attempt_id,))
        return list(cur.fetchall())


# -----------------------------
# Upload + PDF readable check
# -----------------------------
def try_extract_pdf_text(pdf_bytes: bytes) -> str:
    if not HAS_PYPDF2:
        return ""
    try:
        reader = PyPDF2.PdfReader(pdf_bytes)  # type: ignore
        text_chunks = []
        for page in reader.pages:
            t = page.extract_text() or ""
            text_chunks.append(t)
        return "\n".join(text_chunks).strip()
    except Exception:
        return ""


def save_upload(attempt_id: str, uploaded_file) -> Tuple[bool, int, str]:
    """
    Saves uploaded PDF into data/uploads/<attempt_id>/filename.
    Returns (readable, extracted_text_len, stored_path)
    """
    safe_mkdir(UPLOADS_DIR)
    attempt_dir = UPLOADS_DIR / attempt_id
    safe_mkdir(attempt_dir)

    filename = uploaded_file.name
    stored_path = attempt_dir / filename

    file_bytes = uploaded_file.getvalue()
    stored_path.write_bytes(file_bytes)

    extracted = try_extract_pdf_text(file_bytes)
    readable = bool(extracted and len(extracted) >= 30)  # simple confidence heuristic
    extracted_len = len(extracted)

    with db_connect() as conn:
        conn.execute("""
            INSERT INTO uploads (attempt_id, filename, stored_path, extracted_text_len, readable, created_utc)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (attempt_id, filename, str(stored_path), extracted_len, int(readable), utc_now_iso()))
        conn.commit()

    return readable, extracted_len, str(stored_path)


def save_fallback(attempt_id: str, balance_equations: str, notes: str) -> None:
    with db_connect() as conn:
        conn.execute("""
            INSERT INTO fallback (attempt_id, balance_equations, notes, created_utc)
            VALUES (?, ?, ?, ?)
        """, (attempt_id, balance_equations, notes, utc_now_iso()))
        conn.commit()


# -----------------------------
# Session state
# -----------------------------
def init_session_state() -> None:
    st.session_state.setdefault("selected_assignment", None)
    st.session_state.setdefault("selected_problem_id", None)

    # last submission state
    st.session_state.setdefault("last_attempt_id", None)
    st.session_state.setdefault("show_upload", False)
    st.session_state.setdefault("needs_fallback", False)


# -----------------------------
# UI: Sidebar navigation
# -----------------------------
def render_sidebar(assignments: dict) -> None:
    st.sidebar.title("Navigation")
    assignment_names = list(assignments.keys())
    if not assignment_names:
        st.sidebar.error("No assignments found in data/assignments.json.")
        st.stop()

    default_a = 0
    if st.session_state["selected_assignment"] in assignment_names:
        default_a = assignment_names.index(st.session_state["selected_assignment"])

    selected_assignment = st.sidebar.selectbox("Select Assignment", assignment_names, index=default_a)
    st.session_state["selected_assignment"] = selected_assignment

    problem_ids = assignments.get(selected_assignment, [])
    if not problem_ids:
        st.sidebar.warning("No problems listed for this assignment.")
        st.stop()

    default_p = 0
    if st.session_state["selected_problem_id"] in problem_ids:
        default_p = problem_ids.index(st.session_state["selected_problem_id"])

    selected_problem_id = st.sidebar.selectbox(
        "Select Problem", problem_ids, index=default_p,
        help="Students choose from a list to avoid mistyping IDs."
    )
    st.session_state["selected_problem_id"] = selected_problem_id

    st.sidebar.divider()
    st.sidebar.caption("Instructor demo build: logging + uploads + fallback form (no AI yet).")


def render_attempt_history(problem_id: str) -> None:
    st.sidebar.subheader("Recent Attempts")
    rows = list_attempts_for_problem(problem_id, limit=8)
    if not rows:
        st.sidebar.caption("No attempts logged yet.")
        return

    labels = []
    attempt_ids = []
    for created_utc, attempt_id, any_incorrect, any_graded in rows:
        badge = "❌" if any_incorrect else "✅"
        labels.append(f"{badge} {created_utc}  ({attempt_id[:8]})")
        attempt_ids.append(attempt_id)

    selected_idx = st.sidebar.selectbox("View attempt", list(range(len(labels))), format_func=lambda i: labels[i])
    selected_attempt_id = attempt_ids[selected_idx]

    with st.sidebar.expander("Attempt details", expanded=False):
        parts = get_attempt_parts(selected_attempt_id)
        for part_id, ans, is_corr, msg in parts:
            if is_corr is None:
                st.write(f"Part ({part_id}): {ans} — not graded")
            elif is_corr == 1:
                st.write(f"Part ({part_id}): ✅ {ans}")
            else:
                st.write(f"Part ({part_id}): ❌ {ans}")
        st.caption(f"Attempt ID: {selected_attempt_id}")


# -----------------------------
# UI: Main problem + submission
# -----------------------------
def render_problem(problem: Dict[str, Any], assignment_name: str,
                   answer_key: Dict[Tuple[str, str], Dict[str, str]]) -> None:

    pid = problem.get("problem_id", "")
    title = problem.get("title", pid or "Problem")
    statement = problem.get("statement", "_No statement provided._")

    st.markdown(f"## {title}")
    st.markdown(f"**Problem ID:** `{pid}`")
    st.markdown(statement.replace("\n", "  \n"))

    parts = problem.get("parts", [])
    if not parts:
        st.warning("This problem has no 'parts'. Add numeric parts to the problem JSON.")
        return

    st.divider()
    st.subheader("Submit Answers (numeric)")

    responses: Dict[str, str] = {}
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

    if not submitted:
        return

    # Grade parts (if key exists)
    results: Dict[str, Tuple[Optional[bool], str]] = {}
    any_incorrect = False
    any_graded = False

    for p in parts:
        part_id = str(p.get("part_id", "")).strip() or "?"
        student_text = responses.get(part_id, "")
        is_correct, msg = grade_part(pid, part_id, student_text, answer_key)
        results[part_id] = (is_correct, msg)

        if is_correct is not None:
            any_graded = True
            if not is_correct:
                any_incorrect = True
        else:
            # not graded parts shouldn't trigger upload flow
            pass

    # Log attempt + parts
    attempt_id = log_attempt(assignment_name, pid, any_graded=any_graded, any_incorrect=any_incorrect)
    st.session_state["last_attempt_id"] = attempt_id

    for part_id, student_text in responses.items():
        is_correct, msg = results.get(part_id, (None, ""))
        log_attempt_part(attempt_id, part_id, student_text, is_correct, msg)

    st.success(f"Submission logged. Attempt ID: {attempt_id[:8]}")

    st.subheader("Results")
    for p in parts:
        part_id = str(p.get("part_id", "")).strip() or "?"
        is_correct, msg = results.get(part_id, (None, "No result"))
        if is_correct is None:
            st.info(f"Part ({part_id}): {msg}")
        elif is_correct:
            st.success(f"Part ({part_id}): {msg}")
        else:
            st.error(f"Part ({part_id}): {msg}")

    # Show upload workflow only if at least one graded part is incorrect
    st.session_state["show_upload"] = bool(any_incorrect)
    st.session_state["needs_fallback"] = False

    if any_incorrect:
        st.warning("One or more parts are incorrect. Upload your work to receive feedback (next phase: AI).")
        render_upload_and_fallback(problem_id=pid, attempt_id=attempt_id)
    else:
        st.info("All graded parts are correct. No upload needed.")


def render_upload_and_fallback(problem_id: str, attempt_id: str) -> None:
    st.divider()
    st.subheader("Upload Work (PDF)")

    st.caption("Upload your handwritten/typed work as a PDF. If it can't be read, you'll be prompted for extra information.")
    uploaded = st.file_uploader("Upload PDF", type=["pdf"], key=f"{attempt_id}_upload")

    readable = None
    extracted_len = 0
    stored_path = ""

    if uploaded is not None:
        readable, extracted_len, stored_path = save_upload(attempt_id, uploaded)
        if readable:
            st.success(f"Uploaded and readable (extracted text length: {extracted_len}). Stored at: {stored_path}")
        else:
            st.warning(
                "Could not confidently read your work from the PDF. "
                "Please provide the additional information below."
            )
            st.session_state["needs_fallback"] = True

    # Always allow fallback (even if readable) for demo robustness
    with st.expander("Additional information (fallback form)", expanded=bool(st.session_state["needs_fallback"])):
        st.write("Paste the key balances or equations you used, and any notes that explain your setup.")
        balance = st.text_area("Balances / equations (text)", height=140, key=f"{attempt_id}_fallback_balance")
        notes = st.text_area("Notes (what you tried / where you got stuck)", height=120, key=f"{attempt_id}_fallback_notes")

        if st.button("Save additional info", key=f"{attempt_id}_save_fallback"):
            save_fallback(attempt_id, balance_equations=balance, notes=notes)
            st.success("Saved. (Next phase will use this + your PDF to generate feedback.)")


def render_dev_notes() -> None:
    with st.expander("Developer notes (repo structure + files)", expanded=False):
        st.code(
            "meb_tutor_app/\n"
            "  app.py\n"
            "  data/\n"
            "    assignments.json\n"
            "    answer_key.csv\n"
            "    problems/\n"
            "      MEB_001.json\n"
            "      MEB_002.json\n"
            "    uploads/\n"
            "    logs/\n"
            "      attempts.sqlite3\n",
            language="text",
        )
        st.write(
            "• 'logs/attempts.sqlite3' stores attempts, part answers, uploads, and fallback text.\n"
            "• 'uploads/<attempt_id>/' stores uploaded PDFs.\n"
            "• PyPDF2 is optional; if not installed, PDFs will usually trigger fallback form.\n"
        )


# -----------------------------
# App entry
# -----------------------------
st.set_page_config(page_title="MEB Tutor (Instructor Demo)", layout="wide")
init_session_state()

safe_mkdir(DATA_DIR)
safe_mkdir(PROBLEMS_DIR)
safe_mkdir(UPLOADS_DIR)
safe_mkdir(LOGS_DIR)
db_init()

st.title("MEB Homework Tutor — Instructor Demo")
st.caption("Now includes: attempt logging + PDF uploads + fallback form (no AI yet).")

# Load assignments
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

assignment_name = st.session_state["selected_assignment"]
problem_id = st.session_state["selected_problem_id"]

# Attempt history in sidebar
render_attempt_history(problem_id)

# Load selected problem
try:
    problem = load_problem(problem_id)
except Exception as e:
    st.error(
        f"Could not load problem '{problem_id}'.\n\n"
        f"Expected file at {PROBLEMS_DIR}/{problem_id}.json\n\n"
        f"Error: {e}"
    )
    st.stop()

# Load answer key (optional)
try:
    answer_key = load_answer_key()
except Exception as e:
    st.warning(f"Answer key exists but could not be loaded: {e}")
    answer_key = {}

render_problem(problem, assignment_name, answer_key)
render_dev_notes()
