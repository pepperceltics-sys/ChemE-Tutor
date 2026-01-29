# app.py
# Instructor-demo-ready build (per-part uploads + persistent "upload successful" message):
# - Assignment -> Problem navigation
# - Numeric answer entry per part
# - CSV grading with tolerance
# - Attempt logging (SQLite)
# - Upload workflow PER PART (PDF) shown only when that part is incorrect
# - Persistent "✅ Upload successful..." message per part (survives Streamlit reruns)
# - Fallback form PER PART if PDF can't be read (or if student prefers)
#
# File structure:
# meb_tutor_app/
#   app.py
#   data/
#     assignments.json
#     answer_key.csv
#     problems/
#       MEB_001.json
#       MEB_002.json
#     uploads/
#       .gitkeep
#     logs/
#       .gitkeep
#
# Run locally:
#   pip install streamlit
#   streamlit run app.py

import csv
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import streamlit as st

# Optional typed-PDF extraction
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


def upload_state_key(attempt_id: str, part_id: str) -> str:
    """
    Session-state key that tracks whether a PDF upload succeeded
    for (attempt_id, part_id), so we can show a persistent confirmation.
    """
    return f"uploaded_{attempt_id}_{part_id}"


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
# Database logging
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
                problem_id TEXT NOT NULL
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
                part_id TEXT NOT NULL,
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
                part_id TEXT NOT NULL,
                balance_equations TEXT,
                notes TEXT,
                created_utc TEXT NOT NULL,
                FOREIGN KEY(attempt_id) REFERENCES attempts(attempt_id) ON DELETE CASCADE
            );
        """)
        conn.commit()


def log_attempt(assignment: str, problem_id: str) -> str:
    attempt_id = str(uuid.uuid4())
    with db_connect() as conn:
        conn.execute("""
            INSERT INTO attempts (attempt_id, created_utc, assignment, problem_id)
            VALUES (?, ?, ?, ?)
        """, (attempt_id, utc_now_iso(), assignment, problem_id))
        conn.commit()
    return attempt_id


def log_attempt_part(attempt_id: str, part_id: str, student_answer: str, is_correct: Optional[bool], message: str) -> None:
    with db_connect() as conn:
        conn.execute("""
            INSERT INTO attempt_parts (attempt_id, part_id, student_answer, is_correct, message)
            VALUES (?, ?, ?, ?, ?)
        """, (attempt_id, part_id, student_answer, None if is_correct is None else int(is_correct), message))
        conn.commit()


def list_attempts_for_problem(problem_id: str, limit: int = 10) -> List[Tuple[str, str]]:
    with db_connect() as conn:
        cur = conn.execute("""
            SELECT created_utc, attempt_id
            FROM attempts
            WHERE problem_id = ?
            ORDER BY created_utc DESC
            LIMIT ?
        """, (problem_id, limit))
        return list(cur.fetchall())


def get_attempt_parts(attempt_id: str) -> List[Tuple[str, str, Optional[int], str]]:
    with db_connect() as conn:
        cur = conn.execute("""
            SELECT part_id, student_answer, is_correct, message
            FROM attempt_parts
            WHERE attempt_id = ?
            ORDER BY part_id ASC
        """, (attempt_id,))
        return list(cur.fetchall())


# -----------------------------
# Uploads + readability
# -----------------------------
def try_extract_pdf_text(pdf_bytes: bytes) -> str:
    if not HAS_PYPDF2:
        return ""
    try:
        reader = PyPDF2.PdfReader(pdf_bytes)  # type: ignore
        text_chunks = []
        for page in reader.pages:
            text_chunks.append(page.extract_text() or "")
        return "\n".join(text_chunks).strip()
    except Exception:
        return ""


def save_upload(attempt_id: str, part_id: str, uploaded_file) -> Tuple[bool, int, str]:
    """
    Save PDF to: data/uploads/<attempt_id>/<part_id>/<filename>
    Returns: (readable, extracted_text_len, stored_path)
    """
    safe_mkdir(UPLOADS_DIR)
    part_dir = UPLOADS_DIR / attempt_id / part_id
    safe_mkdir(part_dir)

    filename = uploaded_file.name
    stored_path = part_dir / filename

    file_bytes = uploaded_file.getvalue()
    stored_path.write_bytes(file_bytes)

    extracted = try_extract_pdf_text(file_bytes)
    readable = bool(extracted and len(extracted) >= 30)
    extracted_len = len(extracted)

    with db_connect() as conn:
        conn.execute("""
            INSERT INTO uploads (attempt_id, part_id, filename, stored_path, extracted_text_len, readable, created_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (attempt_id, part_id, filename, str(stored_path), extracted_len, int(readable), utc_now_iso()))
        conn.commit()

    return readable, extracted_len, str(stored_path)


def save_fallback(attempt_id: str, part_id: str, balance_equations: str, notes: str) -> None:
    with db_connect() as conn:
        conn.execute("""
            INSERT INTO fallback (attempt_id, part_id, balance_equations, notes, created_utc)
            VALUES (?, ?, ?, ?, ?)
        """, (attempt_id, part_id, balance_equations, notes, utc_now_iso()))
        conn.commit()


# -----------------------------
# Session state
# -----------------------------
def init_session_state() -> None:
    st.session_state.setdefault("selected_assignment", None)
    st.session_state.setdefault("selected_problem_id", None)
    st.session_state.setdefault("last_attempt_id", None)


# -----------------------------
# UI: Sidebar navigation
# -----------------------------
def render_sidebar(assignments: dict) -> None:
    st.sidebar.title("Navigation")
    names = list(assignments.keys())
    if not names:
        st.sidebar.error("No assignments found.")
        st.stop()

    default_a = 0
    if st.session_state["selected_assignment"] in names:
        default_a = names.index(st.session_state["selected_assignment"])

    st.session_state["selected_assignment"] = st.sidebar.selectbox("Assignment", names, index=default_a)

    pids = assignments.get(st.session_state["selected_assignment"], [])
    if not pids:
        st.sidebar.warning("No problems in this assignment.")
        st.stop()

    default_p = 0
    if st.session_state["selected_problem_id"] in pids:
        default_p = pids.index(st.session_state["selected_problem_id"])

    st.session_state["selected_problem_id"] = st.sidebar.selectbox("Problem", pids, index=default_p)
    st.sidebar.divider()


def render_attempt_history(problem_id: str) -> None:
    st.sidebar.subheader("Recent Attempts")
    rows = list_attempts_for_problem(problem_id, limit=8)
    if not rows:
        st.sidebar.caption("No attempts yet.")
        return

    labels = []
    ids = []
    for created_utc, attempt_id in rows:
        labels.append(f"{created_utc} ({attempt_id[:8]})")
        ids.append(attempt_id)

    idx = st.sidebar.selectbox("View", list(range(len(labels))), format_func=lambda i: labels[i])
    attempt_id = ids[idx]
    with st.sidebar.expander("Attempt details", expanded=False):
        parts = get_attempt_parts(attempt_id)
        for part_id, ans, is_corr, _msg in parts:
            if is_corr is None:
                st.write(f"Part ({part_id}): {ans} — not graded")
            elif is_corr == 1:
                st.write(f"Part ({part_id}): ✅ {ans}")
            else:
                st.write(f"Part ({part_id}): ❌ {ans}")
        st.caption(f"Attempt ID: {attempt_id}")


# -----------------------------
# UI: Main problem
# -----------------------------
def render_problem(problem: Dict[str, Any], assignment: str, answer_key: Dict[Tuple[str, str], Dict[str, str]]) -> None:
    pid = problem.get("problem_id", "")
    title = problem.get("title", pid)
    statement = problem.get("statement", "")

    st.markdown(f"## {title}")
    st.markdown(f"**Problem ID:** `{pid}`")
    st.markdown(statement.replace("\n", "  \n"))

    parts = problem.get("parts", [])
    if not parts:
        st.warning("This problem has no numeric parts. Add 'parts' to the JSON.")
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
        responses[part_id] = st.text_input(label, key=f"{pid}_{part_id}_answer")

    submitted = st.button("Submit", key=f"{pid}_submit")
    if not submitted:
        return

    # Create attempt and log parts
    attempt_id = log_attempt(assignment, pid)
    st.session_state["last_attempt_id"] = attempt_id

    st.success(f"Submission logged. Attempt ID: {attempt_id[:8]}")
    st.subheader("Results")

    results: Dict[str, Tuple[Optional[bool], str]] = {}
    for p in parts:
        part_id = str(p.get("part_id", "")).strip() or "?"
        is_correct, msg = grade_part(pid, part_id, responses.get(part_id, ""), answer_key)
        results[part_id] = (is_correct, msg)

        log_attempt_part(attempt_id, part_id, responses.get(part_id, ""), is_correct, msg)

        if is_correct is None:
            st.info(f"Part ({part_id}): {msg}")
        elif is_correct:
            st.success(f"Part ({part_id}): {msg}")
        else:
            st.error(f"Part ({part_id}): {msg}")

    # Per-part uploads for incorrect parts
    incorrect_parts = [part_id for part_id, (ok, _msg) in results.items() if ok is False]
    if incorrect_parts:
        st.warning("Upload your work for the parts you missed (one PDF per part).")
        render_per_part_uploads(attempt_id, incorrect_parts)
    else:
        st.info("All graded parts are correct. No uploads needed.")


def render_per_part_uploads(attempt_id: str, incorrect_parts: List[str]) -> None:
    st.divider()
    st.subheader("Upload Work (Per Part)")

    st.caption("Upload a PDF for each incorrect part. If it can't be read, complete the fallback form for that part.")

    for part_id in incorrect_parts:
        st.markdown(f"### Part ({part_id}) — Upload")

        state_key = upload_state_key(attempt_id, part_id)

        # ✅ Always show persistent success message if uploaded already
        if st.session_state.get(state_key) is True:
            st.success("✅ Upload successful. Your work has been saved.")
        else:
            uploaded = st.file_uploader(
                f"Upload PDF for Part ({part_id})",
                type=["pdf"],
                key=f"{attempt_id}_{part_id}_pdf"
            )

            if uploaded is not None:
                readable, _extracted_len, _stored_path = save_upload(attempt_id, part_id, uploaded)

                # ✅ Persist upload success so message survives reruns
                st.session_state[state_key] = True

                # ✅ Immediate feedback (even if reruns, the block above will still show it)
                st.success("✅ Upload successful. Your work has been saved.")

                if not readable:
                    st.warning(
                        "⚠️ We could not confidently read this PDF. "
                        "Please complete the fallback form below."
                    )

        # Fallback form (always accessible)
        with st.expander(f"Fallback form for Part ({part_id})", expanded=False):
            balance = st.text_area(
                "Paste the balance(s)/equation(s) you used (text)",
                height=120,
                key=f"{attempt_id}_{part_id}_fallback_balance"
            )
            notes = st.text_area(
                "Notes (what you tried / where you think the mistake is)",
                height=100,
                key=f"{attempt_id}_{part_id}_fallback_notes"
            )
            if st.button("Save fallback info", key=f"{attempt_id}_{part_id}_save_fallback"):
                save_fallback(attempt_id, part_id, balance, notes)
                st.success("✅ Additional information saved.")


# -----------------------------
# App entry
# -----------------------------
st.set_page_config(page_title="MEB Tutor (Per-Part Uploads)", layout="wide")
init_session_state()

safe_mkdir(DATA_DIR)
safe_mkdir(PROBLEMS_DIR)
safe_mkdir(UPLOADS_DIR)
safe_mkdir(LOGS_DIR)
db_init()

st.title("MEB Homework Tutor — Instructor Demo (Per-Part Uploads)")
st.caption("Per-part uploads now show a persistent '✅ Upload successful' message.")

# Load assignments
try:
    assignments = load_assignments()
except Exception as e:
    st.error(f"Could not load assignments at {ASSIGNMENTS_FILE}.\n\nError: {e}")
    st.stop()

render_sidebar(assignments)

assignment = st.session_state["selected_assignment"]
problem_id = st.session_state["selected_problem_id"]

# Sidebar attempt history
render_attempt_history(problem_id)

# Load selected problem
try:
    problem = load_problem(problem_id)
except Exception as e:
    st.error(f"Could not load problem '{problem_id}'.\n\nError: {e}")
    st.stop()

# Load answer key
try:
    answer_key = load_answer_key()
except Exception as e:
    st.warning(f"Answer key exists but could not be loaded: {e}")
    answer_key = {}

render_problem(problem, assignment, answer_key)
