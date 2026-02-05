# app.py
# MEB Homework Tutor — Streamlit Cloud stable upload UX (NO AI YET)
#
# Goal (per your request):
# ✅ Students submit numeric answers (graded via CSV tolerance)
# ✅ If incorrect, they can upload a PDF PER PART
# ✅ Upload does NOT "disappear" after upload (Streamlit rerun-safe)
# ✅ Verify upload worked: persistent success panel + filename/size/SHA256 + download button
# ✅ Sidebar shows uploaded files captured in this session
#
# Key fix vs your current behavior:
# ✅ Persist the "active attempt" (attempt_id + incorrect parts) in st.session_state
# so the upload UI keeps rendering after Streamlit reruns.
#
# NEW FIX (for unit-less parts not loading):
# ✅ Make expected_output + units parsing type-safe (handles missing/None/wrong-type)
# ✅ Don’t print “Expected units:” when units are blank in the answer key

import csv
import json
import sqlite3
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import streamlit as st

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent

# Repo data (committed to GitHub)
DATA_DIR = BASE_DIR / "data"
PROBLEMS_DIR = DATA_DIR / "problems"
ASSIGNMENTS_FILE = DATA_DIR / "assignments.json"
ANSWER_KEY_FILE = DATA_DIR / "answer_key.csv"

# Runtime data (writable on Streamlit Cloud)
RUNTIME_DIR = Path("/tmp/cheme_tutor_runtime")
LOGS_DIR = RUNTIME_DIR / "logs"
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


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def upload_state_key(attempt_id: str, part_id: str) -> str:
    return f"uploaded_{attempt_id}_{part_id}"


# -----------------------------
# In-memory upload store (proof of upload)
# -----------------------------
def store_upload_in_memory(attempt_id: str, part_id: str, uploaded_file) -> Dict[str, Any]:
    file_bytes = uploaded_file.getvalue()
    info = {
        "filename": uploaded_file.name,
        "bytes": file_bytes,
        "size": len(file_bytes),
        "sha256": _sha256(file_bytes),
        "uploaded_utc": utc_now_iso(),
    }
    st.session_state["uploaded_files"].setdefault(attempt_id, {})
    st.session_state["uploaded_files"][attempt_id][part_id] = info
    st.session_state[upload_state_key(attempt_id, part_id)] = True
    return info


def get_upload_from_memory(attempt_id: str, part_id: str) -> Optional[Dict[str, Any]]:
    return st.session_state.get("uploaded_files", {}).get(attempt_id, {}).get(part_id)


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
            raise ValueError(f"{ANSWER_KEY_FILE} must include columns: {', '.join(sorted(required))}")

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
    ans_units = (k.get("answer_units", "") or "").strip()

    if ans_val is None or tol_val is None:
        return None, "Answer key row invalid (check CSV)."

    ok = within_tolerance(student_val, ans_val, tol_type, tol_val)

    # ✅ Only show units text if units are non-empty
    units_msg = f" Expected units: {ans_units}" if ans_units else ""

    if ok:
        return True, f"Correct (within {tol_type} tolerance).{units_msg}"
    return False, f"Incorrect.{units_msg}"


# -----------------------------
# Database (attempt logging only)
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


def log_attempt_part(attempt_id: str, part_id: str, student_answer: str,
                     is_correct: Optional[bool], message: str) -> None:
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
# Session state
# -----------------------------
def init_session_state() -> None:
    st.session_state.setdefault("selected_assignment", None)
    st.session_state.setdefault("selected_problem_id", None)

    # Captured uploads for this session: uploaded_files[attempt_id][part_id] = info
    st.session_state.setdefault("uploaded_files", {})

    # ✅ Persist the attempt context so upload UI survives reruns
    # active_attempt = {attempt_id, problem_id, assignment, incorrect_parts, results}
    st.session_state.setdefault("active_attempt", None)


# -----------------------------
# Sidebar UI
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


def sidebar_tab_attempt_history(problem_id: str) -> None:
    st.sidebar.subheader("Recent Attempts")
    rows = list_attempts_for_problem(problem_id, limit=8)
    if not rows:
        st.sidebar.caption("No attempts yet.")
        return

    labels, ids = [], []
    for created_utc, attempt_id in rows:
        labels.append(f"{created_utc} ({attempt_id[:8]})")
        ids.append(attempt_id)

    idx = st.sidebar.selectbox("View attempt", list(range(len(labels))), format_func=lambda i: labels[i])
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


def sidebar_tab_uploaded_files_in_memory() -> None:
    st.sidebar.subheader("Uploaded Files (This Session)")

    if st.sidebar.button("🔄 Refresh view"):
        st.rerun()

    uploaded_files = st.session_state.get("uploaded_files", {})
    if not uploaded_files:
        st.sidebar.caption("No uploads captured in this session yet.")
        st.sidebar.caption("Note: In-memory uploads reset if Streamlit restarts.")
        return

    flat = []
    for attempt_id, parts in uploaded_files.items():
        for part_id, info in parts.items():
            flat.append((attempt_id, part_id, info))

    flat.sort(key=lambda t: t[2].get("uploaded_utc", ""), reverse=True)

    options = [
        f"{info.get('uploaded_utc','')} | attempt {attempt_id[:8]} | part {part_id} | {info.get('filename','')}"
        for attempt_id, part_id, info in flat
    ]

    pick = st.sidebar.selectbox("Select upload", list(range(len(options))), format_func=lambda i: options[i])
    attempt_id, part_id, info = flat[pick]

    with st.sidebar.expander("Upload details", expanded=True):
        st.write(f"**Attempt:** {attempt_id}")
        st.write(f"**Part:** {part_id}")
        st.write(f"**Filename:** {info['filename']}")
        st.write(f"**Size:** {info['size']} bytes")
        st.write(f"**SHA256:** `{info['sha256']}`")
        st.write(f"**Uploaded (UTC):** {info['uploaded_utc']}")

        st.download_button(
            label="⬇️ Download uploaded PDF",
            data=info["bytes"],
            file_name=info["filename"],
            mime="application/pdf",
            key=f"sidebar_dl_{attempt_id}_{part_id}",
        )


# -----------------------------
# Main UI
# -----------------------------
def render_per_part_uploads_in_memory(attempt_id: str, incorrect_parts: List[str]) -> None:
    st.divider()
    st.subheader("Upload Work (Per Part)")
    st.caption("Uploads are stored in memory only for verification (no disk storage yet).")

    for part_id in incorrect_parts:
        st.markdown(f"### Part ({part_id}) — Upload")

        existing = get_upload_from_memory(attempt_id, part_id)
        if existing is not None:
            st.success("✅ Upload received (stored in memory).")
            st.write(f"**File:** {existing['filename']}")
            st.write(f"**Size:** {existing['size']} bytes")
            st.write(f"**SHA256:** `{existing['sha256']}`")
            st.write(f"**Uploaded (UTC):** {existing['uploaded_utc']}")

            st.download_button(
                label="⬇️ Download the uploaded PDF (proof)",
                data=existing["bytes"],
                file_name=existing["filename"],
                mime="application/pdf",
                key=f"dl_mem_{attempt_id}_{part_id}",
            )

            if st.button("Replace upload", key=f"replace_{attempt_id}_{part_id}"):
                st.session_state["uploaded_files"].setdefault(attempt_id, {})
                st.session_state["uploaded_files"][attempt_id].pop(part_id, None)
                st.session_state.pop(upload_state_key(attempt_id, part_id), None)
                st.rerun()
        else:
            uploaded = st.file_uploader(
                f"Upload PDF for Part ({part_id})",
                type=["pdf"],
                key=f"uploader_{attempt_id}_{part_id}",
            )
            if uploaded is not None:
                info = store_upload_in_memory(attempt_id, part_id, uploaded)

                st.success("✅ Upload received (stored in memory).")
                st.write(f"**File:** {info['filename']}")
                st.write(f"**Size:** {info['size']} bytes")
                st.write(f"**SHA256:** `{info['sha256']}`")

                st.download_button(
                    label="⬇️ Download the uploaded PDF (proof)",
                    data=info["bytes"],
                    file_name=info["filename"],
                    mime="application/pdf",
                    key=f"dl_mem_{attempt_id}_{part_id}",
                )


def render_problem(problem: Dict[str, Any], assignment: str,
                   answer_key: Dict[Tuple[str, str], Dict[str, str]]) -> None:
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

        # ✅ Robust handling: expected_output can be missing/None/wrong-type
        expected_raw = p.get("expected_output", {})
        expected: Dict[str, Any] = expected_raw if isinstance(expected_raw, dict) else {}

        units_raw = expected.get("units", "")
        units = (str(units_raw).strip() if units_raw is not None else "")

        st.markdown(f"### Part ({part_id})")
        if prompt:
            st.write(prompt)

        label = "Answer"
        if units:
            label = f"Answer ({units})"
        responses[part_id] = st.text_input(label, key=f"{pid}_{part_id}_answer")

    submitted = st.button("Submit", key=f"{pid}_submit")

    if submitted:
        attempt_id = log_attempt(assignment, pid)
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

        incorrect_parts = [part_id for part_id, (ok, _msg) in results.items() if ok is False]

        st.session_state["active_attempt"] = {
            "attempt_id": attempt_id,
            "problem_id": pid,
            "assignment": assignment,
            "incorrect_parts": incorrect_parts,
            "results": {k: [v[0], v[1]] for k, v in results.items()},
        }

        if incorrect_parts:
            st.warning("Upload your work for the parts you missed (one PDF per part).")
            render_per_part_uploads_in_memory(attempt_id, incorrect_parts)
        else:
            st.info("All graded parts are correct. No uploads needed.")

    active = st.session_state.get("active_attempt")
    if active and active.get("problem_id") == pid:
        attempt_id_active = active.get("attempt_id")
        incorrect_parts_active = active.get("incorrect_parts", [])
        if attempt_id_active and incorrect_parts_active:
            st.warning("Continue uploading your work for the parts you missed (attempt persists across reruns).")
            render_per_part_uploads_in_memory(attempt_id_active, incorrect_parts_active)


# -----------------------------
# App entry
# -----------------------------
st.set_page_config(page_title="MEB Tutor (Stable Uploads)", layout="wide")
init_session_state()

safe_mkdir(DATA_DIR)
safe_mkdir(PROBLEMS_DIR)
safe_mkdir(RUNTIME_DIR)
safe_mkdir(LOGS_DIR)
db_init()

st.title("MEB Homework Tutor — Stable Uploads (Streamlit Cloud)")
st.caption("Upload boxes persist after upload reruns. Uploads stored in-memory for verification only.")

try:
    assignments = load_assignments()
except Exception as e:
    st.error(f"Could not load assignments at {ASSIGNMENTS_FILE}.\n\nError: {e}")
    st.stop()

render_sidebar(assignments)

assignment = st.session_state["selected_assignment"]
problem_id = st.session_state["selected_problem_id"]

tab1, tab2 = st.sidebar.tabs(["Attempt history", "Uploaded files"])
with tab1:
    sidebar_tab_attempt_history(problem_id)
with tab2:
    sidebar_tab_uploaded_files_in_memory()

try:
    problem = load_problem(problem_id)
except Exception as e:
    st.error(f"Could not load problem '{problem_id}'.\n\nError: {e}")
    st.stop()

try:
    answer_key = load_answer_key()
except Exception as e:
    st.warning(f"Answer key exists but could not be loaded: {e}")
    answer_key = {}

render_problem(problem, assignment, answer_key)
