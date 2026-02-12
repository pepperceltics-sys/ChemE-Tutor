# app.py
# MEB Tutor — stable submit + uploads + AI-read pipeline (PDF -> fallback -> feedback-ready)
#
# ✅ Includes (merged):
# - Persisted "active attempt" so uploads/results don't disappear on rerun
# - Uploads saved to disk + SQLite; sidebar shows uploaded files
# - Uploads also tracked in session memory (for instant sidebar updates)
# - Stores extracted PDF text in DB (with safe migration)
# - Readability detection (FIXED: not everything becomes unreadable)
# - Fallback form auto-expands when PDF is unreadable
# - “Get AI feedback” per part (stubbed; replace with real model call)
# - Unitless + legacy expected_output support (dict / str / list-of-dicts)
# - Nomenclature hint derived from problem["tutor_layer"]["expected_setup"] (if present)

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
    return f"uploaded_ok_{attempt_id}_{part_id}"


def mem_store_key(problem_id: str) -> str:
    return f"mem_uploads_{problem_id}"


def ai_feedback_key(attempt_id: str, part_id: str) -> str:
    return f"ai_feedback_{attempt_id}_{part_id}"


def extract_units_from_expected_output(expected_raw: Any) -> str:
    """
    Supports:
      - dict: {"name": "...", "units": "..."}
      - str:  "kmol/hr" (treat as units)
      - list: [{"name": "...", "units": "..."}] (use first dict's units)
      - anything else: unitless
    """
    if isinstance(expected_raw, dict):
        return (expected_raw.get("units") or "").strip()
    if isinstance(expected_raw, str):
        return expected_raw.strip()
    if isinstance(expected_raw, list) and expected_raw and isinstance(expected_raw[0], dict):
        return (expected_raw[0].get("units") or "").strip()
    return ""


# ✅ FIXED: readability heuristic (not overly strict)
def compute_readable(extracted_text: str) -> bool:
    """
    Better Phase-1 readability heuristic for typed PDFs:
    - must have enough text
    - must have enough letters/digits (not just whitespace)
    - equation marker helps, but not required
    """
    t = (extracted_text or "").strip()
    if len(t) < 30:
        return False

    alpha = sum(1 for c in t if c.isalpha())
    alnum = sum(1 for c in t if c.isalnum())
    eq = "=" in t

    return eq or (alpha >= 15 and (alnum / max(len(t), 1)) > 0.20)


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
            raise ValueError(f"{ANSWER_KEY_FILE} must include: {', '.join(sorted(required))}")

        for row in reader:
            pid = (row.get("problem_id") or "").strip()
            part_id = (row.get("part_id") or "").strip()
            if not pid or not part_id:
                continue
            key[(pid, part_id)] = {k: (v or "").strip() for k, v in row.items()}

    return key


# -----------------------------
# Grading (unitless-safe)
# -----------------------------
def grade_part(
    problem_id: str,
    part_id: str,
    student_text: str,
    answer_key: Dict[Tuple[str, str], Dict[str, str]],
) -> Tuple[Optional[bool], str]:
    k = answer_key.get((problem_id, part_id))
    if not k:
        return None, "No answer key for this part (not graded yet)."

    student_val = parse_float(student_text.strip())
    if student_val is None:
        return False, "Please enter a numeric value."

    ans_val = parse_float(k.get("answer_value", ""))
    tol_val = parse_float(k.get("tolerance_value", ""))
    tol_type = (k.get("tolerance_type") or "absolute").strip()
    ans_units = (k.get("answer_units") or "").strip()

    if ans_val is None or tol_val is None:
        return None, "Answer key row invalid (check CSV)."

    ok = within_tolerance(student_val, ans_val, tol_type, tol_val)
    units_msg = f" Expected units: {ans_units}" if ans_units else ""

    if ok:
        return True, f"Correct (within {tol_type} tolerance).{units_msg}"
    return False, f"Incorrect.{units_msg}"


# -----------------------------
# Database (SQLite) logging + migration
# -----------------------------
def db_connect() -> sqlite3.Connection:
    safe_mkdir(LOGS_DIR)
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def db_has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table});")
    cols = [r[1] for r in cur.fetchall()]
    return column in cols


def db_init_and_migrate() -> None:
    safe_mkdir(LOGS_DIR)
    with db_connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY,
                created_utc TEXT NOT NULL,
                assignment TEXT NOT NULL,
                problem_id TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attempt_parts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt_id TEXT NOT NULL,
                part_id TEXT NOT NULL,
                student_answer TEXT NOT NULL,
                is_correct INTEGER,
                message TEXT,
                FOREIGN KEY(attempt_id) REFERENCES attempts(attempt_id) ON DELETE CASCADE
            );
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt_id TEXT NOT NULL,
                part_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                stored_path TEXT NOT NULL,
                extracted_text TEXT,
                extracted_text_len INTEGER NOT NULL,
                readable INTEGER NOT NULL,
                created_utc TEXT NOT NULL,
                FOREIGN KEY(attempt_id) REFERENCES attempts(attempt_id) ON DELETE CASCADE
            );
            """
        )
        # migrate older DBs
        if not db_has_column(conn, "uploads", "extracted_text"):
            conn.execute("ALTER TABLE uploads ADD COLUMN extracted_text TEXT;")
        conn.commit()

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fallback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                attempt_id TEXT NOT NULL,
                part_id TEXT NOT NULL,
                balance_equations TEXT,
                notes TEXT,
                created_utc TEXT NOT NULL,
                FOREIGN KEY(attempt_id) REFERENCES attempts(attempt_id) ON DELETE CASCADE
            );
            """
        )
        conn.commit()


def log_attempt(assignment: str, problem_id: str) -> str:
    attempt_id = str(uuid.uuid4())
    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO attempts (attempt_id, created_utc, assignment, problem_id)
            VALUES (?, ?, ?, ?)
            """,
            (attempt_id, utc_now_iso(), assignment, problem_id),
        )
        conn.commit()
    return attempt_id


def log_attempt_part(attempt_id: str, part_id: str, student_answer: str, is_correct: Optional[bool], message: str) -> None:
    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO attempt_parts (attempt_id, part_id, student_answer, is_correct, message)
            VALUES (?, ?, ?, ?, ?)
            """,
            (attempt_id, part_id, student_answer, None if is_correct is None else int(is_correct), message),
        )
        conn.commit()


def list_attempts_for_problem(problem_id: str, limit: int = 10) -> List[Tuple[str, str]]:
    with db_connect() as conn:
        cur = conn.execute(
            """
            SELECT created_utc, attempt_id
            FROM attempts
            WHERE problem_id = ?
            ORDER BY created_utc DESC
            LIMIT ?
            """,
            (problem_id, limit),
        )
        return list(cur.fetchall())


def get_attempt_parts(attempt_id: str) -> List[Tuple[str, str, Optional[int], str]]:
    with db_connect() as conn:
        cur = conn.execute(
            """
            SELECT part_id, student_answer, is_correct, message
            FROM attempt_parts
            WHERE attempt_id = ?
            ORDER BY part_id ASC
            """,
            (attempt_id,),
        )
        return list(cur.fetchall())


def list_uploads_for_problem(problem_id: str, limit: int = 50) -> List[Tuple[str, str, str, str, int]]:
    """
    Returns rows: (created_utc, attempt_id, part_id, filename, readable)
    """
    with db_connect() as conn:
        cur = conn.execute(
            """
            SELECT u.created_utc, u.attempt_id, u.part_id, u.filename, u.readable
            FROM uploads u
            JOIN attempts a ON a.attempt_id = u.attempt_id
            WHERE a.problem_id = ?
            ORDER BY u.created_utc DESC
            LIMIT ?
            """,
            (problem_id, limit),
        )
        return list(cur.fetchall())


def get_latest_upload_text(attempt_id: str, part_id: str) -> Tuple[str, bool]:
    with db_connect() as conn:
        cur = conn.execute(
            """
            SELECT extracted_text, readable
            FROM uploads
            WHERE attempt_id = ? AND part_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (attempt_id, part_id),
        )
        row = cur.fetchone()
        if not row:
            return "", False
        extracted_text = (row[0] or "")
        readable = bool(row[1] == 1)
        return extracted_text, readable


# -----------------------------
# Uploads: PDF extraction + disk/db save + MEMORY save
# -----------------------------
def try_extract_pdf_text(pdf_bytes: bytes) -> str:
    if not HAS_PYPDF2:
        return ""
    try:
        reader = PyPDF2.PdfReader(pdf_bytes)  # type: ignore
        chunks = []
        for page in reader.pages:
            chunks.append(page.extract_text() or "")
        return "\n".join(chunks).strip()
    except Exception:
        return ""


def save_upload_to_disk_and_db(attempt_id: str, part_id: str, filename: str, file_bytes: bytes) -> Tuple[bool, int, str]:
    safe_mkdir(UPLOADS_DIR)
    part_dir = UPLOADS_DIR / attempt_id / part_id
    safe_mkdir(part_dir)

    stored_path = part_dir / filename
    stored_path.write_bytes(file_bytes)

    extracted = try_extract_pdf_text(file_bytes)
    extracted_len = len(extracted)
    readable = compute_readable(extracted)

    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO uploads (attempt_id, part_id, filename, stored_path, extracted_text, extracted_text_len, readable, created_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (attempt_id, part_id, filename, str(stored_path), extracted, extracted_len, int(readable), utc_now_iso()),
        )
        conn.commit()

    return readable, extracted_len, extracted


def remember_upload_in_memory(problem_id: str, attempt_id: str, part_id: str, filename: str, file_bytes: bytes, readable: bool) -> None:
    key = mem_store_key(problem_id)
    st.session_state.setdefault(key, [])
    st.session_state[key].insert(
        0,
        {
            "created_utc": utc_now_iso(),
            "attempt_id": attempt_id,
            "part_id": part_id,
            "filename": filename,
            "bytes_len": len(file_bytes),
            "readable": bool(readable),
        },
    )


def save_fallback(attempt_id: str, part_id: str, balance_equations: str, notes: str) -> None:
    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO fallback (attempt_id, part_id, balance_equations, notes, created_utc)
            VALUES (?, ?, ?, ?, ?)
            """,
            (attempt_id, part_id, balance_equations, notes, utc_now_iso()),
        )
        conn.commit()


# -----------------------------
# AI feedback (stub)
# -----------------------------
def ai_feedback_stub(problem: Dict[str, Any], part_id: str, work_text: str) -> Dict[str, Any]:
    work = (work_text or "").strip()
    has_eq = "=" in work
    return {
        "schema_version": "1.0",
        "mode": "student",
        "part_id": part_id,
        "confidence": 0.55 if len(work) > 80 else 0.35,
        "evidence_quotes": [
            {"quote": ("=" if has_eq else (work[:80] if work else "")), "reason": "Detected equation marker or snippet from your work."}
        ] if work else [],
        "detected_work": {
            "has_overall_balance": None,
            "has_component_balance": None,
            "uses_correct_symbols": None,
            "uses_correct_compositions": None,
            "units_handling": "unknown",
            "algebra_progress": "unknown",
            "notes": ["Stub feedback: replace ai_feedback_stub() with a real model call."]
        },
        "issues": [
            {
                "category": "missing_info" if len(work) < 40 else "setup",
                "severity": "high" if len(work) < 40 else "medium",
                "diagnosis": "Not enough readable work text to diagnose confidently." if len(work) < 40 else "I can see some work, but you need to clearly show the balances you used.",
                "why_it_matters": "Without seeing the balance equations, feedback can’t be specific.",
                "how_to_fix": "Paste your overall + component balances into the fallback form, then request feedback again."
            }
        ],
        "next_steps": [
            {"action": "Write the overall balance first (In = Out).", "why": "It anchors the unknown flow rates."},
            {"action": "Write the component balance using the given compositions.", "why": "This gives the second independent equation."}
        ],
        "hints": [
            {"level": 1, "hint": "Show the two independent equations you used before substituting numbers.", "gives_final_answer": False}
        ],
        "questions_for_student": [
            "What two equations did you use to solve for the unknowns?"
        ] if len(work) < 40 else [],
        "safety": {
            "revealed_final_numeric_answer": False,
            "revealed_full_solution": False,
            "redactions_applied": False
        }
    }


# -----------------------------
# Session state
# -----------------------------
def init_session_state() -> None:
    st.session_state.setdefault("selected_assignment", None)
    st.session_state.setdefault("selected_problem_id", None)

    # Persist "submitted state" so reruns keep showing uploads/results
    st.session_state.setdefault("active_problem_id", None)
    st.session_state.setdefault("active_attempt_id", None)
    st.session_state.setdefault("active_incorrect_parts", [])
    st.session_state.setdefault("active_results", {})


# -----------------------------
# Sidebar navigation + tabs
# -----------------------------
def render_sidebar(assignments: dict) -> None:
    st.sidebar.title("Navigation")
    names = list(assignments.keys())
    if not names:
        st.sidebar.error("No assignments found.")
        st.stop()

    default_a = names.index(st.session_state["selected_assignment"]) if st.session_state["selected_assignment"] in names else 0
    st.session_state["selected_assignment"] = st.sidebar.selectbox("Assignment", names, index=default_a)

    pids = assignments.get(st.session_state["selected_assignment"], [])
    if not pids:
        st.sidebar.warning("No problems in this assignment.")
        st.stop()

    default_p = pids.index(st.session_state["selected_problem_id"]) if st.session_state["selected_problem_id"] in pids else 0
    st.session_state["selected_problem_id"] = st.sidebar.selectbox("Problem", pids, index=default_p)

    st.sidebar.divider()


def render_sidebar_tabs(problem_id: str) -> None:
    tab_attempts, tab_uploads = st.sidebar.tabs(["Attempt history", "Uploaded files"])

    with tab_attempts:
        st.subheader("Recent Attempts")
        rows = list_attempts_for_problem(problem_id, limit=8)
        if not rows:
            st.caption("No attempts yet.")
        else:
            labels, ids = [], []
            for created_utc, attempt_id in rows:
                labels.append(f"{created_utc} ({attempt_id[:8]})")
                ids.append(attempt_id)

            idx = st.selectbox("View", list(range(len(labels))), format_func=lambda i: labels[i])
            attempt_id = ids[idx]
            with st.expander("Attempt details", expanded=False):
                parts = get_attempt_parts(attempt_id)
                for part_id, ans, is_corr, _msg in parts:
                    if is_corr is None:
                        st.write(f"Part ({part_id}): {ans} — not graded")
                    elif is_corr == 1:
                        st.write(f"Part ({part_id}): ✅ {ans}")
                    else:
                        st.write(f"Part ({part_id}): ❌ {ans}")
                st.caption(f"Attempt ID: {attempt_id}")

    with tab_uploads:
        st.subheader("Uploads")

        mem_uploads = st.session_state.get(mem_store_key(problem_id), [])
        if mem_uploads:
            st.caption("From this session (in memory):")
            for u in mem_uploads[:25]:
                badge = "✅" if u.get("readable") else "⚠️"
                st.write(f"{badge} **{u['filename']}** — Part ({u['part_id']}), Attempt {u['attempt_id'][:8]}")
                st.caption(f"{u['created_utc']} • size={u['bytes_len']} bytes")
        else:
            st.caption("No in-memory uploads yet for this problem (this session).")

        st.divider()

        db_uploads = list_uploads_for_problem(problem_id, limit=25)
        if db_uploads:
            st.caption("From database (saved to disk + logged):")
            for created_utc, attempt_id, part_id, filename, readable in db_uploads:
                badge = "✅" if readable == 1 else "⚠️"
                st.write(f"{badge} **{filename}** — Part ({part_id}), Attempt {attempt_id[:8]}")
                st.caption(f"{created_utc}")
        else:
            st.caption("No database uploads yet for this problem.")


# -----------------------------
# Nomenclature helper (UI)
# -----------------------------
def render_nomenclature_hint(problem: Dict[str, Any]) -> None:
    tutor = problem.get("tutor_layer") or {}
    setup = tutor.get("expected_setup") or {}
    knowns = setup.get("knowns") or {}
    unknowns = setup.get("unknowns") or []

    symbols: List[str] = []
    if isinstance(unknowns, list):
        for u in unknowns:
            if isinstance(u, dict) and u.get("symbol"):
                symbols.append(str(u["symbol"]))
    if isinstance(knowns, dict):
        for k in knowns.keys():
            symbols.append(str(k))

    symbols = [s for s in symbols if s]
    seen = set()
    out = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            out.append(s)

    if out:
        st.info("**Nomenclature to use:** " + ", ".join(out))


# -----------------------------
# Main problem UI
# -----------------------------
def render_problem(problem: Dict[str, Any], assignment: str, answer_key: Dict[Tuple[str, str], Dict[str, str]]) -> None:
    pid = problem.get("problem_id", "")
    title = problem.get("title", pid)
    statement = problem.get("statement", "")

    st.markdown(f"## {title}")
    st.markdown(f"**Problem ID:** `{pid}`")
    if statement:
        st.markdown(statement.replace("\n", "  \n"))

    render_nomenclature_hint(problem)

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

        units = extract_units_from_expected_output(p.get("expected_output", None))

        st.markdown(f"### Part ({part_id})")
        if prompt:
            st.write(prompt)

        label = "Answer" if not units else f"Answer ({units})"
        responses[part_id] = st.text_input(label, key=f"ans_{pid}_{part_id}")

    submitted = st.button("Submit", key=f"submit_{pid}")

    if submitted:
        attempt_id = log_attempt(assignment, pid)

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

        st.session_state["active_problem_id"] = pid
        st.session_state["active_attempt_id"] = attempt_id
        st.session_state["active_incorrect_parts"] = incorrect_parts
        st.session_state["active_results"] = results

    if st.session_state.get("active_problem_id") == pid:
        attempt_id_active = st.session_state.get("active_attempt_id")
        incorrect_parts_active = st.session_state.get("active_incorrect_parts") or []
        if attempt_id_active and incorrect_parts_active:
            st.warning("Upload your work for the parts you missed (one PDF per part).")
            render_per_part_uploads(problem=problem, problem_id=pid, attempt_id=attempt_id_active, incorrect_parts=incorrect_parts_active)
        elif attempt_id_active:
            st.info("All graded parts are correct. No uploads needed.")


def render_per_part_uploads(problem: Dict[str, Any], problem_id: str, attempt_id: str, incorrect_parts: List[str]) -> None:
    st.divider()
    st.subheader("Upload Work (Per Part)")
    st.caption("If we can’t read the PDF text, the fallback form becomes required for AI feedback.")

    parts_map = {str(p.get("part_id", "")).strip(): p for p in (problem.get("parts") or []) if isinstance(p, dict)}
    tutor_layer = problem.get("tutor_layer") or {}

    for part_id in incorrect_parts:
        st.markdown(f"### Part ({part_id}) — Upload")

        state_key = upload_state_key(attempt_id, part_id)

        uploaded = st.file_uploader(
            f"Upload PDF for Part ({part_id})",
            type=["pdf"],
            key=f"uploader_{attempt_id}_{part_id}",
        )

        readable = False
        extracted_text = ""

        if uploaded is not None:
            file_bytes = uploaded.getvalue()
            filename = uploaded.name

            readable, _extracted_len, extracted_text = save_upload_to_disk_and_db(attempt_id, part_id, filename, file_bytes)
            remember_upload_in_memory(problem_id, attempt_id, part_id, filename, file_bytes, readable)

            st.session_state[state_key] = True
            st.success("✅ Upload saved. (See sidebar → Uploaded files)")

        # ✅ FIXED: do not use `if not uploaded:` — use `uploaded is None`
        if uploaded is None:
            extracted_text, readable = get_latest_upload_text(attempt_id, part_id)

        if st.session_state.get(state_key):
            st.caption(f"PDF text status: {'✅ readable' if readable else '⚠️ unreadable (use fallback)'}")
        else:
            st.caption("No PDF uploaded yet for this part.")

        fallback_expanded = bool(st.session_state.get(state_key) and not readable)

        if st.session_state.get(state_key) and not readable:
            st.warning("⚠️ We couldn’t read typed text from your PDF (often scanned/handwritten). Please fill out the fallback form below for AI feedback.")

        with st.expander(f"Fallback form for Part ({part_id})", expanded=fallback_expanded):
            balance = st.text_area(
                "Paste the balance(s)/equation(s) you used (text)",
                height=120,
                key=f"fb_balance_{attempt_id}_{part_id}",
            )
            notes = st.text_area(
                "Notes (what you tried / where you think the mistake is)",
                height=100,
                key=f"fb_notes_{attempt_id}_{part_id}",
            )
            if st.button("Save fallback info", key=f"fb_save_{attempt_id}_{part_id}"):
                save_fallback(attempt_id, part_id, balance, notes)
                st.success("✅ Fallback information saved.")

        fallback_text = (st.session_state.get(f"fb_balance_{attempt_id}_{part_id}", "") or "").strip()
        fallback_notes = (st.session_state.get(f"fb_notes_{attempt_id}_{part_id}", "") or "").strip()

        if readable and extracted_text.strip():
            work_text = extracted_text.strip()
            source_label = "PDF extracted text"
        else:
            work_text = ("\n\n".join([fallback_text, fallback_notes])).strip()
            source_label = "Fallback form text"

        can_request_ai = len(work_text) >= 40

        cols = st.columns([1, 2])
        with cols[0]:
            get_fb = st.button("Get AI feedback", key=f"ai_btn_{attempt_id}_{part_id}", disabled=not can_request_ai)
        with cols[1]:
            if not can_request_ai:
                st.info("To get AI feedback: upload a typed PDF **or** fill in the fallback form (at least a couple sentences/equations).")
            else:
                st.caption(f"AI will use: **{source_label}**")

        if get_fb:
            part_obj = parts_map.get(part_id, {})
            part_prompt = (part_obj.get("prompt") or "")
            feedback = ai_feedback_stub(
                problem={
                    "problem_statement": problem.get("statement", ""),
                    "tutor_layer": tutor_layer,
                    "part_prompt": part_prompt,
                },
                part_id=part_id,
                work_text=work_text,
            )
            st.session_state[ai_feedback_key(attempt_id, part_id)] = feedback

        fb = st.session_state.get(ai_feedback_key(attempt_id, part_id))
        if fb:
            st.subheader("AI feedback")
            st.json(fb)


# -----------------------------
# App entry
# -----------------------------
st.set_page_config(page_title="MEB Tutor", layout="wide")
init_session_state()

safe_mkdir(DATA_DIR)
safe_mkdir(PROBLEMS_DIR)
safe_mkdir(UPLOADS_DIR)
safe_mkdir(LOGS_DIR)
db_init_and_migrate()

st.title("MEB Homework Tutor")
st.caption("Uploads persist across reruns. AI reading uses PDF extracted text when readable, otherwise fallback form.")

try:
    assignments = load_assignments()
except Exception as e:
    st.error(f"Could not load assignments at {ASSIGNMENTS_FILE}.\n\nError: {e}")
    st.stop()

render_sidebar(assignments)

assignment = st.session_state["selected_assignment"]
problem_id = st.session_state["selected_problem_id"]

render_sidebar_tabs(problem_id)

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
