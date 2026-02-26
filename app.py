# app.py
# MEB Tutor — uploads + PDF text extraction + OPENAI API feedback (guardrailed + forced JSON)
#
# Streamlit secrets (local: .streamlit/secrets.toml, cloud: App Settings → Secrets):
#   OPENAI_API_KEY = "sk-..."
#   OPENAI_MODEL   = "gpt-4.1-mini"   # optional (can change)
#
# requirements.txt (repo root, same level as app.py):
#   streamlit
#   openai>=1.0.0
#   PyPDF2
#   pdfplumber
#   pymupdf

import csv
import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import streamlit as st

# ---------- PDF extractors ----------
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False

try:
    import PyPDF2  # type: ignore
    HAS_PYPDF2 = True
except Exception:
    HAS_PYPDF2 = False

try:
    import pdfplumber  # type: ignore
    HAS_PDFPLUMBER = True
except Exception:
    HAS_PDFPLUMBER = False

# ---------- OpenAI ----------
# FIX: Many "AI feedback" errors come from using an older openai package.
# This app expects openai>=1.0.0 (OpenAI client class + Responses API).
try:
    from openai import OpenAI  # type: ignore
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False


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
        return abs(student_val - answer_val) <= tol_value * max(abs(answer_val), 1e-12)
    return abs(student_val - answer_val) <= tol_value


def upload_state_key(attempt_id: str, part_id: str) -> str:
    return f"uploaded_ok_{attempt_id}_{part_id}"


def mem_store_key(problem_id: str) -> str:
    return f"mem_uploads_{problem_id}"


def ai_feedback_key(attempt_id: str, part_id: str) -> str:
    return f"ai_feedback_{attempt_id}_{part_id}"


def extract_units_from_expected_output(expected_raw: Any) -> str:
    if isinstance(expected_raw, dict):
        return (expected_raw.get("units") or "").strip()
    if isinstance(expected_raw, str):
        return expected_raw.strip()
    if isinstance(expected_raw, list) and expected_raw and isinstance(expected_raw[0], dict):
        return (expected_raw[0].get("units") or "").strip()
    return ""


def compute_readable(extracted_text: str) -> bool:
    return len((extracted_text or "").strip()) >= 30


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
# Grading
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

    return (True, f"Correct (within {tol_type} tolerance).{units_msg}") if ok else (False, f"Incorrect.{units_msg}")


# -----------------------------
# Database
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
# PDF extraction (3-pass)
# -----------------------------
def try_extract_pdf_text(pdf_bytes: bytes) -> str:
    # Pass 1: PyMuPDF
    try:
        if HAS_PYMUPDF:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = "\n".join(page.get_text("text") for page in doc).strip()
            if len(text) >= 30:
                return text
    except Exception:
        pass

    # Pass 2: PyPDF2
    try:
        if HAS_PYPDF2:
            import io
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))  # type: ignore
            text = "\n".join((p.extract_text() or "") for p in reader.pages).strip()
            if len(text) >= 30:
                return text
    except Exception:
        pass

    # Pass 3: pdfplumber
    try:
        if HAS_PDFPLUMBER:
            import io
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:  # type: ignore
                text = "\n".join((page.extract_text() or "") for page in pdf.pages).strip()
                return text
    except Exception:
        pass

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
# OpenAI API (forced JSON) — FIXED
# -----------------------------
AI_RULES = """You are a homework coach. Diagnose the student's approach and provide targeted guidance WITHOUT giving away final numeric answers or a complete worked solution.

Hard restrictions:
1) DO NOT compute or reveal the final numeric answer(s) for any asked-for unknown (e.g., V, L, x_B, etc.).
2) DO NOT provide a full step-by-step worked solution. You may show at most ONE equation transformation or ONE algebra step if needed for clarity.
3) DO NOT confirm final numeric answers even if the student wrote them. Confirm setup/logic instead.
4) Use ONLY information from the problem statement, and the student's work text. Do not invent values.

What you SHOULD do:
- Identify missing/incorrect equations, wrong variable meanings (z vs x vs y), sign conventions, unit handling, algebra issues.
- Cite small evidence quotes from the student's text.
- Provide 1–2 next steps and 1 hint.
- Ask 1–2 clarifying questions if the work is unclear.

Output requirements:
- Output MUST be valid JSON (a single JSON object). Return JSON only.
"""

# Optional: a lightweight schema to improve consistency (still "json_object" forced).
# Kept permissive so the model doesn't error if it includes extra fields.
AI_JSON_SCHEMA = {
    "name": "meb_tutor_feedback",
    "schema": {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "schema_version": {"type": "string"},
            "mode": {"type": "string"},
            "part_id": {"type": "string"},
            "confidence": {"type": "number"},
            "issues": {"type": "array"},
            "next_steps": {"type": "array"},
            "hints": {"type": "array"},
            "questions_for_student": {"type": "array"},
            "safety": {"type": "object"},
        },
    },
}

def _openai_error_payload(msg: str) -> dict:
    return {
        "schema_version": "1.0",
        "mode": "student",
        "part_id": "?",
        "confidence": 0.0,
        "issues": [{
            "category": "system",
            "severity": "high",
            "diagnosis": msg,
            "why_it_matters": "The app can’t get AI feedback.",
            "how_to_fix": "Confirm openai>=1.0.0 is installed, OPENAI_API_KEY is set, and check Streamlit logs."
        }],
        "next_steps": [],
        "hints": [],
        "questions_for_student": [],
        "safety": {"revealed_final_numeric_answer": False, "revealed_full_solution": False, "redactions_applied": False},
    }


def call_openai_feedback_json(
    system_prompt: str,
    user_prompt: str,
    *,
    model_default: str = "gpt-4.1-mini",
    temperature: float = 0.25,
    max_output_tokens: int = 900,
) -> dict:
    """
    Calls OpenAI and forces a JSON object response.
    FIX: Use the Responses API (openai>=1.x). This avoids common "response_format not supported"
    and parsing issues seen with older chat.completions codepaths.
    Returns a dict; if anything fails, returns a schema-shaped error dict.
    """
    if not HAS_OPENAI:
        return _openai_error_payload("OpenAI SDK not installed or too old. Ensure requirements.txt has: openai>=1.0.0")

    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return _openai_error_payload("OPENAI_API_KEY missing from Streamlit secrets.")

    model = st.secrets.get("OPENAI_MODEL", model_default)

    try:
        client = OpenAI(api_key=api_key)

        # Prefer Responses API
        # - response_format forces JSON output
        # - input supports a list of role/content items
        resp = client.responses.create(
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format={"type": "json_object"},
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        # Most robust way to get text from Responses API:
        content = (getattr(resp, "output_text", None) or "").strip()
        if not content:
            # fallback: try to reconstruct from resp.output blocks
            try:
                blocks = []
                for item in (resp.output or []):
                    for c in (getattr(item, "content", None) or []):
                        t = getattr(c, "text", None)
                        if isinstance(t, str):
                            blocks.append(t)
                content = "\n".join(blocks).strip()
            except Exception:
                content = ""

        if not content:
            raise ValueError("Model returned empty content")

        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("Model returned non-object JSON")

        return data

    except Exception as e:
        return _openai_error_payload(f"AI call failed: {e}")


# -----------------------------
# Answer leak filter (optional but recommended)
# -----------------------------
def extract_numbers(text: str) -> List[float]:
    nums: List[float] = []
    for m in re.finditer(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text or ""):
        v = parse_float(m.group(0))
        if v is not None:
            nums.append(v)
    return nums


def redact_if_leaks_answer(
    feedback: Dict[str, Any],
    problem_id: str,
    part_id: str,
    answer_key: Dict[Tuple[str, str], Dict[str, str]],
) -> Dict[str, Any]:
    row = answer_key.get((problem_id, part_id))
    if not row:
        return feedback

    ans_val = parse_float(row.get("answer_value", ""))
    tol_val = parse_float(row.get("tolerance_value", ""))
    tol_type = row.get("tolerance_type", "absolute")
    if ans_val is None or tol_val is None:
        return feedback

    blob = json.dumps(feedback, ensure_ascii=False)
    for n in extract_numbers(blob):
        if within_tolerance(n, ans_val, tol_type, tol_val):
            feedback["hints"] = [{
                "level": 1,
                "hint": "I can’t share the final number here — focus on checking your balances and algebra signs.",
                "gives_final_answer": False
            }]
            safety = feedback.get("safety") or {}
            safety["redactions_applied"] = True
            safety["revealed_final_numeric_answer"] = True
            feedback["safety"] = safety
            return feedback
    return feedback


def call_ai_feedback_openended(
    problem_id: str,
    part_id: str,
    problem_statement: str,
    part_prompt: str,
    work_text: str,
    answer_key: Dict[Tuple[str, str], Dict[str, str]],
) -> Dict[str, Any]:
    if not work_text or len(work_text.strip()) < 40:
        return {
            "schema_version": "1.0",
            "mode": "student",
            "part_id": part_id,
            "confidence": 0.2,
            "evidence_quotes": [],
            "detected_work": {},
            "issues": [{
                "category": "missing_info",
                "severity": "high",
                "diagnosis": "Not enough readable work text to analyze.",
                "why_it_matters": "AI needs your equations to give specific feedback.",
                "how_to_fix": "Upload a typed PDF or paste equations into fallback form."
            }],
            "next_steps": [],
            "hints": [{"level": 1, "hint": "Start by writing the overall and component balances clearly.", "gives_final_answer": False}],
            "questions_for_student": [],
            "safety": {"revealed_final_numeric_answer": False, "revealed_full_solution": False, "redactions_applied": False},
        }

    user_prompt = f"""
Problem Statement:
{problem_statement}

Part:
part_id={part_id}
prompt={part_prompt}

Student Work Text:
{work_text}

Return JSON only.
""".strip()

    fb = call_openai_feedback_json(AI_RULES, user_prompt)
    if isinstance(fb, dict):
        fb["schema_version"] = "1.0"
        fb["part_id"] = part_id
    fb = redact_if_leaks_answer(fb, problem_id, part_id, answer_key)
    return fb


# -----------------------------
# USER-FRIENDLY AI FEEDBACK DISPLAY
# -----------------------------
def render_ai_feedback_userfriendly(feedback: Dict[str, Any]) -> None:
    st.subheader("AI Feedback")

    confidence = feedback.get("confidence")
    if isinstance(confidence, (int, float)):
        st.progress(min(max(confidence, 0.0), 1.0))
        st.caption(f"Confidence: {int(confidence * 100)}%")

    issues = feedback.get("issues", [])
    if issues:
        st.markdown("### 🔎 What needs attention")
        for issue in issues:
            severity = issue.get("severity", "low")
            diagnosis = issue.get("diagnosis", "")
            why = issue.get("why_it_matters", "")
            fix = issue.get("how_to_fix", "")

            if severity == "high":
                st.error(diagnosis)
            elif severity == "medium":
                st.warning(diagnosis)
            else:
                st.info(diagnosis)

            if why:
                st.markdown(f"**Why this matters:** {why}")
            if fix:
                st.markdown(f"**How to fix it:** {fix}")

            st.markdown("---")

    steps = feedback.get("next_steps", [])
    if steps:
        st.markdown("### 🪜 Recommended next steps")
        for i, step in enumerate(steps, 1):
            if isinstance(step, dict):
                action = step.get("action", "")
                why = step.get("why", "")
            else:
                action = str(step)
                why = ""
            st.markdown(f"{i}. **{action}**")
            if why:
                st.caption(why)

    hints = feedback.get("hints", [])
    if hints:
        st.markdown("### 💡 Hint")
        for hint in hints:
            if isinstance(hint, dict):
                st.info(hint.get("hint", ""))
            else:
                st.info(str(hint))

    questions = feedback.get("questions_for_student", [])
    if questions:
        st.markdown("### ❓ Questions to consider")
        for q in questions:
            st.markdown(f"- {q}")


# -----------------------------
# Session state
# -----------------------------
def init_session_state() -> None:
    st.session_state.setdefault("selected_assignment", None)
    st.session_state.setdefault("selected_problem_id", None)
    st.session_state.setdefault("active_problem_id", None)
    st.session_state.setdefault("active_attempt_id", None)
    st.session_state.setdefault("active_incorrect_parts", [])
    st.session_state.setdefault("active_results", {})
    st.session_state.setdefault("debug_pdf_extract", False)
    st.session_state.setdefault("debug_ai_input", False)


# -----------------------------
# Sidebar
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
# Nomenclature helper (optional)
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

    out, seen = [], set()
    for s in symbols:
        if s and s not in seen:
            seen.add(s)
            out.append(s)

    if out:
        st.info("**Nomenclature to use:** " + ", ".join(out))


# -----------------------------
# Main UI
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
            render_per_part_uploads(problem=problem, problem_id=pid, attempt_id=attempt_id_active, incorrect_parts=incorrect_parts_active, answer_key=answer_key)
        elif attempt_id_active:
            st.info("All graded parts are correct. No uploads needed.")


def render_per_part_uploads(problem: Dict[str, Any], problem_id: str, attempt_id: str, incorrect_parts: List[str], answer_key: Dict[Tuple[str, str], Dict[str, str]]) -> None:
    st.divider()
    st.subheader("Upload Work (Per Part)")

    parts_map = {str(p.get("part_id", "")).strip(): p for p in (problem.get("parts") or []) if isinstance(p, dict)}

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
            readable, _lenx, extracted_text = save_upload_to_disk_and_db(attempt_id, part_id, filename, file_bytes)
            remember_upload_in_memory(problem_id, attempt_id, part_id, filename, file_bytes, readable)
            st.session_state[state_key] = True
            st.success("✅ Upload saved. (See sidebar → Uploaded files)")

        if uploaded is None:
            extracted_text, readable = get_latest_upload_text(attempt_id, part_id)

        if st.session_state.get("debug_pdf_extract", False) and st.session_state.get(state_key):
            st.caption(f"DEBUG PDF: len={len(extracted_text)} | readable={readable}")
            st.code((extracted_text[:900] if extracted_text else "<EMPTY>"), language="text")

        if st.session_state.get(state_key):
            st.caption(f"PDF text status: {'✅ readable' if readable else '⚠️ not enough extractable text (use fallback)'}")
        else:
            st.caption("No PDF uploaded yet for this part.")

        fallback_expanded = bool(st.session_state.get(state_key) and not readable)
        if st.session_state.get(state_key) and not readable:
            st.warning("⚠️ We couldn’t extract enough text. Please fill the fallback form for AI feedback.")

        with st.expander(f"Fallback form for Part ({part_id})", expanded=fallback_expanded):
            balance = st.text_area("Paste the balance(s)/equation(s) you used (text)", height=120, key=f"fb_balance_{attempt_id}_{part_id}")
            notes = st.text_area("Notes (what you tried / where you think the mistake is)", height=100, key=f"fb_notes_{attempt_id}_{part_id}")
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

        if st.session_state.get("debug_ai_input", False) and st.session_state.get(state_key):
            st.caption(f"DEBUG AI INPUT: source={source_label} | len={len(work_text)}")
            st.code((work_text[:900] if work_text else "<EMPTY>"), language="text")

        can_request_ai = len(work_text) >= 40

        cols = st.columns([1, 2])
        with cols[0]:
            get_fb = st.button("Get AI feedback", key=f"ai_btn_{attempt_id}_{part_id}", disabled=not can_request_ai)
        with cols[1]:
            st.caption(f"AI will use: **{source_label}**" if can_request_ai else "Upload a typed PDF or fill fallback (a couple lines/equations).")

        if get_fb:
            part_obj = parts_map.get(part_id, {})
            part_prompt = (part_obj.get("prompt") or "")
            fb = call_ai_feedback_openended(
                problem_id=problem_id,
                part_id=part_id,
                problem_statement=(problem.get("statement", "") or ""),
                part_prompt=part_prompt,
                work_text=work_text,
                answer_key=answer_key,
            )
            st.session_state[ai_feedback_key(attempt_id, part_id)] = fb

        fb = st.session_state.get(ai_feedback_key(attempt_id, part_id))
        if fb:
            render_ai_feedback_userfriendly(fb)


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

st.sidebar.divider()
st.session_state["debug_pdf_extract"] = st.sidebar.checkbox("Debug PDF extraction", value=st.session_state["debug_pdf_extract"])
st.session_state["debug_ai_input"] = st.sidebar.checkbox("Debug AI input (show what we send)", value=st.session_state["debug_ai_input"])

st.sidebar.divider()
if HAS_OPENAI and st.secrets.get("OPENAI_API_KEY", None):
    st.sidebar.success("AI: OpenAI connected")
else:
    st.sidebar.warning("AI: not connected (set OPENAI_API_KEY in Streamlit secrets)")

assignments = load_assignments()
render_sidebar(assignments)

assignment = st.session_state["selected_assignment"]
problem_id = st.session_state["selected_problem_id"]

render_sidebar_tabs(problem_id)

problem = load_problem(problem_id)
answer_key = load_answer_key()

render_problem(problem, assignment, answer_key)
