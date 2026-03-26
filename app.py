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


def normalize_assignments_structure(raw_assignments: dict) -> Dict[str, Dict[str, List[str]]]:
    """
    Supports both formats:

    Old format:
    {
      "Assignment 1": ["MEB_001", "MEB_002"],
      "Assignment 2": ["MEB_003"]
    }

    New format:
    {
      "Class A": {
        "Assignment 1": ["MEB_001", "MEB_002"]
      },
      "Class B": {
        "Assignment 1": ["MEB_003"]
      }
    }

    Old format is wrapped into a default class.
    """
    if not isinstance(raw_assignments, dict) or not raw_assignments:
        return {}

    first_val = next(iter(raw_assignments.values()))

    # Old flat format
    if isinstance(first_val, list):
        normalized_old: Dict[str, Dict[str, List[str]]] = {"Default Class": {}}
        for assignment_name, problem_ids in raw_assignments.items():
            if isinstance(problem_ids, list):
                normalized_old["Default Class"][str(assignment_name)] = [str(pid) for pid in problem_ids]
        return normalized_old

    # New nested format
    normalized_new: Dict[str, Dict[str, List[str]]] = {}
    for class_name, assignments_map in raw_assignments.items():
        if not isinstance(assignments_map, dict):
            continue

        class_assignments: Dict[str, List[str]] = {}
        for assignment_name, problem_ids in assignments_map.items():
            if isinstance(problem_ids, list):
                class_assignments[str(assignment_name)] = [str(pid) for pid in problem_ids]

        if class_assignments:
            normalized_new[str(class_name)] = class_assignments

    return normalized_new


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_assignments() -> Dict[str, Dict[str, List[str]]]:
    if not ASSIGNMENTS_FILE.exists():
        raise FileNotFoundError(f"Missing {ASSIGNMENTS_FILE}")
    raw = json.loads(ASSIGNMENTS_FILE.read_text(encoding="utf-8"))
    return normalize_assignments_structure(raw)


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
                class_name TEXT NOT NULL DEFAULT '',
                assignment TEXT NOT NULL,
                problem_id TEXT NOT NULL
            );
            """
        )

        if not db_has_column(conn, "attempts", "class_name"):
            conn.execute("ALTER TABLE attempts ADD COLUMN class_name TEXT NOT NULL DEFAULT '';")

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


def log_attempt(class_name: str, assignment: str, problem_id: str) -> str:
    attempt_id = str(uuid.uuid4())
    with db_connect() as conn:
        conn.execute(
            """
            INSERT INTO attempts (attempt_id, created_utc, class_name, assignment, problem_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            (attempt_id, utc_now_iso(), class_name, assignment, problem_id),
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


def list_attempts_for_problem(problem_id: str, limit: int = 10) -> List[Tuple[str, str, str, str]]:
    with db_connect() as conn:
        cur = conn.execute(
            """
            SELECT created_utc, attempt_id, class_name, assignment
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
# OpenAI API (forced JSON)
# -----------------------------
AI_RULES = """You are a homework coach. Diagnose the student's approach and provide targeted guidance WITHOUT giving away final numeric answers or a complete worked solution.

Hard restrictions:
1) DO NOT compute or reveal the final numeric answer(s) for any asked-for unknown (e.g., V, L, x_B, etc.).
2) DO NOT provide a full step-by-step worked solution. You may show at most ONE equation transformation or ONE algebra step if needed for clarity.
3) DO NOT confirm final numeric answers even if the student wrote them. Confirm setup/logic instead.
4) Use ONLY information from the problem statement and the student's work text. Do not invent values.

What you SHOULD do:
- Identify missing/incorrect equations, wrong variable meanings (z vs x vs y), sign conventions, unit handling, algebra issues.
- Quote 1–3 small evidence snippets from the student's work (short, literal quotes).
- Provide 2–4 concrete next steps, plus 1–2 hints (non-answer-revealing).
- Ask 1–2 clarifying questions if the work is unclear.

Output requirements:
- Output MUST be valid JSON (a single JSON object). Return JSON only.

Return JSON fields (best effort):
- schema_version (string)
- part_id (string)
- confidence (number 0..1)
- summary (string, 1–2 sentences)
- evidence_quotes (array of strings)
- issues (array of objects: category, severity=low|medium|high, diagnosis, why_it_matters, how_to_fix)
- next_steps (array of objects: action, why)
- hints (array of objects: level (1..3), hint, gives_final_answer=false)
- questions_for_student (array of strings)
- safety (object with: revealed_final_numeric_answer=false, revealed_full_solution=false, redactions_applied=false)
"""

AI_JSON_SCHEMA = {
    "name": "meb_tutor_feedback",
    "schema": {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "schema_version": {"type": "string"},
            "part_id": {"type": "string"},
            "confidence": {"type": "number"},
            "summary": {"type": "string"},
            "evidence_quotes": {"type": "array", "items": {"type": "string"}},
            "issues": {"type": "array"},
            "next_steps": {"type": "array"},
            "hints": {"type": "array"},
            "questions_for_student": {"type": "array"},
            "safety": {"type": "object"},
        },
        "required": ["schema_version", "part_id", "confidence", "issues", "next_steps", "hints", "questions_for_student", "safety"],
    },
}


def _openai_error_payload(msg: str, part_id: str = "?") -> dict:
    return {
        "schema_version": "1.1",
        "part_id": part_id,
        "confidence": 0.0,
        "summary": "AI feedback is unavailable right now.",
        "evidence_quotes": [],
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
    part_id: str,
    model_default: str = "gpt-4.1-mini",
    temperature: float = 0.25,
    max_output_tokens: int = 950,
) -> dict:
    if not HAS_OPENAI:
        return _openai_error_payload(
            "OpenAI SDK not installed. Add openai>=1.0.0 to requirements.txt",
            part_id=part_id,
        )

    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        return _openai_error_payload(
            "OPENAI_API_KEY missing from Streamlit secrets.",
            part_id=part_id,
        )

    model = st.secrets.get("OPENAI_MODEL", model_default)

    try:
        client = OpenAI(api_key=api_key)

        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_output_tokens,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt + "\n\nIMPORTANT: Return ONLY valid JSON."
                },
                {
                    "role": "user",
                    "content": user_prompt
                },
            ],
        )

        content = (resp.choices[0].message.content or "").strip()

        if not content:
            raise ValueError("Model returned empty content")

        data = json.loads(content)

        if not isinstance(data, dict):
            raise ValueError("Model returned non-object JSON")

        return data

    except Exception as e:
        return _openai_error_payload(
            f"AI call failed: {e}",
            part_id=part_id,
        )


# -----------------------------
# Answer leak filter
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
                "hint": "I can’t share the final number here — focus on checking your balances, units, and algebra signs.",
                "gives_final_answer": False
            }]
            safety = feedback.get("safety") or {}
            safety["redactions_applied"] = True
            safety["revealed_final_numeric_answer"] = True
            feedback["safety"] = safety
            if isinstance(feedback.get("summary"), str) and any(ch.isdigit() for ch in feedback["summary"]):
                feedback["summary"] = "Good progress — let’s verify setup, assumptions, and algebra (without final numbers)."
            return feedback
    return feedback


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def normalize_feedback_for_ui(feedback: Dict[str, Any], *, part_id: str) -> Dict[str, Any]:
    if not isinstance(feedback, dict):
        return _openai_error_payload("AI returned an unexpected format.", part_id=part_id)

    out = dict(feedback)

    out.setdefault("schema_version", "1.1")
    out["part_id"] = _as_str(out.get("part_id") or part_id)
    out["summary"] = _as_str(out.get("summary") or "")

    conf = out.get("confidence", 0.0)
    if not isinstance(conf, (int, float)):
        conf = 0.0
    out["confidence"] = max(0.0, min(float(conf), 1.0))

    eq = out.get("evidence_quotes", [])
    eq_list = []
    for q in _as_list(eq):
        s = _as_str(q).strip()
        if s:
            eq_list.append(s)
    out["evidence_quotes"] = eq_list[:6]

    issues_norm = []
    for it in _as_list(out.get("issues", [])):
        if isinstance(it, dict):
            sev = _as_str(it.get("severity", "low")).lower().strip()
            if sev not in {"low", "medium", "high"}:
                sev = "low"
            issues_norm.append({
                "category": _as_str(it.get("category", "general")).strip() or "general",
                "severity": sev,
                "diagnosis": _as_str(it.get("diagnosis", "")).strip(),
                "why_it_matters": _as_str(it.get("why_it_matters", "")).strip(),
                "how_to_fix": _as_str(it.get("how_to_fix", "")).strip(),
            })
        else:
            s = _as_str(it).strip()
            if s:
                issues_norm.append({
                    "category": "general",
                    "severity": "low",
                    "diagnosis": s,
                    "why_it_matters": "",
                    "how_to_fix": "",
                })
    out["issues"] = issues_norm

    steps_norm = []
    for it in _as_list(out.get("next_steps", [])):
        if isinstance(it, dict):
            steps_norm.append({
                "action": _as_str(it.get("action", "")).strip(),
                "why": _as_str(it.get("why", "")).strip(),
            })
        else:
            s = _as_str(it).strip()
            if s:
                steps_norm.append({"action": s, "why": ""})
    out["next_steps"] = [s for s in steps_norm if s.get("action")][:8]

    hints_norm = []
    for it in _as_list(out.get("hints", [])):
        if isinstance(it, dict):
            lvl = it.get("level", 1)
            try:
                lvl_i = int(lvl)
            except Exception:
                lvl_i = 1
            lvl_i = max(1, min(lvl_i, 3))
            hints_norm.append({
                "level": lvl_i,
                "hint": _as_str(it.get("hint", "")).strip(),
                "gives_final_answer": bool(it.get("gives_final_answer", False)),
            })
        else:
            s = _as_str(it).strip()
            if s:
                hints_norm.append({"level": 1, "hint": s, "gives_final_answer": False})
    out["hints"] = [h for h in hints_norm if h.get("hint")][:6]

    qs_norm = []
    for q in _as_list(out.get("questions_for_student", [])):
        s = _as_str(q).strip()
        if s:
            qs_norm.append(s)
    out["questions_for_student"] = qs_norm[:6]

    safety = out.get("safety") if isinstance(out.get("safety"), dict) else {}
    out["safety"] = {
        "revealed_final_numeric_answer": bool(safety.get("revealed_final_numeric_answer", False)),
        "revealed_full_solution": bool(safety.get("revealed_full_solution", False)),
        "redactions_applied": bool(safety.get("redactions_applied", False)),
    }

    return out


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
            "schema_version": "1.1",
            "part_id": part_id,
            "confidence": 0.2,
            "summary": "I don’t have enough readable work to give precise feedback yet.",
            "evidence_quotes": [],
            "issues": [{
                "category": "missing_info",
                "severity": "high",
                "diagnosis": "Not enough readable work text to analyze.",
                "why_it_matters": "AI needs your equations to give specific feedback.",
                "how_to_fix": "Upload a typed PDF or paste equations into the fallback form."
            }],
            "next_steps": [
                {"action": "Paste your key equations (overall + component balances) and define variables.", "why": "This lets feedback target the exact step that went wrong."}
            ],
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

Return JSON only. Use the field names from the system prompt.
""".strip()

    fb = call_openai_feedback_json(AI_RULES, user_prompt, part_id=part_id)
    if isinstance(fb, dict):
        fb.setdefault("schema_version", "1.1")
        fb["part_id"] = part_id

    fb = redact_if_leaks_answer(fb, problem_id, part_id, answer_key)
    fb = normalize_feedback_for_ui(fb, part_id=part_id)
    return fb


# -----------------------------
# AI feedback display
# -----------------------------
def _severity_icon(sev: str) -> str:
    sev = (sev or "").lower().strip()
    return {"high": "🟥", "medium": "🟨", "low": "🟦"}.get(sev, "🟦")


def _severity_label(sev: str) -> str:
    sev = (sev or "").lower().strip()
    return {"high": "High", "medium": "Medium", "low": "Low"}.get(sev, "Low")


def render_ai_feedback_userfriendly(feedback: Dict[str, Any]) -> None:
    fb = normalize_feedback_for_ui(feedback, part_id=_as_str(feedback.get("part_id", "?")))

    st.markdown("#### 🤖 AI Feedback")
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
        conf = fb.get("confidence", 0.0)
        issues = fb.get("issues", [])
        steps = fb.get("next_steps", [])
        hints = fb.get("hints", [])

        c1.metric("Confidence", f"{int(conf * 100)}%")
        c2.metric("Issues", str(len(issues)))
        c3.metric("Next steps", str(len(steps)))
        c4.metric("Hints", str(len(hints)))

        st.progress(conf)

        summary = fb.get("summary") or ""
        if summary:
            st.info(summary)

        evidence = fb.get("evidence_quotes", [])
        if evidence:
            with st.expander("Evidence from your work (quotes)", expanded=False):
                for q in evidence:
                    st.markdown(f"• “{q}”")

        tab_overview, tab_issues, tab_steps, tab_hints, tab_questions, tab_raw = st.tabs(
            ["Overview", "Issues", "Next steps", "Hints", "Questions", "Raw JSON"]
        )

        with tab_overview:
            high = [i for i in issues if i.get("severity") == "high"]
            med = [i for i in issues if i.get("severity") == "medium"]
            low = [i for i in issues if i.get("severity") == "low"]

            if high or med or low:
                st.markdown("**Priority**")
                if high:
                    st.error(f"Start with: {high[0].get('diagnosis','').strip() or 'Fix the highest-severity issue first.'}")
                elif med:
                    st.warning(f"Start with: {med[0].get('diagnosis','').strip() or 'Fix the medium-severity issue first.'}")
                else:
                    st.info(f"Start with: {low[0].get('diagnosis','').strip() or 'Polish the setup/units.'}")

            if steps:
                st.markdown("**Next step (recommended)**")
                st.success(steps[0].get("action", "Review your setup and units."))
                why = steps[0].get("why", "")
                if why:
                    st.caption(why)

            safety = fb.get("safety", {})
            if safety.get("redactions_applied"):
                st.warning("Some content was redacted to avoid revealing the final numeric answer.")

        with tab_issues:
            if not issues:
                st.success("No issues detected (or not enough work text to diagnose).")
            else:
                for idx, it in enumerate(issues, 1):
                    sev = it.get("severity", "low")
                    icon = _severity_icon(sev)
                    title = it.get("diagnosis", "").strip() or f"Issue {idx}"
                    category = it.get("category", "general")

                    header = f"{icon} {title}  ·  *{category}*  ·  {_severity_label(sev)}"
                    expanded = True if sev in ("high",) and idx == 1 else False
                    with st.expander(header, expanded=expanded):
                        why = it.get("why_it_matters", "")
                        fix = it.get("how_to_fix", "")
                        if why:
                            st.markdown(f"**Why it matters**  \n{why}")
                        if fix:
                            st.markdown(f"**How to fix**  \n{fix}")

        with tab_steps:
            if not steps:
                st.info("No next steps were returned.")
            else:
                st.markdown("Check off items as you fix them:")
                for i, s in enumerate(steps, 1):
                    action = s.get("action", "")
                    why = s.get("why", "")
                    if not action:
                        continue
                    st.checkbox(f"{i}. {action}", key=f"step_done_{fb['part_id']}_{i}")
                    if why:
                        st.caption(why)

        with tab_hints:
            if not hints:
                st.info("No hints were returned.")
            else:
                hints_sorted = sorted(hints, key=lambda h: int(h.get("level", 1)))
                for h in hints_sorted:
                    lvl = int(h.get("level", 1))
                    text = h.get("hint", "")
                    if not text:
                        continue
                    if lvl == 1:
                        st.info(f"Level 1: {text}")
                    elif lvl == 2:
                        st.warning(f"Level 2: {text}")
                    else:
                        st.error(f"Level 3: {text}")

        with tab_questions:
            qs = fb.get("questions_for_student", [])
            if not qs:
                st.info("No questions were returned.")
            else:
                st.markdown("Answer these to pinpoint the mistake:")
                for q in qs:
                    st.markdown(f"- {q}")
                st.divider()
                st.caption("Optional: jot your answers here (not graded):")
                st.text_area(
                    "Your notes",
                    key=f"student_notes_{fb['part_id']}",
                    height=120,
                    placeholder="Example: I used x as liquid mole fraction, but I didn’t define z…",
                )

        with tab_raw:
            st.caption("Raw model output (useful for debugging prompt/schema).")
            st.code(json.dumps(fb, ensure_ascii=False, indent=2), language="json")
            st.download_button(
                "Download JSON",
                data=json.dumps(fb, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name=f"ai_feedback_{fb['part_id']}.json",
                mime="application/json",
            )


# -----------------------------
# Session state
# -----------------------------
def init_session_state() -> None:
    st.session_state.setdefault("selected_class", None)
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
def render_sidebar(assignments_by_class: Dict[str, Dict[str, List[str]]]) -> None:
    st.sidebar.title("Navigation")

    class_names = list(assignments_by_class.keys())
    if not class_names:
        st.sidebar.error("No classes found.")
        st.stop()

    default_c = class_names.index(st.session_state["selected_class"]) if st.session_state["selected_class"] in class_names else 0
    st.session_state["selected_class"] = st.sidebar.selectbox("Class", class_names, index=default_c)

    assignments_for_class = assignments_by_class.get(st.session_state["selected_class"], {})
    assignment_names = list(assignments_for_class.keys())
    if not assignment_names:
        st.sidebar.warning("No assignments found in this class.")
        st.stop()

    default_a = assignment_names.index(st.session_state["selected_assignment"]) if st.session_state["selected_assignment"] in assignment_names else 0
    st.session_state["selected_assignment"] = st.sidebar.selectbox("Assignment", assignment_names, index=default_a)

    pids = assignments_for_class.get(st.session_state["selected_assignment"], [])
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
            for created_utc, attempt_id, class_name, assignment in rows:
                class_prefix = f"{class_name} • " if class_name else ""
                labels.append(f"{created_utc} ({attempt_id[:8]}) — {class_prefix}{assignment}")
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
# Nomenclature helper
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
def render_problem(problem: Dict[str, Any], class_name: str, assignment: str,
                   answer_key: Dict[Tuple[str, str], Dict[str, str]]) -> None:
    pid = problem.get("problem_id", "")
    title = problem.get("title", pid)
    statement = problem.get("statement", "")

    st.markdown(f"## {title}")
    st.markdown(f"**Class:** `{class_name}`")
    st.markdown(f"**Assignment:** `{assignment}`")
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
        attempt_id = log_attempt(class_name, assignment, pid)

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
            render_per_part_uploads(
                problem=problem,
                problem_id=pid,
                attempt_id=attempt_id_active,
                incorrect_parts=incorrect_parts_active,
                answer_key=answer_key,
            )
        elif attempt_id_active:
            st.info("All graded parts are correct. No uploads needed.")


def render_per_part_uploads(problem: Dict[str, Any], problem_id: str, attempt_id: str, incorrect_parts: List[str],
                            answer_key: Dict[Tuple[str, str], Dict[str, str]]) -> None:
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
            balance = st.text_area(
                "Paste the balance(s)/equation(s) you used (text)",
                height=120,
                key=f"fb_balance_{attempt_id}_{part_id}"
            )
            notes = st.text_area(
                "Notes (what you tried / where you think the mistake is)",
                height=100,
                key=f"fb_notes_{attempt_id}_{part_id}"
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
            with st.spinner("Analyzing your work…"):
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

assignments_by_class = load_assignments()
render_sidebar(assignments_by_class)

class_name = st.session_state["selected_class"]
assignment = st.session_state["selected_assignment"]
problem_id = st.session_state["selected_problem_id"]

render_sidebar_tabs(problem_id)

problem = load_problem(problem_id)
answer_key = load_answer_key()

render_problem(problem, class_name, assignment, answer_key)
