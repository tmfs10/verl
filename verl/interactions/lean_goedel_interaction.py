from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from copy import deepcopy
from typing import Any, Optional
from uuid import uuid4

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

LEAN_CODE_BLOCK_RE = re.compile(r"```(?:lean4|lean)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
DECL_LINE_RE = re.compile(r"^\s*(theorem|lemma|example|def)\b")
DECL_ANY_RE = re.compile(r"^\s*(theorem|lemma|example|def)\b", re.MULTILINE)
ATTR_LINE_RE = re.compile(r"^\s*@\[.*\]\s*$")
BY_SPLIT_RE = re.compile(r":=\s*by\b", re.IGNORECASE | re.DOTALL)
ASSIGN_SPLIT_RE = re.compile(r":=", re.DOTALL)
BY_TRAILING_RE = re.compile(r":=\s*by\s*$", re.IGNORECASE | re.DOTALL)
ASSIGN_TRAILING_RE = re.compile(r":=\s*$", re.DOTALL)
THEOREM_ANY_RE = re.compile(r"^\s*(theorem|lemma|example)\b", re.MULTILINE)
DEF_ANY_RE = re.compile(r"^\s*def\b", re.MULTILINE)

GOEDEL_RETRY_PROMPT_TEMPLATE = (
    "The proof (Round {round_number}) is not correct. Following is the compilation error message, where we use "
    "<error></error> to signal the position of the error.\n\n"
    "{error_message_for_prev_round}\n\n"
    "Before producing the Lean 4 code to formally prove the given theorem, provide a detailed analysis of the error "
    "message."
)


def parse_json_maybe(raw: Any) -> Any:
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def get_and_jload(d: dict[str, Any], key: str, default: Any) -> Any:
    if key not in d:
        return default
    value = d[key]
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def clean_chat_tokens(text: str) -> str:
    cleaned = text.replace("<|im_end|>", "")
    cleaned = cleaned.replace("<|im_start|>system", "")
    cleaned = cleaned.replace("<|im_start|>user", "")
    cleaned = cleaned.replace("<|im_start|>assistant", "")
    return cleaned.strip()


def extract_last_lean_block(text: str) -> str:
    stripped = clean_chat_tokens(text)
    matches = LEAN_CODE_BLOCK_RE.findall(stripped)
    if matches:
        return matches[-1].strip()
    return stripped.strip()


def extract_primary_declaration_block(lean_text: str) -> str:
    text = lean_text.strip()
    if not text:
        return text

    lines = text.splitlines(keepends=True)
    decl_idx: Optional[int] = None
    for index, line in enumerate(lines):
        if DECL_LINE_RE.match(line):
            decl_idx = index

    if decl_idx is None:
        return text

    start = decl_idx
    cursor = decl_idx - 1
    while cursor >= 0 and ATTR_LINE_RE.match(lines[cursor]):
        start = cursor
        cursor -= 1

    return "".join(lines[start:]).strip()


def split_statement_and_proof(theorem_text: str) -> tuple[str, str]:
    lean_text = extract_last_lean_block(theorem_text)
    lean_text = extract_primary_declaration_block(lean_text)

    by_match = BY_SPLIT_RE.search(lean_text)
    if by_match:
        statement = lean_text[: by_match.end()].rstrip()
        proof = lean_text[by_match.end() :].strip()
        return statement, proof

    assign_match = ASSIGN_SPLIT_RE.search(lean_text)
    if assign_match:
        statement = lean_text[: assign_match.end()].rstrip()
        proof = lean_text[assign_match.end() :].strip()
        return statement, proof

    return lean_text.strip(), ""


def normalize_statement_suffix(statement: str) -> str:
    normalized = extract_primary_declaration_block(statement).rstrip()

    if BY_TRAILING_RE.search(normalized) or ASSIGN_TRAILING_RE.search(normalized):
        return normalized

    if THEOREM_ANY_RE.search(normalized):
        return normalized + " := by"

    if DEF_ANY_RE.search(normalized):
        return normalized + " :="

    raise ValueError(
        "Could not normalize theorem statement suffix. "
        f"Expected declaration with ':= by' or ':=', got: {normalized[:120]!r}"
    )


def normalize_theorem_text(statement_raw: Any) -> str:
    if not isinstance(statement_raw, str) or not statement_raw.strip():
        raise ValueError("Expected non-empty theorem statement string.")

    statement_part, _ = split_statement_and_proof(statement_raw)
    return normalize_statement_suffix(statement_part).strip()


def extract_proof_body(proof_raw: Any) -> str:
    if not isinstance(proof_raw, str) or not proof_raw.strip():
        raise ValueError("Expected non-empty Lean completion.")

    lean_text = extract_last_lean_block(proof_raw)
    if not lean_text:
        raise ValueError("Lean completion became empty after code extraction.")

    if DECL_ANY_RE.search(lean_text) or BY_SPLIT_RE.search(lean_text) or ASSIGN_SPLIT_RE.search(lean_text):
        _, proof_body = split_statement_and_proof(lean_text)
        if proof_body:
            return proof_body.strip()
        if DECL_ANY_RE.search(lean_text):
            raise ValueError("Completion contains a declaration but no proof body after ':= by' or ':='.")

    return lean_text.strip()


def resolve_header(ground_truth_obj: Any, extra_info: dict[str, Any]) -> str:
    candidates: list[Any] = []
    if isinstance(ground_truth_obj, dict):
        candidates.extend([ground_truth_obj.get("header"), ground_truth_obj.get("imports")])
    candidates.extend([get_and_jload(extra_info, "header", None), get_and_jload(extra_info, "imports", None)])

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.rstrip() + "\n\n"
    return ""


def resolve_formal_statement(ground_truth_obj: Any, extra_info: dict[str, Any]) -> str:
    candidates: list[Any] = []
    if isinstance(ground_truth_obj, dict):
        candidates.extend(
            [
                ground_truth_obj.get("formal_statement"),
                ground_truth_obj.get("statement"),
                ground_truth_obj.get("theorem"),
                ground_truth_obj.get("formal"),
                ground_truth_obj.get("code"),
                ground_truth_obj.get("full_code"),
            ]
        )
    elif isinstance(ground_truth_obj, str):
        candidates.append(ground_truth_obj)

    candidates.extend(
        [
            get_and_jload(extra_info, "formal_statement", None),
            get_and_jload(extra_info, "statement", None),
            get_and_jload(extra_info, "theorem", None),
        ]
    )

    errors: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        try:
            return normalize_theorem_text(candidate)
        except ValueError as exc:
            errors.append(str(exc))

    if errors:
        raise ValueError(errors[-1])
    raise ValueError("Could not find a usable canonical Lean theorem statement.")


def build_submission_code(solution_str: str, ground_truth_obj: Any, extra_info: dict[str, Any]) -> str:
    header = resolve_header(ground_truth_obj, extra_info)
    theorem_text = resolve_formal_statement(ground_truth_obj, extra_info)
    proof_body = extract_proof_body(solution_str)
    return f"{header}{theorem_text}\n{proof_body.rstrip()}\n"


def resolve_project_path(ground_truth_obj: Any, extra_info: dict[str, Any], overrides: dict[str, Any]) -> Optional[str]:
    candidates = [overrides.get("project_path"), get_and_jload(extra_info, "project_path", None)]
    if isinstance(ground_truth_obj, dict):
        candidates.append(ground_truth_obj.get("project_path"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None


def resolve_safe_verify(ground_truth_obj: Any, extra_info: dict[str, Any], overrides: dict[str, Any]) -> bool:
    candidates: list[Any] = [overrides.get("safe_verify"), get_and_jload(extra_info, "safe_verify", None)]
    if isinstance(ground_truth_obj, dict):
        candidates.append(ground_truth_obj.get("safe_verify"))
    candidates.append(True)
    for candidate in candidates:
        if isinstance(candidate, bool):
            return candidate
    return True


def resolve_timeout(ground_truth_obj: Any, extra_info: dict[str, Any], overrides: dict[str, Any]) -> float:
    candidates: list[Any] = [overrides.get("timeout"), get_and_jload(extra_info, "timeout", None)]
    if isinstance(ground_truth_obj, dict):
        candidates.append(ground_truth_obj.get("timeout"))
    candidates.append(20.0)
    for candidate in candidates:
        if isinstance(candidate, (int, float)):
            return float(candidate)
    return 20.0


def discover_sandbox_base_url(overrides: dict[str, Any]) -> str:
    configured = overrides.get("sandbox_base_url")
    if isinstance(configured, str) and configured.strip():
        return configured.strip().rstrip("/")

    configured_many = overrides.get("sandbox_base_urls")
    if isinstance(configured_many, str) and configured_many.strip():
        return configured_many.split(",")[0].strip().rstrip("/")
    if isinstance(configured_many, list):
        for candidate in configured_many:
            candidate_str = str(candidate).strip()
            if candidate_str:
                return candidate_str.rstrip("/")

    env_many = os.getenv("VERL_LEAN_SANDBOX_BASE_URLS", "").strip()
    if env_many:
        return env_many.split(",")[0].strip().rstrip("/")

    host = os.getenv("NEMO_SKILLS_SANDBOX_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = os.getenv("NEMO_SKILLS_SANDBOX_PORT", "6000").strip() or "6000"
    return f"http://{host}:{port}"


def dispatch_execute(host_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    endpoint = f"{host_url.rstrip('/')}/execute"
    default_response = {
        "process_status": "error",
        "stdout": "",
        "stderr": "Lean interaction request failed before a valid response was returned.",
        "safe_verify_passed": False,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json,text/plain,*/*",
        },
        method="POST",
    )
    timeout = max(5.0, float(payload.get("timeout", 20.0)) + 10.0)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body_text = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(body_text) if body_text else {}
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected dict JSON response, got {type(parsed).__name__}")
            return {
                "status": "success",
                "response": parsed,
                "error": "",
                "host": host_url,
            }
    except urllib.error.HTTPError as exc:
        try:
            body_text = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body_text = str(exc)
        return {
            "status": "error",
            "response": default_response,
            "error": body_text,
            "host": host_url,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "response": default_response,
            "error": str(exc),
            "host": host_url,
        }


def format_lean_failure_message(result: dict[str, Any]) -> str:
    lines: list[str] = []
    if result.get("request_error"):
        lines.append(str(result["request_error"]).strip())

    stderr = str(result.get("stderr", "") or "").strip()
    stdout = str(result.get("stdout", "") or "").strip()
    safe_verify_stderr = str(result.get("safe_verify_stderr", "") or "").strip()
    safe_verify_stdout = str(result.get("safe_verify_stdout", "") or "").strip()

    if not result.get("compile_ok", False):
        if stderr:
            lines.append(stderr)
        if stdout:
            lines.append(stdout)
    elif result.get("safe_verify_requested", False) and not result.get("safe_verify_ok", False):
        lines.append("SafeVerify rejected the proof.")
        if safe_verify_stderr:
            lines.append(safe_verify_stderr)
        if safe_verify_stdout:
            lines.append(safe_verify_stdout)

    if not lines:
        process_status = result.get("process_status", "error")
        lines.append(f"error: process_status={process_status}")
    return "\n".join(line for line in lines if line).strip()


def evaluate_lean_attempt(
    solution_str: str,
    *,
    ground_truth_obj: Any,
    extra_info: dict[str, Any],
    overrides: dict[str, Any],
) -> dict[str, Any]:
    try:
        submission_code = build_submission_code(solution_str, ground_truth_obj, extra_info)
        timeout_s = resolve_timeout(ground_truth_obj, extra_info, overrides)
        safe_verify = resolve_safe_verify(ground_truth_obj, extra_info, overrides)
        project_path = resolve_project_path(ground_truth_obj, extra_info, overrides)
    except Exception as exc:  # noqa: BLE001
        result = {
            "submission_code": solution_str,
            "compile_ok": False,
            "safe_verify_ok": False,
            "safe_verify_requested": False,
            "process_status": "error",
            "request_error": str(exc),
            "stdout": "",
            "stderr": "",
            "safe_verify_stdout": "",
            "safe_verify_stderr": "",
            "host": "",
            "final_ok": False,
        }
        result["error_message"] = format_lean_failure_message(result)
        return result

    payload = {
        "generated_code": submission_code,
        "timeout": timeout_s,
        "language": "lean4",
        "safe_verify": bool(safe_verify),
    }
    if project_path is not None:
        payload["project_path"] = project_path

    host_url = discover_sandbox_base_url(overrides)
    sandbox_result = dispatch_execute(host_url, payload)
    response = sandbox_result.get("response", {}) or {}
    process_status = str(response.get("process_status", "error")).strip().lower()
    compile_ok = process_status == "completed"
    safe_verify_requested = bool(safe_verify)
    safe_verify_value = response.get("safe_verify_passed")
    safe_verify_ok = isinstance(safe_verify_value, bool) and safe_verify_value
    final_ok = compile_ok and ((not safe_verify_requested) or safe_verify_ok)

    result = {
        "submission_code": submission_code,
        "compile_ok": compile_ok,
        "safe_verify_ok": safe_verify_ok,
        "safe_verify_requested": safe_verify_requested,
        "process_status": process_status,
        "request_error": str(sandbox_result.get("error", "") or ""),
        "stdout": str(response.get("stdout", "") or ""),
        "stderr": str(response.get("stderr", "") or ""),
        "safe_verify_stdout": str(response.get("safe_verify_stdout", "") or ""),
        "safe_verify_stderr": str(response.get("safe_verify_stderr", "") or ""),
        "host": str(sandbox_result.get("host", "") or ""),
        "final_ok": final_ok,
    }
    result["error_message"] = "" if final_ok else format_lean_failure_message(result)
    return result


class LeanGoedelInteraction(BaseInteraction):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._instance_dict: dict[str, dict[str, Any]] = {}
        self.turn_context_mode = config.get("turn_context_mode", "full_history")
        if self.turn_context_mode != "full_history":
            raise ValueError("LeanGoedelInteraction currently only supports turn_context_mode='full_history'.")
        self.retry_prompt_template = config.get("retry_prompt_template", GOEDEL_RETRY_PROMPT_TEMPLATE)
        self.success_response = config.get("success_response", "Your proof is correct!")
        self.success_reward = float(config.get("success_reward", 1.0))
        self.failure_reward = float(config.get("failure_reward", 0.0))

    @staticmethod
    def _turn_completed(stop_reason: Optional[str]) -> bool:
        return stop_reason not in ("aborted", "abort")

    @staticmethod
    def _extract_last_assistant_content(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "assistant":
                return str(message.get("content") or "")
        return ""

    def _build_runtime_overrides(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        overrides = dict(self.config)
        for key in ("project_path", "safe_verify", "timeout", "sandbox_base_url", "sandbox_base_urls"):
            if key in kwargs and kwargs[key] is not None:
                overrides[key] = kwargs[key]
        return overrides

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[Any] = None,
        raw_prompt: Optional[list[dict[str, Any]]] = None,
        extra_info: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        del kwargs
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "ground_truth": parse_json_maybe(ground_truth),
            "extra_info": deepcopy(extra_info or {}),
            "initial_prompt": deepcopy(raw_prompt or []),
            "attempt_count": 0,
            "has_last_completed_attempt": False,
            "last_completed_success": False,
            "last_completed_submission": "",
        }
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> tuple[bool, str, float, dict[str, Any]]:
        stop_reason = kwargs.get("stop_reason")
        turn_completed = self._turn_completed(stop_reason)
        content = self._extract_last_assistant_content(messages)

        instance = self._instance_dict[instance_id]
        instance["attempt_count"] += 1
        attempt_number = instance["attempt_count"]

        result = evaluate_lean_attempt(
            content,
            ground_truth_obj=instance["ground_truth"],
            extra_info=instance["extra_info"],
            overrides=self._build_runtime_overrides(kwargs),
        )

        if turn_completed:
            instance["has_last_completed_attempt"] = True
            instance["last_completed_success"] = bool(result["final_ok"])
            instance["last_completed_submission"] = result["submission_code"]

        metadata = {
            "last_turn_completed": turn_completed,
            "has_last_completed_proof": instance["has_last_completed_attempt"],
            "last_completed_proof_correct": instance["last_completed_success"],
            "lean_retry_round": attempt_number,
            "lean_turn_compile_acc": float(result["compile_ok"]),
            "lean_turn_safe_verify_acc": float(result["safe_verify_ok"]),
            "lean_turn_safe_verify_requested": float(result["safe_verify_requested"]),
            "lean_turn_process_status": result["process_status"],
            "lean_turn_error": result["request_error"],
            "lean_turn_stdout": result["stdout"],
            "lean_turn_stderr": result["stderr"],
            "lean_turn_safe_verify_stdout": result["safe_verify_stdout"],
            "lean_turn_safe_verify_stderr": result["safe_verify_stderr"],
            "lean_turn_host": result["host"],
            "lean_last_error_message": result["error_message"],
        }

        if result["final_ok"]:
            return True, self.success_response, self.success_reward, metadata

        retry_prompt = self.retry_prompt_template.format(
            round_number=attempt_number,
            error_message_for_prev_round=result["error_message"],
        )
        metadata["reset_generation_prompt"] = True
        metadata["next_generation_messages"] = deepcopy(messages) + [{"role": "user", "content": retry_prompt}]
        return False, retry_prompt, self.failure_reward, metadata

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        del kwargs
        instance = self._instance_dict[instance_id]
        if not instance["has_last_completed_attempt"]:
            return 0.0
        return 1.0 if instance["last_completed_success"] else 0.0

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        del kwargs
        self._instance_dict.pop(instance_id, None)
