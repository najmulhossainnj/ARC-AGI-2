from __future__ import annotations
import re
import time
import traceback
import numpy as np
from typing import List, Tuple, Optional, Callable

from .prompt_builder import build_prompt, build_correction_prompt


def _call_gemini_flash(prompt: str, api_key: Optional[str] = None, timeout: int = 60) -> Optional[str]:
    """Call LLMs in priority order: OpenRouter -> Groq -> Gemini."""
    import os

    # 1. First Priority: OpenRouter
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        res = _call_openrouter_fallback(prompt, openrouter_key)
        if res:
            return res
        print("[LLM] OpenRouter failed, trying Groq...")

    # 2. Groq fallback
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        print("[LLM Model Used] Groq -> llama-3.3-70b-versatile")
        res = _call_groq_fallback(prompt, groq_key)
        if res:
            return res
        print("[LLM] Groq failed, trying Gemini...")

    # 3. Gemini API Key cascade
    raw_keys = api_key or os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    if not keys:
        print("[LLM] No GEMINI_API_KEY, GROQ_API_KEY, or OPENROUTER_API_KEY found in environment.")
        return None

    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash-latest",
    ]

    try:
        from google import genai
        for key in keys:
            client = genai.Client(api_key=key)
            for model_name in models_to_try:
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                    )
                    if response and response.text:
                        print(f"[LLM Model Used] Gemini -> {model_name} (Key ...{key[-6:]})")
                        return response.text
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Quota" in err_str:
                        m = re.search(r"(\d+)(?:\.\d+)?s", err_str, re.IGNORECASE)
                        wait_sec = int(m.group(1)) + 1 if m else 10
                        wait_sec = min(max(wait_sec, 5), 45)
                        print(f"[LLM Rate Limit] {model_name} rate limited. Pausing {wait_sec}s...")
                        time.sleep(wait_sec)
                        try:
                            resp = client.models.generate_content(model=model_name, contents=prompt)
                            if resp and resp.text:
                                print(f"[LLM Model Used] Gemini -> {model_name} after retry")
                                return resp.text
                        except Exception:
                            continue
                    else:
                        continue
    except ImportError:
        pass

    print("[LLM] All LLM endpoints / keys were exhausted or rate limited.")
    return None


def _call_groq_fallback(prompt: str, api_key: str) -> Optional[str]:
    """Fallback to Groq API (Llama 3.3 70B)."""
    import requests
    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=30)
        if r.status_code == 200:
            data = r.json()
            return data["choices"][0]["message"]["content"]
        else:
            print(f"[LLM Groq Notice] HTTP {r.status_code}: {r.text[:120]}")
    except Exception as e:
        print(f"[LLM] Groq API error: {e}")
    return None


def _call_openrouter_fallback(prompt: str, api_key: str) -> Optional[str]:
    """Fallback to OpenRouter API across valid model slugs."""
    import requests
    models = [
        "deepseek/deepseek-chat",
        "meta-llama/llama-3.3-70b-instruct:free",
        "qwen/qwen-2.5-72b-instruct:free",
    ]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for model_name in models:
        try:
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            }
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=30)
            if r.status_code == 200:
                data = r.json()
                print(f"[LLM Model Used] OpenRouter -> {model_name}")
                return data["choices"][0]["message"]["content"]
            else:
                print(f"[LLM OpenRouter Notice] {model_name} HTTP {r.status_code}: {r.text[:120]}")
        except Exception as e:
            print(f"[LLM OpenRouter Notice] {model_name} error: {e}")
    return None


def _extract_python_code(text: str) -> Optional[str]:
    """Extract Python code from markdown code block in LLM response."""
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"(def solve\s*\(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def _compile_and_extract_solve(code: str) -> tuple[Optional[Callable], Optional[str]]:
    """Safely compile generated code and extract the solve() function.
    Returns (solve_fn, error_str)."""
    namespace = {"np": np}
    try:
        from scipy.ndimage import label
        namespace["label"] = label
    except ImportError:
        pass
    try:
        exec(compile(code, "<llm_generated>", "exec"), namespace)
    except Exception as e:
        err = f"Compile error: {type(e).__name__}: {e}"
        print(f"[LLM] {err}")
        return None, err
    solve_fn = namespace.get("solve")
    if not callable(solve_fn):
        err = "Generated code has no callable 'solve' function."
        print(f"[LLM] {err}")
        return None, err
    return solve_fn, None


def _verify_solve_fn(
    solve_fn: Callable,
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    timeout_per_pair: float = 5.0,
) -> tuple[bool, str]:
    """Verify solve_fn achieves 100% exact match on all train pairs.
    Returns (passed, failure_reason)."""
    for i, (inp, out) in enumerate(train_pairs):
        inp, out = np.asarray(inp, dtype=np.int16), np.asarray(out, dtype=np.int16)
        try:
            pred = solve_fn(inp.copy())
            pred = np.asarray(pred, dtype=np.int16)
            if pred.shape != out.shape:
                reason = f"Pair {i}: output shape mismatch: expected {list(out.shape)}, got {list(pred.shape)}"
                return False, reason
            if not np.array_equal(pred, out):
                diff = int((pred != out).sum())
                reason = f"Pair {i}: {diff} cells differ from expected output"
                return False, reason
        except Exception as e:
            reason = f"Pair {i}: solve() raised {type(e).__name__}: {e}\n{traceback.format_exc()}"
            return False, reason
    return True, ""


class GeminiFlashCodegen:
    """Calls LLMs to generate a solve() function for a MISS task, with self-correction on failure."""

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 3):
        self.api_key = api_key
        self.max_retries = max_retries

    def generate(
        self,
        task_id: str,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        features: dict,
        tried_ops: List[str],
    ) -> Optional[Callable]:
        last_code: Optional[str] = None
        last_failure: Optional[str] = None

        for attempt in range(1, self.max_retries + 1):
            print(f"[LLM] Code generation attempt {attempt}/{self.max_retries} for task {task_id}")

            # On first attempt, use fresh diagnostic prompt.
            # On subsequent attempts, use self-correction prompt with previous code + failure.
            if attempt == 1 or last_code is None:
                prompt = build_prompt(task_id, train_pairs, features, tried_ops)
            else:
                print(f"[LLM] Sending self-correction prompt (prev failure: {last_failure[:80] if last_failure else '?'}...)")
                prompt = build_correction_prompt(
                    task_id, train_pairs, features, tried_ops,
                    previous_code=last_code,
                    failure_reason=last_failure or "Unknown mismatch",
                )

            raw = _call_gemini_flash(prompt, api_key=self.api_key)
            if raw is None:
                print(f"[LLM] API returned None on attempt {attempt}. Breaking.")
                break

            code = _extract_python_code(raw)
            if code is None:
                print("[LLM] Could not extract Python code from response.")
                last_failure = "No Python code block found in LLM response"
                continue

            print(f"[LLM] Extracted code ({len(code)} chars). Compiling...")
            solve_fn, compile_err = _compile_and_extract_solve(code)
            if solve_fn is None:
                last_code = code
                last_failure = compile_err or "Compile failed"
                continue

            print(f"[LLM] Verifying solve() on all train pairs...")
            passed, failure_reason = _verify_solve_fn(solve_fn, train_pairs)
            if passed:
                print(f"[LLM] OK Generated solve() passes 100% of train pairs for task {task_id}!")
                solve_fn._llm_code = code
                return solve_fn
            else:
                print(f"[LLM] Candidate solve() failed verification (attempt {attempt}): {failure_reason[:120]}")
                last_code = code
                last_failure = failure_reason

        return None
