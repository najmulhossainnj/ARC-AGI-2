from __future__ import annotations
import re
import time
import textwrap
import numpy as np
from typing import List, Tuple, Optional, Callable

from .prompt_builder import build_prompt


def _call_gemini_flash(prompt: str, api_key: Optional[str] = None, timeout: int = 30) -> Optional[str]:
    """Call Gemini Flash API across multiple free models and API keys."""
    import os
    # Support comma-separated list of keys for automatic key rotation
    raw_keys = api_key or os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    if not keys:
        print("[LLM] No GEMINI_API_KEY set in environment.")
        return None

    # Try all free models (Flash, Pro, Experimental, Lite)
    models_to_try = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash-thinking-exp",
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
                        return response.text
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Quota" in err_str:
                        # Continue to next model or next API key immediately
                        continue
                    else:
                        continue
    except ImportError:
        pass

    # Fallback to Groq / OpenRouter if GROQ_API_KEY / OPENROUTER_API_KEY present
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        res = _call_groq_fallback(prompt, groq_key)
        if res:
            return res

    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        res = _call_openrouter_fallback(prompt, openrouter_key)
        if res:
            return res

    print("[LLM] All LLM endpoints / keys were exhausted or rate limited.")
    return None


def _call_groq_fallback(prompt: str, api_key: str) -> Optional[str]:
    """Fallback to Groq API (Llama 3.3 70B / DeepSeek R1)."""
    import requests
    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        r = requests.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers, timeout=20)
        if r.status_code == 200:
            data = r.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[LLM] Groq API fallback notice: {e}")
    return None


def _call_openrouter_fallback(prompt: str, api_key: str) -> Optional[str]:
    """Fallback to OpenRouter API (DeepSeek / Qwen free tiers)."""
    import requests
    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": "deepseek/deepseek-chat:free",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=20)
        if r.status_code == 200:
            data = r.json()
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[LLM] OpenRouter API fallback notice: {e}")
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


def _compile_and_extract_solve(code: str) -> Optional[Callable]:
    """Safely compile generated code and extract the solve() function."""
    namespace = {"np": np}
    try:
        exec(compile(code, "<llm_generated>", "exec"), namespace)
    except Exception as e:
        print(f"[LLM] Compile error: {e}")
        return None
    solve_fn = namespace.get("solve")
    if not callable(solve_fn):
        print("[LLM] Generated code has no callable 'solve' function.")
        return None
    return solve_fn


def _verify_solve_fn(
    solve_fn: Callable,
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    timeout_per_pair: float = 5.0,
) -> bool:
    """Verify solve_fn achieves 100% exact match on all train pairs."""
    for inp, out in train_pairs:
        inp, out = np.asarray(inp, dtype=np.int16), np.asarray(out, dtype=np.int16)
        try:
            pred = solve_fn(inp.copy())
            pred = np.asarray(pred, dtype=np.int16)
            if pred.shape != out.shape or not np.array_equal(pred, out):
                return False
        except Exception:
            return False
    return True


class GeminiFlashCodegen:
    """Calls LLMs to generate a solve() function for a MISS task."""

    def __init__(self, api_key: Optional[str] = None, max_retries: int = 2):
        self.api_key = api_key
        self.max_retries = max_retries

    def generate(
        self,
        task_id: str,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        features: dict,
        tried_ops: List[str],
    ) -> Optional[Callable]:
        prompt = build_prompt(task_id, train_pairs, features, tried_ops)

        for attempt in range(1, self.max_retries + 1):
            print(f"[LLM] Code generation attempt {attempt}/{self.max_retries} for task {task_id}")
            raw = _call_gemini_flash(prompt, api_key=self.api_key)
            if raw is None:
                break

            code = _extract_python_code(raw)
            if code is None:
                print("[LLM] Could not extract Python code from response.")
                continue

            print(f"[LLM] Extracted code ({len(code)} chars). Verifying...")
            solve_fn = _compile_and_extract_solve(code)
            if solve_fn is None:
                continue

            if _verify_solve_fn(solve_fn, train_pairs):
                print(f"[LLM] OK Generated solve() passes 100% of train pairs for task {task_id}!")
                solve_fn._llm_code = code
                return solve_fn
            else:
                print(f"[LLM] Candidate generated solve() failed verification (attempt {attempt}).")

        return None
