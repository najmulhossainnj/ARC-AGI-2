from __future__ import annotations
import re
import time
import textwrap
import numpy as np
from typing import List, Tuple, Optional, Callable

from .prompt_builder import build_prompt

# Global rate limit cooldown tracking
_last_llm_call_time = 0.0
_llm_cooldown_seconds = 10.0  # Minimum seconds between LLM calls to avoid rate limits


def _call_gemini_flash(prompt: str, api_key: Optional[str] = None, timeout: int = 30, attempt_num: int = 1) -> Optional[str]:
    """Call LLM API across multiple providers/models, rotating based on attempt number."""
    import os
    import requests
    global _last_llm_call_time

    # Apply cooldown between LLM calls to avoid rate limits
    elapsed = time.time() - _last_llm_call_time
    if elapsed < _llm_cooldown_seconds:
        wait_time = _llm_cooldown_seconds - elapsed
        print(f"[LLM] Cooldown: waiting {wait_time:.1f}s...")
        time.sleep(wait_time)

    # 1. FIRST PRIORITY: Groq API (fastest, has free tier with credits)
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        print(f"[LLM] Trying Groq (attempt {attempt_num})...")
        res = _call_groq_with_rotation(prompt, groq_key, attempt_num)
        if res:
            _last_llm_call_time = time.time()
            return res
        print("[LLM] Groq failed, trying other endpoints...")

    # 2. Second Priority: OpenRouter API (has some free models)
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        print(f"[LLM] Trying OpenRouter (attempt {attempt_num})...")
        res = _call_openrouter_fallback(prompt, openrouter_key, attempt_num=attempt_num)
        if res:
            _last_llm_call_time = time.time()
            return res
        print("[LLM] OpenRouter failed, trying other endpoints...")

    # 3. Third Priority: Minimax API
    minimax_key = os.environ.get("MINIMAX_API_KEY")
    if minimax_key:
        print("[LLM] Trying Minimax -> MiniMax-Text-01")
        res = _call_minimax_fallback(prompt, minimax_key)
        if res:
            _last_llm_call_time = time.time()
            return res
        print("[LLM] Minimax failed, trying Gemini endpoints...")

    # 4. Gemini API via OpenAI-compatible endpoint (works better for some keys)
    raw_keys = api_key or os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
    if not keys:
        print("[LLM] No GROQ_API_KEY, OPENROUTER_API_KEY, MINIMAX_API_KEY, or GEMINI_API_KEY found in environment.")
        return None

    # Try OpenAI-compatible Gemini API first (better for quota management)
    for key in keys:
        try:
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            payload = {
                "model": "gemini-2.0-flash",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "max_tokens": 2048,
            }
            r = requests.post(
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                json=payload,
                headers=headers,
                timeout=60
            )
            if r.status_code == 200:
                data = r.json()
                _last_llm_call_time = time.time()
                print(f"[LLM Model Used] Gemini-OpenAI-Compat -> gemini-2.0-flash")
                return data["choices"][0]["message"]["content"]
            elif r.status_code == 429:
                print(f"[LLM Rate Limit] Gemini-OpenAI-Compat rate limited, backing off...")
                # Exponential backoff on rate limit
                time.sleep(30)
                # Try once more after backoff
                try:
                    r2 = requests.post(
                        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=60
                    )
                    if r2.status_code == 200:
                        data = r2.json()
                        _last_llm_call_time = time.time()
                        print(f"[LLM Model Used] Gemini-OpenAI-Compat -> gemini-2.0-flash (after backoff)")
                        return data["choices"][0]["message"]["content"]
                except:
                    pass
                continue
            else:
                print(f"[LLM] Gemini-OpenAI-Compat error: HTTP {r.status_code}: {r.text[:100]}")
        except Exception as e:
            print(f"[LLM] Gemini-OpenAI-Compat error: {e}")
            continue

    # 5. Fall back to google-genai library
    models_to_try = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
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
                    _last_llm_call_time = time.time()
                    if response and response.text:
                        print(f"[LLM Model Used] Gemini -> {model_name} (Key ...{key[-6:]})")
                        return response.text
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                        print(f"[LLM Rate Limit] Skipping {model_name}, trying next...")
                        continue
                    print(f"[LLM] Error with {model_name}: {err_str[:100]}")
                    continue
    except ImportError:
        pass

    print("[LLM] All LLM endpoints / keys were exhausted or rate limited.")
    return None


def _call_groq_with_rotation(prompt: str, api_key: str, attempt_num: int = 1) -> Optional[str]:
    """Call Groq API with model rotation for variety."""
    import requests
    
    # Groq models - use only working models
    groq_models = [
        "llama-3.3-70b-versatile",  # Most capable
        "llama-3.1-8b-instant",      # Fast and cheap
    ]
    
    # Rotate model based on attempt
    primary_model = groq_models[(attempt_num - 1) % len(groq_models)]
    
    # Vary temperature
    temp = 0.1 + (attempt_num % 4) * 0.1  # 0.1, 0.2, 0.3, 0.4
    
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    
    for model in [primary_model] + [m for m in groq_models if m != primary_model]:
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temp,
            }
            r = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=30
            )
            if r.status_code == 200:
                data = r.json()
                print(f"[LLM Model Used] Groq -> {model} (temp={temp})")
                return data["choices"][0]["message"]["content"]
            elif r.status_code == 429:
                print(f"[LLM] Groq {model} rate limited")
                continue
            else:
                print(f"[LLM] Groq {model} HTTP {r.status_code}: {r.text[:100]}")
        except Exception as e:
            print(f"[LLM] Groq {model} error: {e}")
    
    return None


def _call_minimax_fallback(prompt: str, api_key: str) -> Optional[str]:
    """Fallback to Minimax API (MiniMax-Text-01)."""
    import requests
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "MiniMax-Text-01",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 4096,
        }
        r = requests.post(
            "https://api.minimax.chat/v1/text/chatcompletion_v2",
            json=payload,
            headers=headers,
            timeout=60
        )
        if r.status_code == 200:
            data = r.json()
            # Minimax returns choices[0].delta.content or choices[0].message.content
            return data["choices"][0]["message"]["content"]
        else:
            print(f"[LLM] Minimax API notice: HTTP {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"[LLM] Minimax API fallback notice: {e}")
    return None


def _call_openrouter_fallback(prompt: str, api_key: str, attempt_num: int = 1) -> Optional[str]:
    """Fallback to OpenRouter API with free/low-cost models, rotating based on attempt."""
    import requests
    
    # Free/low-cost OpenRouter models
    model_pools = [
        ["deepseek/deepseek-chat", "anthropic/claude-3-haiku", "google/gemini-flash-1.5", "mistralai/mistral-7b-instruct"],
        ["anthropic/claude-3-haiku", "mistralai/mixtral-8x7b-instruct", "deepseek/deepseek-chat", "google/gemini-flash-1.5"],
        ["google/gemini-flash-1.5", "deepseek/deepseek-chat", "anthropic/claude-3-haiku", "mistralai/mistral-7b-instruct"],
        ["mistralai/mistral-7b-instruct", "anthropic/claude-3-haiku", "deepseek/deepseek-chat", "google/gemini-flash-1.5"],
    ]
    
    models = model_pools[(attempt_num - 1) % len(model_pools)]
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for model_name in models:
        try:
            temp = 0.1 + (attempt_num % 4) * 0.1
            payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temp,
            }
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers, timeout=30)
            if r.status_code == 200:
                data = r.json()
                print(f"[LLM Model Used] OpenRouter -> {model_name} (temp={temp})")
                return data["choices"][0]["message"]["content"]
            elif r.status_code == 402:
                print(f"[LLM OpenRouter Notice] {model_name} - insufficient credits")
            elif r.status_code == 429:
                print(f"[LLM OpenRouter Notice] {model_name} - rate limited")
            elif r.status_code == 404:
                print(f"[LLM OpenRouter Notice] {model_name} - not found")
            else:
                print(f"[LLM OpenRouter Notice] {model_name} HTTP {r.status_code}: {r.text[:100]}")
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
        attempt_num: int = 1,
    ) -> Optional[Callable]:
        # Use different strategies based on attempt number
        prompt = build_prompt(task_id, train_pairs, features, tried_ops, attempt_num=attempt_num)
        
        # More retries on later attempts
        effective_retries = min(self.max_retries + (attempt_num - 1), 4)

        for attempt in range(1, effective_retries + 1):
            print(f"[LLM] Code generation attempt {attempt}/{effective_retries} for task {task_id}")
            raw = _call_gemini_flash(prompt, api_key=self.api_key, attempt_num=(attempt_num - 1) * effective_retries + attempt)
            if raw is None:
                break

            code = _extract_python_code(raw)
            if code is None:
                print("[LLM] Could not extract Python code from response.")
                continue

            print(f"[LLM] Extracted code ({len(code)} chars). Verifying...")
            print(f"[LLM DEBUG] Generated code:\n{code[:500]}...")
            solve_fn = _compile_and_extract_solve(code)
            if solve_fn is None:
                continue

            if _verify_solve_fn(solve_fn, train_pairs):
                print(f"[LLM] OK Generated solve() passes 100% of train pairs for task {task_id}!")
                solve_fn._llm_code = code
                return solve_fn
            else:
                print(f"[LLM] Candidate generated solve() failed verification (attempt {attempt}).")
                # Debug: test each pair individually and show differences
                for i, (inp, out) in enumerate(train_pairs):
                    try:
                        inp = np.asarray(inp, dtype=np.int16)
                        out = np.asarray(out, dtype=np.int16)
                        pred = solve_fn(inp.copy())
                        pred = np.asarray(pred, dtype=np.int16)
                        match = pred.shape == out.shape and np.array_equal(pred, out)
                        diff_count = int((pred != out).sum()) if pred.shape == out.shape else -1
                        print(f"[LLM DEBUG] Pair {i}: shape={pred.shape}, expected={out.shape}, match={match}, diff_cells={diff_count}")
                        if not match and pred.shape == out.shape:
                            # Show first few differences
                            diff_mask = pred != out
                            diff_coords = np.argwhere(diff_mask)
                            if len(diff_coords) > 0:
                                samples = diff_coords[:5]
                                for r, c in samples:
                                    print(f"[LLM DEBUG]   ({r},{c}): pred={pred[r,c]}, expected={out[r,c]}")
                    except Exception as e:
                        print(f"[LLM DEBUG] Pair {i}: ERROR - {e}")

        return None
