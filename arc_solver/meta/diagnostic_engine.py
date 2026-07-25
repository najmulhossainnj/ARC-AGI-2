from __future__ import annotations
"""
diagnostic_engine.py
--------------------
Orchestrates parallel rule-based analyzers + Gemini Flash LLM fallback.
"""
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable, Any

from .analyzers import ALL_ANALYZERS
from .analyzers.base import ProgramCandidate, Analyzer
from .hypothesis_tester import verify_100pct, verify_solve_fn


@dataclass
class DiagnosisResult:
    success: bool = False
    source: str = "none"           # "rule_based" | "llm" | "none"
    candidate: Optional[ProgramCandidate] = None
    solve_fn: Optional[Callable] = None
    analyzer_name: Optional[str] = None
    elapsed: float = 0.0


def _extract_features(
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
) -> dict:
    """Lightweight feature extraction used by analyzers and the LLM prompt."""
    features: dict = {}
    same_size = all(
        np.asarray(i).shape == np.asarray(o).shape
        for i, o in train_pairs
    )
    features["same_size"] = same_size
    features["n_pairs"] = len(train_pairs)

    all_in_colors, all_out_colors = set(), set()
    nonbg_in, diff_fracs = [], []
    for inp, out in train_pairs:
        inp, out = np.asarray(inp), np.asarray(out)
        all_in_colors.update(int(v) for v in np.unique(inp))
        all_out_colors.update(int(v) for v in np.unique(out))
        nonbg_in.append(int((inp != 0).sum()))
        if inp.shape == out.shape:
            diff_fracs.append(float((inp != out).sum()) / max(inp.size, 1))

    features["n_colors_in"] = len(all_in_colors)
    features["n_colors_out"] = len(all_out_colors)
    features["avg_nonbg_in"] = sum(nonbg_in) / max(len(nonbg_in), 1)
    features["avg_diff_frac"] = sum(diff_fracs) / max(len(diff_fracs), 1) if diff_fracs else 0.0
    return features


def _run_analyzer(
    analyzer: Analyzer,
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    features: dict,
) -> Optional[ProgramCandidate]:
    """Run a single analyzer, catching all exceptions safely."""
    try:
        return analyzer.analyze(train_pairs, features)
    except Exception as e:
        print(f"  [Analyzer:{analyzer.name}] notice: {e}")
        return None


class DiagnosticEngine:
    """
    Two-phase diagnostic engine:
    1. Run all rule-based analyzers in parallel (ThreadPoolExecutor).
    2. If all fail, call Gemini Flash for code generation.
    """

    def __init__(
        self,
        analyzers: Optional[List[Analyzer]] = None,
        use_llm: bool = True,
        llm_api_key: Optional[str] = None,
        analyzer_timeout: float = 10.0,
        max_workers: int = 8,
    ):
        self.analyzers = sorted(
            analyzers if analyzers is not None else ALL_ANALYZERS,
            key=lambda a: a.priority,
        )
        self.use_llm = use_llm
        self.llm_api_key = llm_api_key
        self.analyzer_timeout = analyzer_timeout
        self.max_workers = max_workers

    def diagnose(
        self,
        task_id: str,
        train_pairs: List[Tuple[np.ndarray, np.ndarray]],
        test_pairs: Optional[List] = None,
    ) -> DiagnosisResult:
        t0 = time.time()
        features = _extract_features(train_pairs)
        pairs_np = [
            (np.asarray(i, dtype=np.int16), np.asarray(o, dtype=np.int16))
            for i, o in train_pairs
        ]

        print(f"\n[Diagnostic] Task {task_id}: running {len(self.analyzers)} analyzers in parallel...")

        # ── Phase 1: Parallel rule-based analyzers ────────────────────────────
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_analyzer = {
                pool.submit(_run_analyzer, a, pairs_np, features): a
                for a in self.analyzers
            }

            pending = dict(future_to_analyzer)
            verified_candidate: Optional[ProgramCandidate] = None
            winning_analyzer: Optional[Analyzer] = None

            try:
                for future in as_completed(pending.keys(), timeout=self.analyzer_timeout):
                    analyzer = pending[future]
                    try:
                        candidate = future.result()
                    except Exception:
                        continue

                    if candidate is None:
                        continue

                    print(f"  [Analyzer:{analyzer.name}] candidate: {candidate.op} — verifying...")
                    try:
                        if verify_100pct(candidate, pairs_np):
                            print(f"  [Analyzer:{analyzer.name}] OK VERIFIED 100%!")
                            verified_candidate = candidate
                            winning_analyzer = analyzer
                            break
                    except Exception:
                        pass
            except FuturesTimeout:
                print("  [Diagnostic] Parallel analyzers reached total timeout limit.")

        if verified_candidate is not None:
            return DiagnosisResult(
                success=True,
                source="rule_based",
                candidate=verified_candidate,
                analyzer_name=winning_analyzer.name if winning_analyzer else None,
                elapsed=time.time() - t0,
            )

        # ── Phase 2: LLM fallback ─────────────────────────────────────────────
        if not self.use_llm:
            print(f"[Diagnostic] All analyzers failed; LLM disabled. Task {task_id} -> UNRESOLVED.")
            return DiagnosisResult(success=False, source="none", elapsed=time.time() - t0)

        print(f"\n[Diagnostic] All {len(self.analyzers)} analyzers failed -> Gemini Flash fallback...")
        tried_ops = [a.name for a in self.analyzers]

        try:
            from .llm.primitive_codegen import GeminiFlashCodegen
            codegen = GeminiFlashCodegen(api_key=self.llm_api_key, max_retries=2)
            solve_fn = codegen.generate(task_id, pairs_np, features, tried_ops)

            if solve_fn is not None and verify_solve_fn(solve_fn, pairs_np):
                return DiagnosisResult(
                    success=True,
                    source="llm",
                    solve_fn=solve_fn,
                    elapsed=time.time() - t0,
                )
        except Exception as e:
            print(f"[Diagnostic] LLM generator error: {e}")

        print(f"[Diagnostic] FAIL LLM also failed for task {task_id}.")
        return DiagnosisResult(success=False, source="none", elapsed=time.time() - t0)
