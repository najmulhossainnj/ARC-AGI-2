from dataclasses import dataclass
from ..dsl.ast import Instruction,Program
from ..dsl.executor import execute
from ..verification.exact import program_error
from .grammar import primitive_programs, COMPOSABLE_FAMILIES, SOLVER_FAMILIES
from .classifier import classify_task, filter_families

@dataclass
class Candidate:
    program: Program
    error: float
    neural_score: float=0.0
    mdl_score: float=0.0
    @property
    def score(self):
        return self.error + self.mdl_score - self.neural_score

class BeamSearcher:
    """
    The grammar has two kinds of primitive:

    - "composable" ops (rotate, flip, crop, gravity, scale, recolor, ...):
      cheap, mostly parameter-free, meant to be chained across depth.
    - "solver" ops (colormap, tile, panel_logic, object_translate, ...):
      data-driven and higher-arity, each trying to explain the whole
      remaining input->output gap in a single instruction.

    Previously solver ops were only ever tried alone, as one-shot depth-1
    programs, and never entered the compositional beam. That made any task
    needing two semantically different steps unreachable even when every
    primitive it needed already existed -- e.g. "rotate, then translate the
    largest object", "recolor, then crop", "colormap, then tile".

    This version folds solver ops into the same beam:

    1. They still get a one-shot pass against the original pairs (so a
       single solver op that already solves the task is found immediately,
       same as before).
    2. The compositional beam is seeded with `Program(())` *and* the most
       promising imperfect solver programs, so composable ops can be
       appended after a solver step.
    3. Once a beam member already has a prefix (composable or a seeded
       solver), solver families are re-learned against the *residual*
       pairs -- train pairs passed through that prefix -- so a solver op's
       parameters (an offset, a colormap, a panel-logic axis, ...) are
       correct for what the prefix actually produced, not just for the raw
       task input. This is what lets "rotate, then translate" work: the
       translation vector has to be learned on the rotated grid, not the
       original one.

    Re-deriving solver families is more expensive than reusing the fixed
    composable list, so `solver_rederive_width` can cap it to only the
    top-scoring N beam members each depth instead of all `beam_width` of
    them. It defaults to None (the whole beam) -- a low-error-but-wrong-shape
    composable prefix (e.g. a RECOLOR guess that happens to zero out a few
    pixels) can easily outrank the *structurally* correct prefix on raw
    pixel error alone, so capping this by default silently cuts off exactly
    the compositions this class exists to find. Pass a smaller value to
    trade completeness for speed once search depth/width, not reachability,
    is the bottleneck.
    """

    def __init__(self,beam_width=100,max_depth=5,solver_seed_width=20,solver_rederive_width=None):
        self.beam_width=beam_width
        self.max_depth=max_depth
        self.solver_seed_width=solver_seed_width
        self.solver_rederive_width=solver_rederive_width

    @staticmethod
    def _residual_pairs(base,train_pairs):
        """train_pairs passed through `base`; None if `base` fails anywhere."""
        residual=[]
        for a,b in train_pairs:
            pred=execute(base,a)
            if pred is None: return None
            residual.append((pred,b))
        return residual

    def search(self,train_pairs,families=None,ranker=None,collector=None):
        if families is None:
            composable_families=COMPOSABLE_FAMILIES
            solver_families=SOLVER_FAMILIES
        else:
            composable_families=families
            solver_families=set()

        cat = classify_task(train_pairs)
        if solver_families:
            solver_families = filter_families(cat, solver_families)

        found=[]
        seen=set()

        def consider(prog):
            key=str(prog)
            if key in seen: return None
            seen.add(key)
            err=program_error(prog,train_pairs)
            ns=ranker.score(prog,train_pairs) if ranker else 0.0
            cand=Candidate(prog,err,ns,0.1*prog.complexity)
            if collector is not None: collector.append(cand)
            if err==0: found.append(cand)
            return cand

        # One-shot solver programs against the original pairs (unchanged
        # from before: a single solver op that already solves the task is
        # found here immediately).
        solver_candidates=[]
        if solver_families:
            for prog in primitive_programs(train_pairs, families=solver_families):
                cand=consider(prog)
                if cand is not None: solver_candidates.append(cand)
                if found:
                    found.sort(key=lambda c: c.score)
                    return found, []

        # Seed the compositional beam with `Program(())` *and* the most
        # promising imperfect solver programs, so composable ops (and,
        # below, re-derived solver ops) can be chained around a solver step
        # instead of it being a dead end when it doesn't solve the task
        # outright.
        solver_candidates.sort(key=lambda c:c.score)
        beam=[Program(())] + [
            c.program for c in solver_candidates if c.error>0
        ][:self.solver_seed_width]

        for depth in range(self.max_depth):
            expanded=[]
            # If capped, only the top-scoring bases pay for a fresh
            # solver-family re-derivation each depth; the rest are still
            # extended with the cheap composable ops. Uncapped (default),
            # every current beam member is eligible.
            if self.solver_rederive_width is None:
                rederive=None
            else:
                rederive={str(b) for b in beam[:self.solver_rederive_width]}
            for base in beam:
                residual=self._residual_pairs(base,train_pairs)
                if residual is None: continue
                step_pool=primitive_programs(residual, families=composable_families)
                if solver_families and base.instructions and (rederive is None or str(base) in rederive):
                    step_pool=step_pool + primitive_programs(residual, families=solver_families)
                for p in step_pool:
                    prog=Program(base.instructions+p.instructions)
                    cand=consider(prog)
                    if cand is not None: expanded.append(cand)
            expanded.sort(key=lambda x:x.score)
            beam=[x.program for x in expanded[:self.beam_width]]
            if not beam: break
        found.sort(key=lambda x:x.score)
        return found, beam
