from dataclasses import dataclass

VALID_OPS={
    "IDENTITY","ROTATE","FLIP","CROP","GRAVITY","SCALE","RECOLOR",
    "TRANSPOSE","COLORMAP","TILE","MOSAIC","DOWNSCALE","BORDER","STRIP_BORDER",
    "FILL_HOLES","SYMMETRY_REPAIR","SYMMETRY_REPAIR_CROP","APPLY_PER_REGION","PANEL_LOGIC","PATTERN_COMPLETE",
    "SELECT_RECOLOR","SELECT_CROP",
    "RELOCATE_OBJECT","COPY_OBJECT","TRANSLATE_OBJECT",
    "DELETE_OBJECTS","RANK_RECOLOR","RANK_RESIZE","OBJECTS_TO_STRIP",
    "FRACTAL_TILE","FRACTAL_TILE_INVERSE","REFLECT_TILE","RECOLOR_BY_INDICATOR","DIAGONAL_PATTERN_COMPLETE","MIRROR_4WAY_QUAD","EXTRACT_UNIQUE_COLOR_PANEL","FILL_FRAME_BY_SIZE","CYCLE_BLOCK_EXTEND","SHIFT_PARALLELOGRAM","DIAGONAL_STACK_CHAIN","LLM_50CB2852",
}

@dataclass
class TypeCheckResult:
    valid: bool
    reason: str=""

def check_program(program):
    for ins in program.instructions:
        if ins.op not in VALID_OPS:
            return TypeCheckResult(False,f"unknown op {ins.op}")
        if ins.op=="ROTATE" and ins.args not in [(90,),(180,),(270,)]:
            return TypeCheckResult(False,"invalid rotation")
        if ins.op=="FLIP" and ins.args not in [("H",),("V",)]:
            return TypeCheckResult(False,"invalid flip")
    return TypeCheckResult(True)
