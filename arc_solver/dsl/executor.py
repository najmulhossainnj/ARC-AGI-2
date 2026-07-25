from .transforms import apply_grid_op
from .type_checker import check_program

def execute(program,grid):
    result=grid.copy()
    check=check_program(program)
    if not check.valid: return None
    try:
        for ins in program.instructions:
            result=apply_grid_op(result,ins.op,ins.args)
            if result is None:
                return None
        return result
    except Exception:
        return None
