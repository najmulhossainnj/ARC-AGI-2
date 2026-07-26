from arc_solver.dsl.ast import Program,Instruction
from arc_solver.dsl.type_checker import check_program

def test_valid():
    assert check_program(Program((Instruction("IDENTITY"),))).valid
