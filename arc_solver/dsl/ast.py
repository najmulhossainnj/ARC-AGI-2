from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Instruction:
    op: str
    args: tuple[Any,...]=()

@dataclass(frozen=True)
class Program:
    instructions: tuple[Instruction,...]

    def __str__(self):
        return " -> ".join(
            i.op + ("" if not i.args else "(" + ",".join(map(str,i.args)) + ")")
            for i in self.instructions
        )
    @property
    def complexity(self):
        return len(self.instructions)
