def mdl_score(program):
    constants=sum(len(i.args) for i in program.instructions)
    return 0.1*len(program.instructions)+0.03*constants
