def ascii_grid(grid):
    return "\n".join(" ".join(map(str,row)) for row in grid)
