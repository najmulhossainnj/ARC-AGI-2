from collections import Counter
from .components import connected_components
from .shapes import shape_features
from ..core.objects import ARCObject

def extract_objects(grid, background=0, diagonal=False):
    components = connected_components(grid, background, diagonal)
    objects=[]
    for oid,cells in enumerate(components):
        rows=[r for r,c in cells]; cols=[c for r,c in cells]
        bbox=(min(rows),min(cols),max(rows),max(cols))
        colors=tuple(sorted(set(int(grid[r,c]) for r,c in cells)))
        objects.append(ARCObject(oid,tuple(sorted(cells)),colors,bbox,oid))
    return objects
