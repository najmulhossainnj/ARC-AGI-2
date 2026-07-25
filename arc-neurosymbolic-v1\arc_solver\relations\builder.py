from .graph import RelationGraph
from .predicates import relation_pairs, distance

def build_relation_graph(grid, objects):
    g=RelationGraph()
    for o in objects:
        g.add_node(o.id, o)
    for a in objects:
        for b in objects:
            if a.id==b.id: continue
            for p in relation_pairs(a,b):
                g.add_edge(a.id,p,b.id)
            g.add_edge(a.id,"distance",b.id,distance(a,b))
    return g
