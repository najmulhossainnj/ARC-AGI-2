from dataclasses import dataclass, field

@dataclass(frozen=True)
class Relation:
    source: int
    predicate: str
    target: int
    value: object = None

@dataclass
class RelationGraph:
    nodes: dict = field(default_factory=dict)
    edges: list[Relation] = field(default_factory=list)

    def add_node(self, node_id, data):
        self.nodes[node_id]=data

    def add_edge(self, source,predicate,target,value=None):
        self.edges.append(Relation(source,predicate,target,value))

    def has(self, source,predicate,target):
        return any(e.source==source and e.predicate==predicate and e.target==target for e in self.edges)

    def neighbors(self,node_id,predicate=None):
        return [e.target for e in self.edges if e.source==node_id and (predicate is None or e.predicate==predicate)]
