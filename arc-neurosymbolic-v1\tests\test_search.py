import numpy as np
from arc_solver.synthesis.beam_search import BeamSearcher
from arc_solver.dsl.transforms import apply_grid_op
from arc_solver.dsl.object_ops import translate_object

def test_identity_found():
    g=np.array([[1,0],[0,0]])
    valid,_=BeamSearcher(20,2).search([(g,g)])
    assert any(c.error==0 for c in valid)

def test_composable_and_solver_families_compose():
    """A task solvable only by ROTATE followed by TRANSLATE_OBJECT (a
    composable op, then a solver op) must be found by the beam search.
    Guards against solver-family ops only ever being tried alone."""
    positions=[(0,1),(0,2),(0,3),(0,4)]
    def make_pair(r,c):
        g=np.zeros((8,8),dtype=int)
        g[r:r+2,c:c+2]=3
        g[7,7]=7
        rotated=apply_grid_op(g,"ROTATE",(90,))
        out=translate_object(rotated,"COLOR",3,1,1,True,0)
        return g,out
    train_pairs=[make_pair(r,c) for r,c in positions]
    assert all(o is not None for _,o in train_pairs)
    found,_=BeamSearcher(beam_width=60,max_depth=2).search(train_pairs)
    zero=[c for c in found if c.error==0]
    assert zero, "expected a zero-error program combining ROTATE with TRANSLATE_OBJECT"
    assert any(len(c.program.instructions)>=2 for c in zero)

def test_delete_objects_family():
    """DELETE_OBJECTS should be discoverable when the smallest object is
    consistently erased across training pairs."""
    from arc_solver.dsl.object_ops import delete_objects
    def make_pair(r,c):
        g=np.zeros((8,8),dtype=int)
        g[1:3,1:3]=3
        g[r,c]=5
        out=delete_objects(g,"SMALLEST",None,0)
        return g,out
    train_pairs=[make_pair(r,c) for r,c in [(4,4),(4,6),(6,4),(6,6)]]
    found,_=BeamSearcher(beam_width=40,max_depth=1).search(train_pairs)
    zero_ops={i.op for c in found if c.error==0 for i in c.program.instructions}
    assert "DELETE_OBJECTS" in zero_ops

def test_rank_recolor_family():
    """RANK_RECOLOR should recover a consistent rank->color mapping
    (largest object -> 7, smallest -> 8) across training pairs."""
    from arc_solver.dsl.object_ops import recolor_by_rank
    def make_pair(r1,c1,r2,c2):
        g=np.zeros((8,8),dtype=int)
        g[r1:r1+2,c1:c1+2]=3
        g[r2,c2]=5
        out=recolor_by_rank(g,"size_desc",((0,7),(1,8)),0)
        return g,out
    train_pairs=[make_pair(0,0,6,6),make_pair(0,4,6,0),make_pair(2,2,5,6),make_pair(1,1,6,5)]
    found,_=BeamSearcher(beam_width=40,max_depth=1).search(train_pairs)
    zero_ops={i.op for c in found if c.error==0 for i in c.program.instructions}
    assert "RANK_RECOLOR" in zero_ops

def test_rank_resize_family():
    """RANK_RESIZE should recover a consistent rank->scale-factor mapping
    (the largest object doubles in size) across training pairs."""
    from arc_solver.dsl.object_ops import resize_objects_by_rank
    def make_pair(r1,c1,r2,c2):
        g=np.zeros((12,12),dtype=int)
        g[r1:r1+2,c1:c1+2]=3
        g[r2,c2]=5
        out=resize_objects_by_rank(g,"size_desc",((0,2),),0)
        return g,out
    train_pairs=[make_pair(0,0,10,10),make_pair(0,6,10,0),make_pair(4,4,10,8)]
    assert all(o is not None for _,o in train_pairs)
    found,_=BeamSearcher(beam_width=40,max_depth=1).search(train_pairs)
    zero_ops={i.op for c in found if c.error==0 for i in c.program.instructions}
    assert "RANK_RESIZE" in zero_ops

def test_objects_to_strip_family():
    """OBJECTS_TO_STRIP should be discoverable when the output is a 1xN
    strip whose length and colors come from the input's objects -- a
    shape derived from object count, not from the input grid's shape."""
    from arc_solver.dsl.object_ops import objects_to_strip
    def make_pair(r,c):
        g=np.zeros((8,8),dtype=int)
        g[0:3,0:1]=4
        g[r,c]=2
        g[6,6]=6
        out=objects_to_strip(g,"ROW","size_desc",0)
        return g,out
    train_pairs=[make_pair(4,0),make_pair(4,2),make_pair(5,1),make_pair(5,0)]
    assert all(o is not None for _,o in train_pairs)
    found,_=BeamSearcher(beam_width=40,max_depth=1).search(train_pairs)
    zero_ops={i.op for c in found if c.error==0 for i in c.program.instructions}
    assert "OBJECTS_TO_STRIP" in zero_ops

def test_fractal_tile_family():
    """FRACTAL_TILE should be discoverable when the output is the classic
    self-replicating NxN supertile (copy(r,c) = input if input[r,c] is
    non-background, else blank) -- a shape/content pattern TILE alone
    can't express since it depends on the input's own cell values."""
    from arc_solver.dsl.advanced_transforms import fractal_tile
    def make_pair(seed):
        rng=np.random.RandomState(seed)
        g=(rng.rand(3,3)<0.5).astype(int)*6
        out=fractal_tile(g,background=0)
        return g,out
    train_pairs=[make_pair(s) for s in (1,2,3,4)]
    assert all(o is not None for _,o in train_pairs)
    found,_=BeamSearcher(beam_width=40,max_depth=1).search(train_pairs)
    zero_ops={i.op for c in found if c.error==0 for i in c.program.instructions}
    assert "FRACTAL_TILE" in zero_ops

def test_reflect_tile_family():
    """REFLECT_TILE should be discoverable when the output is a tiled
    grid with alternating mirrored row-bands (the general form of the
    'mirror-tiled wallpaper' pattern MOSAIC only covers for a single 2x
    repetition)."""
    from arc_solver.dsl.advanced_transforms import reflect_tile
    def make_pair(seed):
        rng=np.random.RandomState(seed)
        g=(rng.rand(2,2)*4).astype(int)+1
        out=reflect_tile(g,3,3,"H",None)
        return g,out
    train_pairs=[make_pair(s) for s in (10,11,12,13)]
    assert all(o is not None for _,o in train_pairs)
    found,_=BeamSearcher(beam_width=40,max_depth=1).search(train_pairs)
    zero_ops={i.op for c in found if c.error==0 for i in c.program.instructions}
    assert "REFLECT_TILE" in zero_ops
