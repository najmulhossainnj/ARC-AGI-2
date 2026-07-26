from itertools import product
import numpy as np

from ..dsl.ast import Instruction,Program
from ..dsl.primitives import ROTATIONS,FLIPS,GRAVITIES,SCALES
from . import param_learning as pl

MOSAIC_MODES = ("H_MIRROR","H_PLAIN","V_MIRROR","V_PLAIN","QUAD_MIRROR")
LOGIC_OPS = ("AND","OR","XOR","DIFF")

# "Composable" families are cheap, mostly parameter-free ops meant to be
# chained together across search depth (as in the original solver).
COMPOSABLE_FAMILIES = {
    "identity","rotate","flip","crop","gravity","scale","recolor","transpose","panel_logic","tile","colormap",
}

# "Solver" families are data-driven, higher-arity, one-shot transforms
# (each is trying to solve the whole task in a single instruction). They are
# evaluated once each rather than composed across depth, so their candidate
# counts can be larger without causing combinatorial blow-up.
SOLVER_FAMILIES = {
    "colormap","tile","mosaic","downscale","border","fill_holes",
    "symmetry_repair","symmetry_repair_crop","apply_per_region","panel_logic","pattern_complete",
    "select_recolor","select_crop","object_relocate","object_translate",
    "delete_objects","rank_recolor","rank_resize","objects_to_strip",
    "fractal_tile","fractal_tile_inverse","reflect_tile","recolor_by_indicator","diagonal_pattern_complete",
    "mirror_4way_quad","extract_unique_color_panel","fill_frame_by_size","cycle_block_extend","shift_parallelogram","diagonal_stack_chain","ray_trace",
}

ALL_FAMILIES = COMPOSABLE_FAMILIES | SOLVER_FAMILIES

def primitive_programs(train_pairs, families=None):
    families = families or ALL_FAMILIES
    out=[]
    if "identity" in families: out.append(Program((Instruction("IDENTITY"),)))
    if "rotate" in families:
        out += [Program((Instruction("ROTATE",(x,)),)) for x in ROTATIONS]
    if "flip" in families:
        out += [Program((Instruction("FLIP",(x,)),)) for x in FLIPS]
    if "crop" in families: out.append(Program((Instruction("CROP"),)))
    if "gravity" in families:
        out += [Program((Instruction("GRAVITY",(x,)),)) for x in GRAVITIES]
    if "scale" in families:
        out += [Program((Instruction("SCALE",(x,)),)) for x in SCALES]
    if "recolor" in families:
        colors=set()
        for a,b in train_pairs:
            colors.update(map(int,np.unique(a)))
            colors.update(map(int,np.unique(b)))
        for old,new in product(colors,repeat=2):
            if old!=new:
                out.append(Program((Instruction("RECOLOR",(old,new)),)))

    if "transpose" in families:
        out.append(Program((Instruction("TRANSPOSE"),)))

    if "colormap" in families:
        for mapping in pl.learn_colormap(train_pairs):
            out.append(Program((Instruction("COLORMAP",(mapping,)),)))

    if "tile" in families:
        for rh,rw in pl.learn_tile_factors(train_pairs):
            out.append(Program((Instruction("TILE",(rh,rw)),)))

    if "fractal_tile" in families:
        for background in pl.learn_fractal_tile(train_pairs):
            out.append(Program((Instruction("FRACTAL_TILE",(background,)),)))

    if "reflect_tile" in families:
        for rh,rw,row_flip,col_flip in pl.learn_reflect_tile(train_pairs):
            out.append(Program((Instruction(
                "REFLECT_TILE",(rh,rw,row_flip,col_flip)),)))

    if "mosaic" in families:
        for mode in MOSAIC_MODES:
            out.append(Program((Instruction("MOSAIC",(mode,)),)))

    if "downscale" in families:
        for fh,fw in pl.learn_downscale_factors(train_pairs):
            out.append(Program((Instruction("DOWNSCALE",(fh,fw)),)))

    if "border" in families:
        for color,width in pl.learn_border(train_pairs):
            out.append(Program((Instruction("BORDER",(color,width)),)))

    if "fill_holes" in families:
        for color in pl.learn_fill_holes_color(train_pairs):
            out.append(Program((Instruction("FILL_HOLES",(color,)),)))

    if "symmetry_repair" in families:
        for color in pl.symmetry_noise_candidates(train_pairs):
            out.append(Program((Instruction("SYMMETRY_REPAIR",(color,)),)))

    if "symmetry_repair_crop" in families:
        for color in pl.symmetry_noise_candidates(train_pairs):
            out.append(Program((Instruction("SYMMETRY_REPAIR_CROP",(color,)),)))

    if "apply_per_region" in families:
        out.append(Program((Instruction("APPLY_PER_REGION",("PATTERN_COMPLETE",)),)))

    if "panel_logic" in families:
        layouts = pl.learn_panel_logic_layout(train_pairs)
        colors = set()
        for _,o in train_pairs:
            colors.update(map(int,np.unique(o)))
        colors = sorted(colors)[:4] or [0]
        for axis,has_sep in layouts:
            for logic in LOGIC_OPS:
                for out_color in colors:
                    out.append(Program((Instruction(
                        "PANEL_LOGIC",(axis,has_sep,logic,out_color)),)))

    if "pattern_complete" in families:
        for ph,pw in pl.learn_pattern_periods(train_pairs):
            out.append(Program((Instruction("PATTERN_COMPLETE",(ph,pw)),)))

    if "apply_per_region" in families:
        out.append(Program((Instruction("APPLY_PER_REGION",("PATTERN_COMPLETE",)),)))
        out.append(Program((Instruction("APPLY_PER_REGION",("SYMMETRY_REPAIR", 0)),)))

    if "select_recolor" in families:
        selectors,new_colors = pl.select_recolor_candidates(train_pairs)
        for selector,value in selectors:
            for newcolor in new_colors:
                out.append(Program((Instruction(
                    "SELECT_RECOLOR",(selector,value,newcolor)),)))

    if "select_crop" in families:
        selectors,_ = pl.select_recolor_candidates(train_pairs)
        for selector,value in selectors:
            out.append(Program((Instruction("SELECT_CROP",(selector,value)),)))

    if "object_relocate" in families:
        for m_sel,m_val,a_sel,a_val,relation,erase in pl.learn_object_relocation(train_pairs):
            op = "RELOCATE_OBJECT" if erase else "COPY_OBJECT"
            out.append(Program((Instruction(
                op,(m_sel,m_val,a_sel,a_val,relation)),)))

    if "object_translate" in families:
        for sel,val,dr,dc,erase in pl.learn_object_translation(train_pairs):
            out.append(Program((Instruction(
                "TRANSLATE_OBJECT",(sel,val,dr,dc,erase)),)))

    if "delete_objects" in families:
        for sel,val in pl.learn_delete_objects(train_pairs):
            out.append(Program((Instruction("DELETE_OBJECTS",(sel,val)),)))

    if "rank_recolor" in families:
        for rank_key,mapping in pl.learn_rank_recolor(train_pairs):
            out.append(Program((Instruction("RANK_RECOLOR",(rank_key,mapping)),)))

    if "rank_resize" in families:
        for rank_key,mapping in pl.learn_rank_resize(train_pairs):
            out.append(Program((Instruction("RANK_RESIZE",(rank_key,mapping)),)))

    if "objects_to_strip" in families:
        for axis,order in pl.learn_objects_to_strip(train_pairs):
            out.append(Program((Instruction("OBJECTS_TO_STRIP",(axis,order)),)))

    if "recolor_by_indicator" in families:
        for target_c, ind_c, feat_map in pl.learn_recolor_by_indicator_feature(train_pairs):
            out.append(Program((Instruction("RECOLOR_BY_INDICATOR",(target_c, ind_c, feat_map)),)))

    if "diagonal_pattern_complete" in families:
        out.append(Program((Instruction("DIAGONAL_PATTERN_COMPLETE"),)))

    if "fractal_tile_inverse" in families:
        out.append(Program((Instruction("FRACTAL_TILE_INVERSE"),)))

    if "mirror_4way_quad" in families:
        out.append(Program((Instruction("MIRROR_4WAY_QUAD"),)))

    if "extract_unique_color_panel" in families:
        out.append(Program((Instruction("EXTRACT_UNIQUE_COLOR_PANEL"),)))

    if "fill_frame_by_size" in families:
        out.append(Program((Instruction("FILL_FRAME_BY_SIZE"),)))

    if "cycle_block_extend" in families:
        for block_h, num_blocks, recolor_tuples in pl.learn_cycle_block_extend(train_pairs):
            out.append(Program((Instruction("CYCLE_BLOCK_EXTEND", (block_h, num_blocks, recolor_tuples)),)))

    if "shift_parallelogram" in families:
        out.append(Program((Instruction("SHIFT_PARALLELOGRAM"),)))

    if "diagonal_stack_chain" in families:
        out.append(Program((Instruction("DIAGONAL_STACK_CHAIN"),)))

    if "ray_trace" in families:
        colors = set()
        for a, b in train_pairs:
            colors.update(map(int, np.unique(a)))
            colors.update(map(int, np.unique(b)))
        colors.discard(0)
        if not colors: colors = {1}
        for sc in colors:
            for pc in colors:
                for dm in ('DIAGONAL', 'ORTHOGONAL'):
                    for bw in (True, False):
                        for bo in (True, False):
                            out.append(Program((Instruction("RAY_TRACE", (sc, pc, dm, bw, bo)),)))

    return out
