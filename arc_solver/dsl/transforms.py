import numpy as np
from ..core.grid import crop_non_background
from . import advanced_transforms as adv
from . import object_ops as obj_ops

def gravity(grid,direction,background=0):
    g=np.asarray(grid); h,w=g.shape; out=np.full_like(g,background)
    if direction in ("UP","DOWN"):
        for c in range(w):
            vals=g[:,c][g[:,c]!=background]
            if direction=="UP": out[:len(vals),c]=vals
            else: out[h-len(vals):,c]=vals
    else:
        for r in range(h):
            vals=g[r,:][g[r,:]!=background]
            if direction=="LEFT": out[r,:len(vals)]=vals
            else: out[r,w-len(vals):]=vals
    return out

def apply_grid_op(grid,op,args):
    g=np.asarray(grid)
    if op=="IDENTITY": return g.copy()
    if op=="ROTATE": return np.rot90(g,{90:1,180:2,270:3}[args[0]]).copy()
    if op=="FLIP": return (np.fliplr(g) if args[0]=="H" else np.flipud(g)).copy()
    if op=="CROP": return crop_non_background(g)
    if op=="GRAVITY": return gravity(g,args[0])
    if op=="SCALE":
        f=int(args[0]); return np.repeat(np.repeat(g,f,0),f,1)
    if op=="RECOLOR":
        old,new=args; out=g.copy(); out[g==old]=new; return out
    if op=="TRANSPOSE": return adv.transpose(g)
    if op=="COLORMAP": return adv.apply_colormap(g,args[0])
    if op=="TILE": return adv.tile_grid(g,*args)
    if op=="FRACTAL_TILE": return adv.fractal_tile(g,*args)
    if op=="FRACTAL_TILE_INVERSE": return adv.fractal_tile_inverse(g,*args)
    if op=="REFLECT_TILE": return adv.reflect_tile(g,*args)
    if op=="MOSAIC": return adv.mosaic(g,args[0])
    if op=="DOWNSCALE": return adv.downscale(g,*args)
    if op=="BORDER": return adv.add_border(g,*args)
    if op=="STRIP_BORDER": return adv.strip_border(g,*args)
    if op=="FILL_HOLES": return adv.fill_holes(g,args[0])
    if op=="SYMMETRY_REPAIR": return adv.symmetry_repair(g,args[0])
    if op=="SYMMETRY_REPAIR_CROP": return adv.symmetry_repair_crop(g,args[0])
    if op=="APPLY_PER_REGION": return adv.apply_per_region(g,args[0],args[1:])
    if op=="PANEL_LOGIC": return adv.panel_logic(g,*args)
    if op=="PATTERN_COMPLETE": return adv.pattern_complete(g,*args)
    if op=="SELECT_RECOLOR": return adv.select_recolor(g,*args)
    if op=="SELECT_CROP": return adv.select_crop(g,*args)
    if op=="RELOCATE_OBJECT": return obj_ops.relocate_object(g,*args)
    if op=="COPY_OBJECT": return obj_ops.copy_object(g,*args)
    if op=="TRANSLATE_OBJECT": return obj_ops.translate_object(g,*args)
    if op=="DELETE_OBJECTS": return obj_ops.delete_objects(g,*args)
    if op=="RANK_RECOLOR": return obj_ops.recolor_by_rank(g,*args)
    if op=="RANK_RESIZE": return obj_ops.resize_objects_by_rank(g,*args)
    if op=="OBJECTS_TO_STRIP": return obj_ops.objects_to_strip(g,*args)
    if op=="RECOLOR_BY_INDICATOR": return adv.recolor_by_indicator_feature(g,*args)
    if op=="DIAGONAL_PATTERN_COMPLETE": return adv.diagonal_pattern_complete(g,*args)
    if op=="MIRROR_4WAY_QUAD": return adv.mirror_4way_quad(g,*args)
    if op=="EXTRACT_UNIQUE_COLOR_PANEL": return adv.extract_unique_color_panel(g,*args)
    if op=="FILL_FRAME_BY_SIZE": return adv.fill_frame_by_size(g,*args)
    if op=="CYCLE_BLOCK_EXTEND": return adv.cycle_block_extend(g,*args)
    if op=="SHIFT_PARALLELOGRAM": return adv.shift_parallelogram_fix_right(g,*args)
    if op=="DIAGONAL_STACK_CHAIN": return adv.diagonal_stack_chain(g,*args)
    if op=="LLM_50CB2852": return adv.llm_50cb2852(g,*args)
    raise ValueError(op)
