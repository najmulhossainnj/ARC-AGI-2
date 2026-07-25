import math

def bbox_touch(a,b):
    return abs(a.right-b.left)==1 and not (a.bottom<b.top or b.bottom<a.top) or abs(a.bottom-b.top)==1 and not (a.right<b.left or b.right<a.left)

def overlap(a,b):
    return bool(set(a.cells) & set(b.cells))

def distance(a,b):
    ar,ac=a.center; br,bc=b.center
    return math.hypot(ar-br,ac-bc)

def relation_pairs(a,b):
    out=[]
    if a.right < b.left: out.append("left_of")
    if a.left > b.right: out.append("right_of")
    if a.bottom < b.top: out.append("above")
    if a.top > b.bottom: out.append("below")
    if bbox_touch(a,b): out.append("touching")
    if overlap(a,b): out.append("overlapping")
    if set(a.colors)&set(b.colors): out.append("same_color")
    if a.shape_signature()==b.shape_signature(): out.append("same_shape")
    if a.size==b.size: out.append("same_size")
    if a.top==b.top: out.append("aligned_top")
    if a.bottom==b.bottom: out.append("aligned_bottom")
    if a.left==b.left: out.append("aligned_left")
    if a.right==b.right: out.append("aligned_right")
    return out
