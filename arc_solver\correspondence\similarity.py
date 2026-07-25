def object_similarity(a,b):
    color = 1.0 if set(a.colors)==set(b.colors) else 0.0
    shape = 1.0 if a.shape_signature()==b.shape_signature() else 0.0
    size = 1.0/(1.0+abs(a.size-b.size))
    pos = 1.0/(1.0+abs(a.top-b.top)+abs(a.left-b.left))
    return 0.4*shape+0.25*color+0.2*size+0.15*pos
