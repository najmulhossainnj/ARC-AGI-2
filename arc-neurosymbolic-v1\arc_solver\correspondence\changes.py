def translation(a, b):
    """(dr, dc) that would move a's bbox onto b's bbox, if shape is preserved."""
    if a.shape_signature() != b.shape_signature():
        return None
    return (b.top - a.top, b.left - a.left)


def classify_change(a,b):
    if a.shape_signature()==b.shape_signature() and a.colors==b.colors and a.bbox==b.bbox:
        return "unchanged"
    if a.shape_signature()==b.shape_signature() and a.colors!=b.colors:
        return "recolor"
    if a.shape_signature()==b.shape_signature() and a.bbox!=b.bbox:
        return "move"
    if a.size!=b.size:
        return "resize"
    return "transform"
