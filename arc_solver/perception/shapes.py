import numpy as np

def object_mask(grid, obj):
    mask = np.zeros((obj.height,obj.width), dtype=np.int16)
    for r,c in obj.cells:
        mask[r-obj.top,c-obj.left] = grid[r,c]
    return mask

def count_holes(mask):
    from collections import deque
    binary = mask != 0
    h,w = binary.shape
    seen = np.zeros((h,w), bool)
    q=deque()
    for r in range(h):
        for c in range(w):
            if (r in (0,h-1) or c in (0,w-1)) and not binary[r,c] and not seen[r,c]:
                seen[r,c]=True; q.append((r,c))
    while q:
        r,c=q.popleft()
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr,nc=r+dr,c+dc
            if 0<=nr<h and 0<=nc<w and not binary[nr,nc] and not seen[nr,nc]:
                seen[nr,nc]=True; q.append((nr,nc))
    return int(np.sum((~binary) & (~seen)))

def shape_features(grid,obj):
    m=object_mask(grid,obj)
    return {
        "size": obj.size,
        "height": obj.height,
        "width": obj.width,
        "density": obj.size/(obj.height*obj.width),
        "square": obj.height==obj.width,
        "rectangle": obj.size==obj.height*obj.width,
        "horizontal": obj.width>obj.height,
        "vertical": obj.height>obj.width,
        "single_cell": obj.size==1,
        "symmetric_h": np.array_equal(m,np.fliplr(m)),
        "symmetric_v": np.array_equal(m,np.flipud(m)),
        "symmetric_rot": np.array_equal(m,np.rot90(m,2)),
        "holes": count_holes(m),
    }
