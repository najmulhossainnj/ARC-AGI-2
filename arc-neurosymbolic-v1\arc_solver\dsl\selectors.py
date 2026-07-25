def select_objects(scene, selector, value=None):
    objs=scene.objects
    if selector=="ALL": return objs
    if selector=="COLOR": return [o for o in objs if value in o.colors]
    if selector=="LARGEST": return [max(objs,key=lambda x:x.size)] if objs else []
    if selector=="SMALLEST": return [min(objs,key=lambda x:x.size)] if objs else []
    if selector=="UNIQUE_COLOR":
        counts={}
        for o in objs:
            for c in o.colors: counts[c]=counts.get(c,0)+1
        return [o for o in objs if any(counts[c]==1 for c in o.colors)]
    return []
