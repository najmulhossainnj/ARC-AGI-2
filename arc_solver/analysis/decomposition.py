def enabled_families(signature):
    families={"identity","rotate","flip","crop","gravity","recolor","scale"}
    if signature.likely_geometry:
        families |= {"rotate","flip","crop","scale"}
    if signature.likely_recolor:
        families.add("recolor")
    if signature.likely_creation_deletion:
        families |= {"gravity","copy"}
    return families
