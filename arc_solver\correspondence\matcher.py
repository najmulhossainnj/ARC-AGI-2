from .similarity import object_similarity

def match_objects(input_objects, output_objects, min_score=0.0):
    """Greedy best-first matching between input and output objects.

    Each input object is paired with its highest-similarity, not-yet-used
    output object. Matching is greedy rather than globally optimal (no
    Hungarian algorithm), but for ARC's typical small object counts this is
    fast and good enough: it's re-run per training pair, many times, inside
    parameter learning.

    Returns a list of (input_object, output_object, score) triples. Input
    objects with no acceptable match (best score below `min_score`, or no
    output objects left) are omitted rather than paired incorrectly.
    """
    pairs = []
    used = set()
    # Resolve matches in order of how "confident" the best available pairing
    # is, so an unambiguous match doesn't get its target stolen by a worse,
    # more ambiguous one processed first.
    candidates = []
    for a in input_objects:
        for b in output_objects:
            candidates.append((object_similarity(a, b), a, b))
    candidates.sort(key=lambda x: -x[0])

    matched_inputs = set()
    for score, a, b in candidates:
        if a.id in matched_inputs or b.id in used:
            continue
        if score < min_score:
            continue
        matched_inputs.add(a.id)
        used.add(b.id)
        pairs.append((a, b, score))
    return pairs


def unmatched(input_objects, output_objects, pairs):
    """Objects that appeared in one scene but weren't matched to the other."""
    matched_in = {a.id for a, _, _ in pairs}
    matched_out = {b.id for _, b, _ in pairs}
    removed = [a for a in input_objects if a.id not in matched_in]
    added = [b for b in output_objects if b.id not in matched_out]
    return removed, added
