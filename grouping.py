def replace_values(value, model):
    if model == 1:
        return 10 if value in [0, 1] else 20
    elif model == 2:
        return 10 if value == 2 else 20

def determine_size_group(w, h):
    max_dim = max(w, h)
    if max_dim == 128: return "128×128"
    elif max_dim == 64: return "64×64"
    elif max_dim == 32: return "32×32"
    elif max_dim == 16: return "16×16"
    elif max_dim == 8: return "8×8"
    else: return "4×4"

def determine_area_group(w, h):
    area_to_group = {
        16: "G0", 32: "G1", 64: "G2", 128: "G3",
        256: "G4", 512: "G5", 1024: "G6", 2048: "G7",
        4096: "G8", 8192: "G9", 16384: "G10"
    }
    area = min(w, h) * max(w, h)
    return area_to_group.get(area, "other")

def determine_all_group(w, h):
    return f"{w}×{h}"

def determine_orientation_group(w, h):
    if w == h: return "Square"
    elif w > h: return "Horizontal"
    else: return "Vertical"

def determine_aspect_ratio_group(w, h):
    ratio = max(w, h) / min(w, h)
    if abs(ratio - 1) < 0.01: return "1:1"
    elif abs(ratio - 2) < 0.01: return "2:1"
    elif abs(ratio - 4) < 0.01: return "4:1"
    elif abs(ratio - 8) < 0.01: return "8:1"
    elif abs(ratio - 16) < 0.01: return "16:1"
    elif abs(ratio - 32) < 0.01: return "32:1"
    else: return "other"
