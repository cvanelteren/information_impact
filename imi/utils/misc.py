def flattenDict(d):
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out += flattenDict(v)
        else:
            if isinstance(v, list):
                out += v
            else:
                out.append(v)
    return out
