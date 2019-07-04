
def require_not_none(obj, name: str, raise_error: bool = True) -> bool:
    result: bool = True
    if obj is None:
        if raise_error:
            raise TypeError(name+": None is encountered.")
        else:
            result = False
    return result
