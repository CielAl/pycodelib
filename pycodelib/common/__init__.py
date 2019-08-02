__validate_message = ": None is encountered."


def require_not_none(obj, name: str, raise_error: bool = True) -> bool:
    result: bool = True
    if obj is None:
        if raise_error:
            raise TypeError(f"{name}{__validate_message}")
        else:
            result = False
    return result


def default_not_none(obj, default):
    if not require_not_none(obj, name='object', raise_error=False):
        return default
    return obj
