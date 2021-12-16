def fqn(fn):
    """Fully-Qualified Name"""
    return f"{fn.__module__}.{fn.__qualname__}"
