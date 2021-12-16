from functools import wraps


def chainable(fn):
    @wraps(fn)
    def inner(*a, **kw):
        from minimal_pandas_api_for_polars.lib.wrappers.mpp_wrap import (
            mpp_wrap,
        )

        output = fn(*a, **kw)
        return mpp_wrap(output)

    return inner
