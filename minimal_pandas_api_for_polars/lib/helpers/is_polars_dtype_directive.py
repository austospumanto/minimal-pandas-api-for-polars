import polars as pl

from minimal_pandas_api_for_polars.lib.helpers.types import DtypeDirective


def is_polars_dtype_directive(dtype_directive: DtypeDirective) -> bool:
    v = dtype_directive
    if isinstance(v, str):
        return False
    #
    elif issubclass(v, pl.DataType):
        return True
    elif v is str:
        return True
    elif v is int:
        return True
    elif v is float:
        return True
    #
    else:
        return False
