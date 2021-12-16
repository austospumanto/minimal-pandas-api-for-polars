from datetime import date, datetime, time
from typing import Type

import polars as pl

from minimal_pandas_api_for_polars.lib.helpers.types import PolarsDtype


def convert_builtin_python_class_to_polars_dtype(dtype_directive: Type) -> PolarsDtype:
    if dtype_directive is str:
        return pl.Utf8
    if dtype_directive is bool:
        return pl.Boolean
    if dtype_directive is int:
        return pl.Int64
    if dtype_directive is float:
        return pl.Float64
    elif dtype_directive is list:
        return pl.List
    elif dtype_directive is object:
        return pl.Object
    elif dtype_directive is date:
        return pl.Date
    elif dtype_directive is datetime:
        return pl.Datetime
    elif dtype_directive is time:
        return pl.Time
    else:
        raise NotImplementedError(dtype_directive)
