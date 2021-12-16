from datetime import datetime, date, time
from typing import Type

import polars as pl
from polars import DataType

from minimal_pandas_api_for_polars.lib.helpers.types import PolarsDtype


def polars_dtype_to_python_primitive_type(polars_dtype: PolarsDtype) -> Type:
    # noinspection Assert
    assert issubclass(polars_dtype, DataType), (
        type(polars_dtype),
        polars_dtype,
    )
    v = polars_dtype
    if v is pl.Int8:
        return int
    if v is pl.Int16:
        return int
    if v is pl.Int32:
        return int
    if v is pl.Int64:
        return int
    if v is pl.UInt8:
        return int
    if v is pl.UInt16:
        return int
    if v is pl.UInt32:
        return int
    if v is pl.UInt64:
        return int
    if v is pl.Float32:
        return float
    if v is pl.Float64:
        return float
    if v is pl.Boolean:
        return bool
    if v is pl.Utf8:
        return str
    if v is pl.Datetime:
        return datetime
    if v is pl.Date:
        return date
    if v is pl.datatypes.Time:
        return time
    if v is pl.Object:
        return object
    if v is pl.List:
        return list
    raise NotImplementedError(polars_dtype)
