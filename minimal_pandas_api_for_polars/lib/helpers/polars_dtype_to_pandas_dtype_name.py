import polars as pl
from polars import DataType

from minimal_pandas_api_for_polars.lib.helpers.types import PolarsDtype


def polars_dtype_to_mpp_dtype_name(
    polars_dtype: PolarsDtype,
    nullable: bool = True,
) -> str:
    # noinspection Assert
    assert issubclass(polars_dtype, DataType), (
        type(polars_dtype),
        polars_dtype,
    )
    v = polars_dtype
    if v is pl.List:
        return "list"
    else:
        return polars_dtype_to_pandas_dtype_name(polars_dtype, nullable)


def polars_dtype_to_pandas_dtype_name(
    polars_dtype: PolarsDtype,
    nullable: bool = True,
) -> str:
    # noinspection Assert
    assert issubclass(polars_dtype, DataType), (
        type(polars_dtype),
        polars_dtype,
    )
    v = polars_dtype
    if v is pl.Int8:
        return "Int8" if nullable else "int8"
    if v is pl.Int16:
        return "Int16" if nullable else "int16"
    if v is pl.Int32:
        return "Int32" if nullable else "int32"
    if v is pl.Int64:
        return "Int64" if nullable else "int64"
    if v is pl.UInt8:
        return "UInt8" if nullable else "uint8"
    if v is pl.UInt16:
        return "UInt16" if nullable else "uint16"
    if v is pl.UInt32:
        return "UInt32" if nullable else "uint32"
    if v is pl.UInt64:
        return "UInt64" if nullable else "uint64"
    if v is pl.Float32:
        return "float32"
    if v is pl.Float64:
        return "float64"
    if v is pl.Boolean:
        return "boolean" if nullable else "bool"
    if v is pl.Utf8:
        return "string"
    if v is pl.datatypes.Datetime:
        return "datetime64[ns]"
    if v is pl.Date:
        return "datetime64[d]"
    if v is pl.datatypes.Time:
        # return "datetime64[ns]"
        # return "timedelta64[ns]"
        raise NotImplementedError(pl.datatypes.Time)
    if v is pl.Object:
        return "object"
    if v is pl.Categorical:
        return "category"
    if v is pl.List:
        return "object"
    raise NotImplementedError(polars_dtype)
