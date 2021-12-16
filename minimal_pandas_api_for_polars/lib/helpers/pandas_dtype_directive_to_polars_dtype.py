import polars as pl

from minimal_pandas_api_for_polars.lib.helpers.types import (
    PandasDtypeDirective,
    PolarsDtype,
)


def pandas_dtype_directive_to_polars_dtype(
    dtype_directive: PandasDtypeDirective,
) -> PolarsDtype:
    # noinspection Assert
    assert isinstance(dtype_directive, PandasDtypeDirective) and dtype_directive.strip(), (
        type(dtype_directive),
        dtype_directive,
    )
    v = dtype_directive.strip().lower()
    if v == "int8":
        return pl.Int8
    elif v == "int16":
        return pl.Int16
    elif v == "int32":
        return pl.Int32
    elif v == "int64":
        return pl.Int64
    elif v == "uint8":
        return pl.UInt8
    elif v == "uint16":
        return pl.UInt16
    elif v == "uint32":
        return pl.UInt32
    elif v == "uint64":
        return pl.UInt64
    elif v == "float32":
        return pl.Float32
    elif v == "float64":
        return pl.Float64
    elif v == "bool":
        return pl.Boolean
    elif v == "boolean":
        return pl.Boolean
    elif v == "string":
        return pl.Utf8
    elif v.startswith("datetime64[ns"):
        return pl.Datetime
        # return pl.datatypes.TimestampNanosecond
    elif v.startswith("datetime64[d"):
        return pl.Date
    elif v == "object":
        return pl.Object
    elif v == "category":
        return pl.Categorical
    else:
        raise NotImplementedError(dtype_directive)
