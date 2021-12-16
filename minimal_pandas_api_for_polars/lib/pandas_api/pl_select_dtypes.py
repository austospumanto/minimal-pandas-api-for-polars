from typing import Type, Optional

import polars as pl
from polars import DataType

from minimal_pandas_api_for_polars.lib.helpers.polars_dtype_to_pandas_dtype_name import (
    polars_dtype_to_pandas_dtype_name,
)
from minimal_pandas_api_for_polars.lib.helpers.polars_dtype_to_python_primitive_type import (
    polars_dtype_to_python_primitive_type,
)


def pl_select_dtypes(obj, include, exclude):
    from minimal_pandas_api_for_polars.lib.polars_utils import plu

    # noinspection Assert
    assert include or exclude, (include, exclude)
    should_include = factory__should_include(include=include)
    should_exclude = factory__should_exclude(exclude=exclude)
    sel = [
        pl.col(c)
        for c, d in obj.pipe(plu).get_dtypes_(as_dict=True, raw=True).items()
        if should_include(c, d) and not should_exclude(c, d)
    ]
    # noinspection Assert
    assert len(sel), (sel, include, exclude)
    selected = obj.select(sel)
    return selected


def factory__should_exclude(exclude):
    if exclude is not None and not isinstance(exclude, list):
        # noinspection Assert
        assert isinstance(exclude, (str, Type[DataType]))
        exclude = [exclude]

    def should_exclude(c: str, d: Type[DataType]) -> bool:
        if exclude is None:
            ret = False
        else:
            ret = polars_dtype_matches_dtype_directives_list(d, ddlist=exclude)
        if ret is None:
            raise NotImplementedError((c, d, exclude))
        return ret

    return should_exclude


def factory__should_include(include):
    if include is not None and not isinstance(include, list):
        # noinspection Assert
        assert isinstance(include, (str, Type[DataType]))
        include = [include]

    def should_include(c: str, d: Type[DataType]) -> bool:
        if include is None:
            ret = True
        else:
            ret = polars_dtype_matches_dtype_directives_list(d, ddlist=include)
        if ret is None:
            raise NotImplementedError((c, d, include))
        return ret

    return should_include


def polars_dtype_matches_dtype_directives_list(d, ddlist) -> Optional[bool]:
    ret = None
    if d in ddlist:
        ret = True
    elif d.__name__ in ddlist:
        ret = True
    elif polars_dtype_to_pandas_dtype_name(d) in ddlist:
        ret = True
    elif polars_dtype_to_python_primitive_type(d) in ddlist:
        ret = True
    elif is_string_pldtype(d):
        ret = any(["string" in ddlist, str in ddlist])
    elif is_integer_pldtype(d):
        ret = any(["integer" in ddlist, "numeric" in ddlist, int in ddlist])
    elif is_floating_pldtype(d):
        ret = any(["floating" in ddlist, "numeric" in ddlist, float in ddlist])
    elif is_object_pldtype(d):
        ret = any(["object" in ddlist, object in ddlist])
    elif is_boolean_pldtype(d):
        ret = any(["boolean" in ddlist, "bool" in ddlist, bool in ddlist])
    elif is_datetime_pldtype(d):
        ret = any(["datetime" in ddlist])
    elif is_list_pldtype(d):
        ret = any(["list" in ddlist])
    return ret


def is_string_pldtype(d):
    return d is pl.Utf8


def is_integer_pldtype(d):
    return d in (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    )


def is_floating_pldtype(d):
    return d in (
        pl.Float32,
        pl.Float64,
    )


def is_datetime_pldtype(d):
    return d in (
        pl.Date,
        pl.Datetime,
        pl.datatypes.Time,
    )


def is_object_pldtype(d):
    return d is pl.Object


def is_boolean_pldtype(d):
    return d is pl.Boolean


def is_list_pldtype(d):
    return d is pl.List
