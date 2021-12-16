from typing import cast, Any, Type, Union

import numpy as np

from minimal_pandas_api_for_polars.lib.helpers.types import (
    Int2Int,
    Int2Float,
    Str2Int,
    Str2Str,
    DF,
    PF,
)


def assert_eq(t1, t2, *to_show):
    assert t1 == t2, (t1, t2, *to_show)


assert_show = assert_eq
assert_equal = assert_eq


def assert_neq(t1, t2):
    assert t1 != t2, (t1, t2)


assert_show_not = assert_neq
assert_not_equal = assert_neq


def assert_gt(v1, v2):
    assert v1 > v2, (v1, v2)


def ensure_sets(fn):
    def inner(s1, s2):
        return fn(set(s1), set(s2))

    return inner


@ensure_sets
def assert_eq__sets(s1, s2):
    assert s1 == s2, (s1 - s2, s2 - s1)


assert_show_set = assert_eq__sets
assert_set_equal = assert_eq__sets
assert_set_show = assert_eq__sets
assert_equal_sets = assert_eq__sets
assert_sets_equal = assert_eq__sets


@ensure_sets
def assert_neq__sets(s1, s2):
    assert s1 != s2, (s1, s2)


@ensure_sets
def assert_disjoint(s1, s2):
    assert s1.isdisjoint(s2), (s1 & s2, s2 - s1, s1 - s2)


@ensure_sets
def assert_subset(s_sub, s_super):
    assert s_sub <= s_super, (s_super - s_sub, s_sub - s_super)


@ensure_sets
def assert_strict_subset(s_sub, s_super):
    assert s_sub < s_super, (s_super - s_sub, s_sub - s_super)


def assert_isinstance(value: Any, types: Union[Type, tuple], *to_show) -> None:
    assert isinstance(value, types), (type(value), value, *to_show)


def assert_isInt2Int(i2i: Int2Int) -> None:
    for k, v in i2i.items():
        assert_isinstance(k, (int, np.integer))
        assert_isinstance(v, (int, np.integer))


def assert_isInt2Float(i2f: Int2Float) -> None:
    for k, v in i2f.items():
        assert_isinstance(k, (int, np.integer))
        assert_isinstance(v, (float, np.floating))


def assert_isStr2Int(s2i: Str2Int) -> None:
    for k, v in s2i.items():
        assert_isinstance(k, str)
        assert_isinstance(v, (int, np.integer))


def assert_isStr2Str(s2s: Str2Str) -> None:
    for k, v in s2s.items():
        assert_isinstance(k, str)
        assert_isinstance(v, str)


def ensure_isInt2Int(obj) -> Int2Int:
    assert_isinstance(obj, dict)
    ret = {}
    for k, v in obj.items():
        if not isinstance(k, (int, np.integer)):
            assert isinstance(k, (bool, np.bool_, str)) or is_float_whole_number(k), (
                k,
                v,
            )
            k = int(k)
        if not isinstance(v, (int, np.integer)):
            assert isinstance(v, (bool, np.bool_, str)) or is_float_whole_number(v), (
                k,
                v,
            )
            v = int(v)
        ret[k] = v
    return ret


def is_float_whole_number(obj) -> bool:
    if isinstance(obj, (float, np.floating)):
        return (obj % 1.0) == 0.0
    else:
        return False


def assert_dtypes(df: Union[DF, PF, "MF", dict], expected_dtypes: Str2Str) -> None:
    from minimal_pandas_api_for_polars.lib.wrappers.mpp_wrap import mpp_wrap
    from minimal_pandas_api_for_polars.lib.wrappers.mpp_frame import MF

    assert_isStr2Str(expected_dtypes)

    if isinstance(df, dict):
        actual_dtypes = df
    elif isinstance(df, DF):
        actual_dtypes = dict(df.dtypes.map(str).to_dict())
    elif isinstance(df, (MF, PF)):
        actual_dtypes = mpp_wrap(df).dtypes_
    else:
        raise NotImplementedError(type(df))
    actual_dtypes: Str2Str = cast(Str2Str, actual_dtypes)

    for colname, exp_dtyp in expected_dtypes.items():
        assert_show(exp_dtyp, actual_dtypes[colname], colname)
