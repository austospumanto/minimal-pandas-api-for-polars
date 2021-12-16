from typing import Type

import polars as pl

from minimal_pandas_api_for_polars.lib.helpers.types import PS
from minimal_pandas_api_for_polars.lib.helpers.testing_utils import assert_show


def pl_assert_series_equal(
    s1: PS,
    s2: PS,
    check_name: bool = True,
    null_equal: bool = True,
    fast: bool = False,
) -> None:
    if fast:
        assert s1.series_equal(s2, null_equal=null_equal)
    else:
        _pl_assert_same_num_rows(s1, s2)
        if check_name:
            _pl_assert_same_name(s1, s2)
        _pl_assert_same_dtype(s1, s2)
        _pl_assert_same_values(s1, s2, null_equal=null_equal)


def _pl_assert_same_num_rows(s1: PS, s2: PS) -> None:
    assert_show(s1.len(), s2.len())


def _pl_assert_same_name(s1: PS, s2: PS) -> None:
    assert_show(s1.name, s2.name)


def _pl_assert_same_dtype(s1: PS, s2: PS) -> None:
    d1: Type[pl.DataType] = s1.dtype
    d2: Type[pl.DataType] = s2.dtype
    assert_show(d1, d2)


def _pl_assert_same_values(
    s1: PS,
    s2: PS,
    null_equal: bool,
) -> None:
    same_non_null_value: PS = s1 == s2
    if null_equal:
        both_null: PS = s1.is_null() & s2.is_null()
        same_value: PS = same_non_null_value | both_null
    else:
        same_value: PS = same_non_null_value
    num_same_values: int = same_value.sum()
    assert_show(num_same_values, s1.len(), s1.name, s2.name)
