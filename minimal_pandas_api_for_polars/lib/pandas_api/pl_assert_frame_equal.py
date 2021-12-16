from minimal_pandas_api_for_polars.lib.helpers.types import Str2Str
from minimal_pandas_api_for_polars.lib.helpers.types import PF, PS
from minimal_pandas_api_for_polars.lib.helpers.testing_utils import assert_show, assert_set_equal


def pl_assert_frame_equal(
    f1: PF,
    f2: PF,
    check_column_order: bool = True,
    null_equal: bool = True,
    fast: bool = False,
) -> None:
    if fast:
        assert f1.frame_equal(f2, null_equal=null_equal)
    else:
        _pl_assert_same_num_rows(f1, f2)
        _pl_assert_same_column_names(f1, f2, check_column_order=check_column_order)
        _pl_assert_same_column_dtypes(f1, f2)
        _pl_assert_same_column_values(f1, f2, null_equal=null_equal)


def _pl_assert_same_num_rows(f1: PF, f2: PF) -> None:
    assert_show(f1.height, f2.height)


def _pl_assert_same_column_names(
    f1: PF,
    f2: PF,
    check_column_order: bool,
) -> None:
    assert_set_equal(f1.columns, f2.columns)
    if check_column_order:
        assert_show(f1.columns, f2.columns)


def _pl_assert_same_column_dtypes(f1: PF, f2: PF) -> None:
    from minimal_pandas_api_for_polars.lib.polars_utils import plu

    f1_dtypes: Str2Str = plu(f1).get_dtypes_(as_dict=True, raw=False)
    f2_dtypes: Str2Str = plu(f2).get_dtypes_(as_dict=True, raw=False)
    for colname, d1 in f1_dtypes.items():
        d2 = f2_dtypes[colname]
        assert_show(d1, d2, colname)


def _pl_assert_same_column_values(
    f1: PF,
    f2: PF,
    null_equal: bool,
) -> None:
    for c1, c2 in zip(f1, f2):
        assert_show(c1, c2)
        s1: PS = f1[c1]
        s2: PS = f2[c2]
        same_non_null_value: PS = s1 == s2
        if null_equal:
            both_null: PS = s1.is_null() & s2.is_null()
            same_value: PS = same_non_null_value | both_null
        else:
            same_value: PS = same_non_null_value
        num_same_values: int = same_value.sum()
        assert_show(num_same_values, f1.height, s1.name, s2.name)
