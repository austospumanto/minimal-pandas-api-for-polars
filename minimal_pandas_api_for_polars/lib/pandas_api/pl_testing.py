from minimal_pandas_api_for_polars.lib.helpers.types import PF, PS
from minimal_pandas_api_for_polars.lib.pandas_api.pl_assert_frame_equal import (
    pl_assert_frame_equal,
)
from minimal_pandas_api_for_polars.lib.pandas_api.pl_assert_series_equal import (
    pl_assert_series_equal,
)


class pl_testing:
    @staticmethod
    def assert_frame_equal(
        f1: PF,
        f2: PF,
        check_column_order: bool = True,
        null_equal: bool = True,
        fast: bool = False,
    ) -> None:
        return pl_assert_frame_equal(
            f1,
            f2,
            check_column_order=check_column_order,
            null_equal=null_equal,
            fast=fast,
        )

    @staticmethod
    def assert_series_equal(
        s1: PS,
        s2: PS,
        check_name: bool = True,
        null_equal: bool = True,
        fast: bool = False,
    ) -> None:
        return pl_assert_series_equal(
            s1,
            s2,
            check_name=check_name,
            null_equal=null_equal,
            fast=fast,
        )
