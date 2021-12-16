from datetime import date, datetime, time

import polars as pl

from minimal_pandas_api_for_polars.lib.helpers.types import Strs
from minimal_pandas_api_for_polars.lib.helpers.finalize_select_exprs import (
    finalize_select_exprs,
)
from minimal_pandas_api_for_polars.lib.helpers.to_polars_dtype_directive import (
    to_polars_dtype_directive,
)
from minimal_pandas_api_for_polars.lib.helpers.types import (
    PF,
    DtypeDirectives,
    Exprs,
    PolarsDtypeDirectives,
    DtypeDirective,
    PolarsDtypeDirective,
)


def pl_astype(pf: PF, dtype: DtypeDirectives) -> PF:
    astype_select_exprs: Exprs = get_astype_select_exprs(dtype)

    all_column_names: Strs = pf.columns
    selected_column_names: Strs = list(dtype)
    select_exprs = finalize_select_exprs(
        astype_select_exprs, all_column_names, selected_column_names
    )

    return pf.select(select_exprs)


def to_polars_dtype_directives(dtype: DtypeDirectives) -> PolarsDtypeDirectives:
    return {
        column_name: to_polars_dtype_directive(column_dtype_directive)
        for column_name, column_dtype_directive in dtype.items()
    }


def get_astype_select_exprs(dtype: DtypeDirectives) -> Exprs:
    return [
        get_astype_select_expr(column_name, column_dtype)
        for column_name, column_dtype in dtype.items()
    ]


def get_astype_select_expr(column_name: str, column_dtype_directive: DtypeDirective) -> pl.Expr:
    polars_dtype_directive: PolarsDtypeDirective = to_polars_dtype_directive(column_dtype_directive)
    return pl.col(column_name).cast(polars_dtype_directive)


class TestPlAstype:
    @classmethod
    def run_tests(cls):
        cls.test__to_polars_dtype_directives()

    @classmethod
    def test__to_polars_dtype_directives(cls) -> None:
        params = [
            (
                "string",
                {"A": str, "B": "string", "C": pl.Utf8},
                {"A": str, "B": pl.Utf8, "C": pl.Utf8},
            ),
            (
                "int",
                {"A": int, "C": pl.Int64},
                {"A": int, "C": pl.Int64},
            ),
            (
                "float",
                {"A": float, "C": pl.Float64},
                {"A": float, "C": pl.Float64},
            ),
            (
                "object",
                {"A": object, "B": "object", "C": pl.Object},
                {"A": pl.Object, "B": pl.Object, "C": pl.Object},
            ),
            (
                "list",
                {"A": list, "C": pl.List},
                {"A": pl.List, "C": pl.List},
            ),
            (
                "date",
                {"A": date, "B": "datetime64[D]", "C": pl.Date},
                {"A": pl.Date, "B": pl.Date, "C": pl.Date},
            ),
            (
                "time",
                {"A": time, "C": pl.datatypes.Time},
                {"A": pl.datatypes.Time, "C": pl.datatypes.Time},
            ),
            (
                "datetime",
                {"A": datetime, "B": "datetime64[ns]", "C": pl.Datetime},
                {"A": pl.Datetime, "B": pl.Datetime, "C": pl.Datetime},
            ),
            (
                "bool",
                {"A": bool, "B1": "bool", "B2": "boolean", "C": pl.Boolean},
                {"A": pl.Boolean, "B1": pl.Boolean, "B2": pl.Boolean, "C": pl.Boolean},
            ),
            (
                "category",
                {"B": "category", "C": pl.Categorical},
                {"B": pl.Categorical, "C": pl.Categorical},
            ),
            (
                "integer_strings",
                {
                    "A": "int8",
                    "B": "int16",
                    "C": "int32",
                    "D": "int64",
                    "E": "Int8",
                    "F": "Int16",
                    "G": "Int32",
                    "H": "Int64",
                },
                {
                    "A": pl.Int8,
                    "B": pl.Int16,
                    "C": pl.Int32,
                    "D": pl.Int64,
                    "E": pl.Int8,
                    "F": pl.Int16,
                    "G": pl.Int32,
                    "H": pl.Int64,
                },
            ),
        ]
        for param_name, in_dtypes, out_dtypes__exp in params:
            out_dtypes__act = to_polars_dtype_directives(in_dtypes)
            for colname in in_dtypes:
                out_dtype__exp = out_dtypes__exp[colname]
                out_dtype__act = out_dtypes__act[colname]
                # noinspection Assert
                assert out_dtype__exp == out_dtype__act, (
                    param_name,
                    colname,
                    out_dtype__exp,
                    out_dtype__act,
                )


TestPlAstype.run_tests()
