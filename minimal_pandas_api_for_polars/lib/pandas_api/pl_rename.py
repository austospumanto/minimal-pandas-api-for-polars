import inspect

import polars as pl

from minimal_pandas_api_for_polars.lib.helpers.finalize_select_exprs import (
    finalize_select_exprs,
)
from minimal_pandas_api_for_polars.lib.helpers.types import (
    ColumnsRenamingDict,
    ColumnsRenamingDirective,
)
from minimal_pandas_api_for_polars.lib.pandas_api.pl_astype import PF, Exprs


def pl_rename(
    pf: PF,
    columns: ColumnsRenamingDirective,
) -> PF:
    columns_renaming_dict = to_columns_renaming_dict(pf=pf, columns=columns)
    polars_alias_exprs = get_polars_alias_exprs(columns_renaming_dict=columns_renaming_dict)
    select_exprs = finalize_select_exprs(
        polars_alias_exprs, pf.columns, list(columns_renaming_dict)
    )
    return pf.select(select_exprs)


def get_polars_alias_exprs(columns_renaming_dict: ColumnsRenamingDict) -> Exprs:
    # noinspection Assert
    assert isinstance(columns_renaming_dict, dict), (
        type(columns_renaming_dict),
        columns_renaming_dict,
    )
    return [
        get_polars_alias_expr(current_column_name, desired_column_name)
        for current_column_name, desired_column_name in columns_renaming_dict.items()
    ]


def get_polars_alias_expr(current_column_name: str, desired_column_name: str) -> pl.Expr:
    return pl.col(current_column_name).alias(desired_column_name)


def to_columns_renaming_dict(pf: PF, columns: ColumnsRenamingDirective) -> ColumnsRenamingDict:
    if inspect.isfunction(columns):
        return {colname: columns(colname) for colname in pf.columns}
    else:
        return columns
