from typing import Optional

import polars as pl

from minimal_pandas_api_for_polars.lib.helpers.get_restcols_select_expr import (
    get_restcols_select_expr,
)


def finalize_select_exprs(select_exprs, all_column_names, selected_column_names):
    restcols_select_expr: Optional[pl.Expr] = get_restcols_select_expr(
        all_column_names, selected_column_names
    )
    if restcols_select_expr is None:
        select_exprs = select_exprs
    else:
        select_exprs = [restcols_select_expr, *select_exprs]
    return select_exprs
