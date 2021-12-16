from typing import Optional

import polars as pl

from minimal_pandas_api_for_polars.lib.helpers.types import Strs


def get_restcols_select_expr(
    all_column_names: Strs,
    already_selected_column_names: Strs,
) -> Optional[pl.Expr]:
    restcolnames = [c for c in all_column_names if c not in already_selected_column_names]
    if restcolnames:
        restcolnames_patt = r"|".join(restcolnames)
        patt = rf"^{restcolnames_patt}$"
        return pl.col(patt)
