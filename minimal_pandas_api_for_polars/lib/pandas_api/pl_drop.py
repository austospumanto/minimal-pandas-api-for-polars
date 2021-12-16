import polars as pl

from minimal_pandas_api_for_polars.lib.helpers.types import Strs
from minimal_pandas_api_for_polars.lib.helpers.types import PF


def pl_drop(pf: PF, columns: Strs) -> PF:
    if isinstance(columns, str):
        columns = [columns]
    sel = pl.all()
    for c in columns:
        sel = sel.exclude(c)
    return pf.select(sel)
