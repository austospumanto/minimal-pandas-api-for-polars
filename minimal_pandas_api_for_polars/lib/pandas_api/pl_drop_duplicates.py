from typing import TypeVar

import polars as pl

from minimal_pandas_api_for_polars.lib.helpers.types import PF, PS
from minimal_pandas_api_for_polars.lib.pandas_api.pl_dropna import SubsetArg
from minimal_pandas_api_for_polars.lib.helpers.testing_utils import assert_show

ObjT = TypeVar("ObjT", PF, PS)


def pl_drop_duplicates(
    obj: ObjT,
    subset: SubsetArg = None,
    keep: str = "first",
) -> ObjT:
    if isinstance(obj, PF):
        pf: PF = obj
        if keep == "first":
            return pf.drop_duplicates(subset=subset)
        elif keep == "last":
            return pf.groupby(subset, maintain_order=True).agg(
                pl.exclude(subset).last().keep_name()
            )
        else:
            raise NotImplementedError(keep)
    elif isinstance(obj, PS):
        assert_show(subset, None)
        assert_show(keep, "first")
        ps: PS = obj
        pf: PF = ps.to_frame()
        res: PF = pl_drop_duplicates(pf, subset=None, keep="first")
        return res[:, 0]
    else:
        raise NotImplementedError(type(obj))
