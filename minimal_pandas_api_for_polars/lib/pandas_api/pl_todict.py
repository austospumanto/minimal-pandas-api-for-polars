from typing import Union, Dict, Any, List

import polars as pl

from minimal_pandas_api_for_polars.lib.helpers.types import PF, PS


def pl_todict(
    pf: PF,
    k: Union[str, pl.Expr],
    v: Union[str, pl.Expr],
) -> Dict[Any, Any]:
    selected: PF = pf.select([k, v])

    keys_ser: PS = selected[:, 0]
    values_ser: PS = selected[:, 1]

    keys: List[Any] = keys_ser.to_list()
    values: List[Any] = values_ser.to_list()

    return dict(zip(keys, values))
