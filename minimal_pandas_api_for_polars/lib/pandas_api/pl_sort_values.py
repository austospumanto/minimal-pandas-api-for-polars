from typing import Union, Sequence

from minimal_pandas_api_for_polars.lib.helpers.types import PF
from minimal_pandas_api_for_polars.lib.pandas_api.pl_dropna import SubsetArg

SortDirection = Union[bool, Sequence[bool]]


def invert_sort_direction(sd: SortDirection) -> SortDirection:
    if isinstance(sd, bool):
        return not sd
    else:
        return [not v for v in sd]


def pl_sort_values(
    pf: PF,
    subset: SubsetArg,
    ascending: SortDirection = True,
    inplace: bool = False,
) -> PF:
    descending: SortDirection = invert_sort_direction(ascending)
    return pf.sort(by=subset, reverse=descending, in_place=inplace)
