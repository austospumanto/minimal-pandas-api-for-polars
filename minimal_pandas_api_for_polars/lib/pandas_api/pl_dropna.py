from typing import Union, List

import polars as pl

from minimal_pandas_api_for_polars.lib.helpers.types import PF

SubsetArg = Union[
    None,
    str,
    pl.Expr,
    List[str],
    List[pl.Expr],
]
SubsetExprs = Union[pl.Expr, List[pl.Expr]]


def pl_dropna(
    pf: PF,
    axis="index",
    how="any",
    subset: SubsetArg = None,
) -> PF:
    if axis in (0, "index"):
        return _pl_dropna_index(pf, how=how, subset=subset)
    elif axis in (1, "columns"):
        return _pl_dropna_columns(pf, how=how)
    else:
        raise NotImplementedError(axis)


def _pl_dropna_columns(pf: PF, how="any") -> PF:
    if how == "any":
        return _pl_dropna_columns_any(pf)
    elif how == "all":
        return _pl_dropna_columns_all(pf)
    else:
        raise NotImplementedError(how)


def _pl_dropna_columns_all(pf: PF) -> PF:
    return pf[:, [not (s.null_count() == pf.height) for s in pf]]


def _pl_dropna_columns_any(pf: PF) -> PF:
    return pf[:, [(s.null_count() == 0) for s in pf]]


def _pl_dropna_index(
    pf: PF,
    how="any",
    subset: SubsetArg = None,
) -> PF:
    if how == "any":
        return _pl_dropna_index_any(pf, subset)
    elif how == "all":
        ret = _pl_dropna_index_all(pf, subset)
        return ret
    else:
        raise NotImplementedError(how)


def _pl_dropna_index_all(pf: PF, subset: SubsetArg) -> PF:
    subset_cols_exprs = _subset_2_subset_col_exprs(subset)
    fltr_expr = ~pl.fold(
        acc=True,
        f=lambda acc, s: acc & s.is_null(),
        exprs=subset_cols_exprs,
    )
    return pf.filter(fltr_expr)


def _subset_2_subset_col_exprs(subset: SubsetArg) -> SubsetExprs:
    if not subset:
        subset_cols_exprs = pl.all()
    elif isinstance(subset, str):
        subset_cols_exprs = [pl.col(subset)]
    elif isinstance(subset, pl.Expr):
        subset_cols_exprs = [subset]
    elif isinstance(subset, list):
        subset_cols_exprs = []
        for elem in subset:
            if isinstance(elem, str):
                subset_cols_exprs.append(pl.col(elem))
            elif isinstance(elem, pl.Expr):
                subset_cols_exprs.append(elem)
            else:
                raise NotImplementedError((type(elem), elem))
    else:
        raise NotImplementedError((type(subset), subset))
    return subset_cols_exprs


def _pl_dropna_index_any(pf: PF, subset: SubsetArg) -> PF:
    return pf.drop_nulls(subset=subset)
