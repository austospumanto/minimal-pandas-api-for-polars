import re
from typing import Any, Dict, Union

import polars as pl

from minimal_pandas_api_for_polars.lib.helpers.types import (
    PF,
    DtypeDirectives,
    ColumnsRenamingDirective,
    DF,
    Strs,
)
from minimal_pandas_api_for_polars.lib.pandas_api.pl_assign import pl_assign
from minimal_pandas_api_for_polars.lib.pandas_api.pl_astype import pl_astype
from minimal_pandas_api_for_polars.lib.pandas_api.pl_drop import pl_drop
from minimal_pandas_api_for_polars.lib.pandas_api.pl_drop_duplicates import (
    pl_drop_duplicates,
)
from minimal_pandas_api_for_polars.lib.pandas_api.pl_dropna import pl_dropna, SubsetArg
from minimal_pandas_api_for_polars.lib.pandas_api.pl_dtypes import pl_dtypes
from minimal_pandas_api_for_polars.lib.pandas_api.pl_info import pl_info
from minimal_pandas_api_for_polars.lib.pandas_api.pl_rename import pl_rename
from minimal_pandas_api_for_polars.lib.pandas_api.pl_select_dtypes import pl_select_dtypes
from minimal_pandas_api_for_polars.lib.pandas_api.pl_sort_values import (
    SortDirection,
    pl_sort_values,
)
from minimal_pandas_api_for_polars.lib.pandas_api.pl_testing import pl_testing
from minimal_pandas_api_for_polars.lib.interop.pl_to_pandas import pl2pd
from minimal_pandas_api_for_polars.lib.pandas_api.pl_todict import pl_todict


class PolarsUtils:
    def __init__(self, obj: PF):
        self._obj = obj

    def assign(self, **kwargs) -> PF:
        return pl_assign(self._obj, **kwargs)

    def astype(self, dtype: DtypeDirectives) -> PF:
        return pl_astype(self._obj, dtype=dtype)

    def drop(self, columns: Strs) -> PF:
        return pl_drop(self._obj, columns=columns)

    def drop_duplicates(
        self,
        subset: SubsetArg = None,
        keep: str = "first",
    ) -> PF:
        return pl_drop_duplicates(
            self._obj,
            subset=subset,
            keep=keep,
        )

    def dropna(
        self,
        axis="index",
        how="any",
        subset: SubsetArg = None,
    ) -> PF:
        return pl_dropna(
            self._obj,
            axis=axis,
            how=how,
            subset=subset,
        )

    @property
    def dtypes(self) -> PF:
        return pl_dtypes(self._obj)

    def filter(self, *a, **kw) -> PF:
        if "like" in kw:
            like = kw["like"]
            cols = [c for c in self._obj.columns if like in c]
            if not cols:
                return PF()
            return self._obj.select(cols)
        elif "ilike" in kw:
            ilike = kw["ilike"].lower()
            cols = [c for c in self._obj.columns if ilike in c.lower()]
            if not cols:
                return PF()
            return self._obj.select(cols)
        elif "regex" in kw:
            re_kw = {k: v for k, v in kw.items() if k in ("flags",)}
            rgx = re.compile(kw["regex"], **re_kw)
            cols = list(filter(rgx.match, self._obj.columns))
            if not cols:
                return PF()
            return self._obj.select(cols)
        else:
            return self._obj.filter(*a, **kw)

    def info(self) -> None:
        return pl_info(self._obj)

    def rename(self, columns: ColumnsRenamingDirective) -> PF:
        return pl_rename(self._obj, columns=columns)

    def select_dtypes(
        self,
        include=None,
        exclude=None,
    ) -> PF:
        return pl_select_dtypes(self._obj, include=include, exclude=exclude)

    def sort_values(
        self,
        subset: SubsetArg,
        ascending: SortDirection = True,
        inplace: bool = False,
    ) -> PF:
        return pl_sort_values(
            self._obj,
            subset=subset,
            ascending=ascending,
            inplace=inplace,
        )

    ####################################################
    # Props/Methods that do not appear in pd.DataFrame #
    ####################################################

    def get_dtypes_(
        self,
        as_dict: bool = True,
        raw: bool = False,
    ) -> Union[PF, Dict[Any, Any]]:
        starter = self.raw_dtypes_ if raw else self.dtypes
        if as_dict:
            return self.__class__(starter).zip_columns_as_dict_(k="column", v="dtype")
        else:
            return starter

    @property
    def raw_dtypes_(self) -> PF:
        return pl_dtypes(self._obj, raw=True)

    def to_pandas_(self) -> DF:
        return pl2pd(self._obj)

    def to_polars_(self) -> DF:
        return self._obj.clone()

    def zip_columns_as_dict_(
        self,
        k: Union[str, pl.Expr],
        v: Union[str, pl.Expr],
    ) -> Dict[Any, Any]:
        return pl_todict(self._obj, k=k, v=v)

    testing = pl_testing


class PolarsUtilsMethods:
    astype = lambda obj, *a, **kw: plu(obj).astype(*a, **kw)
    dropna = lambda obj, *a, **kw: plu(obj).dropna(*a, **kw)
    assign = lambda obj, **kw: plu(obj).assign(**kw)
    astype = lambda obj, *a, **kw: plu(obj).astype(*a, **kw)
    drop = lambda obj, *a, **kw: plu(obj).drop(*a, **kw)
    drop_duplicates = lambda obj, *a, **kw: plu(obj).drop_duplicates(*a, **kw)
    dropna = lambda obj, *a, **kw: plu(obj).dropna(*a, **kw)
    dtypes = lambda obj: plu(obj).dtypes
    filter = lambda obj, *a, **kw: plu(obj).filter(*a, **kw)
    info = lambda obj: plu(obj).info()
    rename = lambda obj, *a, **kw: plu(obj).rename(*a, **kw)
    select_dtypes = lambda obj, *a, **kw: plu(obj).select_dtypes(*a, **kw)
    sort_values = lambda obj, *a, **kw: plu(obj).sort_values(*a, **kw)

    get_dtypes_ = lambda obj, *a, **kw: plu(obj).get_dtypes_(*a, **kw)
    raw_dtypes_ = lambda obj: plu(obj).raw_dtypes_
    to_pandas_ = lambda obj: plu(obj).to_pandas_()
    to_polars_ = lambda obj: plu(obj).to_polars_()
    zip_columns_as_dict_ = lambda obj, *a, **kw: plu(obj).zip_columns_as_dict_(*a, **kw)

    testing = pl_testing


plu = PolarsUtils
PLU = PolarsUtils
PlU = PolarsUtils
pu = PolarsUtils
PU = PolarsUtils

plum = PolarsUtilsMethods
PLUM = PolarsUtilsMethods
plm = PolarsUtilsMethods
PLM = PolarsUtilsMethods
pum = PolarsUtilsMethods
PUM = PolarsUtilsMethods
