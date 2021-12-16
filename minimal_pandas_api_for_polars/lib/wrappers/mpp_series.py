from __future__ import annotations

import inspect
from datetime import datetime
from typing import Type, Union, Any

import numpy as np
import polars as pl
from polars import DataType

from minimal_pandas_api_for_polars.lib.helpers.types import T, Ints, S, ND
from minimal_pandas_api_for_polars.lib.helpers.to_polars_dtype_directive import (
    to_polars_dtype_directive,
)
from minimal_pandas_api_for_polars.lib.helpers.types import PS
from minimal_pandas_api_for_polars.lib.pandas_api.pl_drop_duplicates import (
    pl_drop_duplicates,
)
from minimal_pandas_api_for_polars.lib.polars_utils import PolarsUtils, plu
from minimal_pandas_api_for_polars.lib.wrappers.chainable import chainable
from minimal_pandas_api_for_polars.lib.wrappers.mpp_wrap import (
    mpp_wrap,
    mpp_unwrap,
)
from minimal_pandas_api_for_polars.lib.wrappers.constants import SERIES_REPR_PATT
from minimal_pandas_api_for_polars.lib.helpers.testing_utils import assert_isinstance, assert_show


class MppSeries:
    _obj: PS
    _plu: PolarsUtils

    def __init__(self, obj: PS):
        assert_isinstance(obj, PS)
        self._obj = obj

    def __repr__(self):
        raw = repr(self._obj)
        return SERIES_REPR_PATT.sub("\nMppSeries: ", raw, 1)

    @property
    def _(self) -> PS:
        return self._obj

    # noinspection PyUnresolvedReferences
    @property
    def str(self) -> "MppStrns":
        from minimal_pandas_api_for_polars.lib.wrappers.mpp_strns import (
            MppStrns,
        )

        return MppStrns(self._obj.str)

    @chainable
    def __getitem__(self, *a, **kw) -> Ser:
        if len(a) == 1 and callable(a[0]):
            predicate_fn = a[0]
            predicate_series: PS = mpp_unwrap(predicate_fn(self))
            return self._obj.filter(predicate_series)
        else:
            if a and isinstance(a[0], MS):
                a = (mpp_unwrap(a[0]), *a[1:])
            return self._obj.__getitem__(*a, **kw)

    def __len__(self) -> int:
        return len(self._obj)

    @chainable
    def __lt__(self, other):
        if isinstance(other, MS):
            other = other._obj
        return self._obj < other

    @chainable
    def __le__(self, other):
        if isinstance(other, MS):
            other = other._obj
        return self._obj <= other

    @chainable
    def __eq__(self, other):
        if isinstance(other, MS):
            other = other._obj
        return self._obj == other

    @chainable
    def __ne__(self, other):
        if isinstance(other, MS):
            other = other._obj
        return self._obj != other

    @chainable
    def __neg__(self):
        return -self._obj

    @chainable
    def __add__(self, other):
        if isinstance(other, MS):
            other = other._obj
        return self._obj + other

    @chainable
    def __sub__(self, other):
        if isinstance(other, MS):
            other = other._obj
        return self._obj - other

    @chainable
    def __deepcopy__(self, **kw):
        return self._obj.__deepcopy__(**kw)

    @chainable
    def __invert__(self):
        return ~self._obj

    @chainable
    def __gt__(self, other):
        if isinstance(other, MS):
            other = other._obj
        return self._obj > other

    @chainable
    def __ge__(self, other):
        if isinstance(other, MS):
            other = other._obj
        return self._obj >= other

    @chainable
    def __and__(self, other):
        if isinstance(other, MS):
            other = other._obj
        return self._obj & other

    @chainable
    def __or__(self, other):
        if isinstance(other, MS):
            other = other._obj
        return self._obj | other

    @chainable
    def __xor__(self, other):
        if isinstance(other, MS):
            other = other._obj
        return self._obj ^ other

    def __getattr__(self, item):
        if hasattr(self._obj, item):
            item_value = getattr(self._obj, item)
        else:
            raise AttributeError(item)

        if inspect.ismethod(item_value):
            ret = chainable(item_value)
        else:
            ret = mpp_wrap(item_value)
        return ret

    def all(self) -> bool:
        assert self.dtype is pl.Boolean, (self.dtype, self.name, self.len())
        num_true = self.sum()
        ret = self.len() == num_true
        assert_isinstance(ret, bool)
        return ret

    def any(self) -> bool:
        assert self.dtype is pl.Boolean, (self.dtype, self.name, self.len())
        num_true = self.sum()
        ret = num_true > 0
        assert_isinstance(ret, bool)
        return ret

    @chainable
    def isin(self, values) -> Ser:
        return self._obj.is_in(values)

    @chainable
    def isna(self) -> Ser:
        return self._obj.is_null()

    @chainable
    def cast(self, dtype: Type, strict: bool = True) -> Ser:
        # noinspection PyTypeChecker
        return self._obj.cast(dtype, strict)

    @chainable
    def unique(self) -> Ser:
        return self._obj.unique()

    @chainable
    def map(self, arg, na_action=None, return_dtype=None) -> Ser:
        if na_action is not None:
            raise NotImplementedError(na_action)

        if isinstance(arg, S):
            arg = arg.to_dict()

        if isinstance(arg, dict):
            sample_key = next(iter(arg.keys()))
            sample_val = next(iter(arg.values()))

            if isinstance(sample_key, str):
                kt = str
            elif isinstance(sample_key, (int, np.integer)):
                kt = int
            elif isinstance(sample_key, (float, np.floating)):
                kt = float
            elif isinstance(sample_key, (bool, np.bool_)):
                kt = bool
            elif isinstance(sample_key, (T, np.datetime64)):
                kt = datetime
            else:
                raise NotImplementedError((type(sample_key), sample_key, arg))

            if isinstance(sample_val, str):
                vt = str
            elif isinstance(sample_val, (int, np.integer)):
                vt = int
            elif isinstance(sample_val, (float, np.floating)):
                vt = float
            elif isinstance(sample_val, (bool, np.bool_)):
                vt = bool
            elif isinstance(sample_val, (T, np.datetime64)):
                vt = datetime
            else:
                raise NotImplementedError((type(sample_val), sample_val, arg))

            # noinspection PyTypeChecker
            return self._obj.cast(kt).apply(arg.get, return_dtype=vt)
        elif callable(arg):
            if return_dtype is None:
                try:
                    return self._obj.apply(arg, return_dtype=None)
                except:
                    pass
                return self._obj.apply(arg, return_dtype=pl.Object)
            else:
                return self._obj.apply(arg, return_dtype=return_dtype)
        else:
            raise NotImplementedError((type(arg), arg))

    @chainable
    def duplicated(self, keep: str = "first") -> Ser:
        if keep == "first":
            # Mark duplicates as True except for the first occurence.
            return self.is_duplicated().mask(self.is_first(), False)
        elif keep is False:
            return self.is_duplicated()
        else:
            raise NotImplementedError(keep)

    @chainable
    def dropna(self) -> Ser:
        return self._obj.drop_nulls()

    @chainable
    def fillna(self, fill_value) -> Ser:
        return self.mask(self.isna(), fill_value)

    @chainable
    def sort_values(
        self,
        ascending: bool = True,
        inplace: bool = False,
    ) -> Ser:
        return self._obj.sort(reverse=not ascending, in_place=inplace)

    @chainable
    def combine_first(self, other: PS) -> Ser:
        if isinstance(other, MS):
            other = other._obj
        assert_isinstance(other, PS)

        # noinspection PyUnresolvedReferences
        mask = self.notna()._obj

        # Where mask evaluates true, take values from self._obj.
        # Where mask evaluates false, take values from other.
        return self._obj.zip_with(mask, other)

    @chainable
    def mask(self, cond, fill_value=None) -> Ser:
        if isinstance(cond, MS):
            fltr = cond._obj
        elif isinstance(cond, PS):
            fltr = cond
        elif inspect.isfunction(cond):
            fltr = mpp_unwrap(cond(self))
        else:
            raise NotImplementedError(type(cond))

        to_show = (cond, fill_value, self.name)
        assert_isinstance(fltr, PS, *to_show)
        assert_show(fltr.len(), self._obj.len(), *to_show)
        assert_show(str(fltr.dtype), str(pl.Boolean), *to_show)

        # if pd.isna(fill_value):
        #     return self._obj
        # else:
        return self._obj.set(fltr, fill_value)

    @chainable
    def astype(self, dtype: Union[str, Type[DataType]], strict: bool = True) -> Ser:
        if isinstance(dtype, str):
            dtype = to_polars_dtype_directive(dtype)
        dtype: Type[DataType]
        assert issubclass(dtype, DataType), dtype
        return self._obj.cast(dtype, strict=strict)

    @chainable
    def drop_duplicates(self) -> Ser:
        return pl_drop_duplicates(self._obj)

    @chainable
    def notna(self) -> Ser:
        return self._obj.is_not_null()

    @chainable
    def nlargest(self, n: int) -> Ser:
        largest_ixs: Ints = self._obj.argsort(reverse=True)[:n].to_list()
        return self._obj.take(largest_ixs)

    @chainable
    def nsmallest(self, n: int) -> Ser:
        smallest_ixs: Ints = self._obj.argsort(reverse=False)[:n].to_list()
        return self._obj.take(smallest_ixs)

    @chainable
    def pipe(self, fn, *a, **kw) -> Any:
        return fn(self, *a, **kw)

    @chainable
    def p(self, fn, *a, **kw) -> Any:
        return fn(self, *a, **kw)

    def nunique(self) -> int:
        # TODO: Add `keepna: bool` or `dropna: bool` argument (follow pandas' API)
        return self._obj.n_unique()

    def tolist(self) -> list:
        return self._obj.to_list()

    def to_numpy(self, *a, **kw) -> ND:
        ret = self._obj.to_numpy()
        if a and len(a) == 1 and str(ret.dtype) != str(a[0]):
            ret = ret.astype(a[0], *a[1:], **kw)
        return ret

    #################
    # Not Chainable #
    #################
    def to_pandas_(self) -> S:
        return plu(self._obj.to_frame()).to_pandas_()[self._obj.name]

    def to_polars_(self) -> PS:
        return self._obj


MS = MppSeries
Ser = Union[MS, PS]
