from __future__ import annotations

import inspect
import typing as tp

import polars as pl
from pandas.core.dtypes.inference import is_scalar
from polars import from_pandas

from minimal_pandas_api_for_polars.lib.helpers.types import DF, ND, Strs, Str2Str, S
from minimal_pandas_api_for_polars.lib.helpers.polars_dtype_to_pandas_dtype_name import (
    polars_dtype_to_mpp_dtype_name,
)
from minimal_pandas_api_for_polars.lib.helpers.types import PF, PS, PE
from minimal_pandas_api_for_polars.lib.interop.pd_to_pl import pd2pl
from minimal_pandas_api_for_polars.lib.polars_utils import PolarsUtils, plu
from minimal_pandas_api_for_polars.lib.wrappers.chainable import chainable
from minimal_pandas_api_for_polars.lib.wrappers.mpp_series import MS
from minimal_pandas_api_for_polars.lib.wrappers.mpp_wrap import (
    mpp_wrap,
    mpp_unwrap,
)
from minimal_pandas_api_for_polars.lib.wrappers.constants import FRAME_REPR_PATT
from minimal_pandas_api_for_polars.lib.helpers.testing_utils import assert_isinstance


class MppFrame:
    _obj: PF
    _plu: PolarsUtils

    def __init__(self, obj: PF):
        assert_isinstance(obj, PF)
        self._obj = obj
        self._plu = plu(self._obj)

    def __repr__(self):
        raw = repr(self._obj)
        return FRAME_REPR_PATT.sub(r"\1MppFrame: \n", raw, 1)

    @classmethod
    def from_records(cls, records: tp.List[dict]) -> MF:
        return cls(DF.from_records(records).p(pd2pl))

    @property
    def loc(self) -> MppFrameLoc:
        return MppFrameLoc(self)

    @property
    def iloc(self) -> MppFrameLoc:
        return MppFrameLoc(self)

    @property
    def empty(self) -> bool:
        return len(self) == 0

    @property
    def _(self) -> PF:
        return self._obj

    @property
    def T(self) -> F:
        return mpp_wrap(self._obj.transpose())

    #############
    # Chainable #
    #############

    def __getitem__(self, item: tp.Any) -> tp.Union[MF, MS, tp.Any]:
        if not isinstance(item, tuple):
            item = (item,)
        first_arg, *rest_args = item

        if inspect.isfunction(first_arg) or inspect.ismethod(first_arg):
            # Convert `first_arg` from a predicate function to a predicate series
            first_arg = mpp_unwrap(first_arg(self))

        if isinstance(first_arg, ND):
            first_arg = PS(first_arg)
        elif isinstance(first_arg, MS):
            first_arg = mpp_unwrap(first_arg)
        elif isinstance(first_arg, S):
            first_arg = from_pandas(first_arg)
        if rest_args:
            item = (first_arg, *rest_args)
        else:
            item = first_arg
        ret = self._obj.__getitem__(item)
        if (
            rest_args
            and is_scalar(rest_args[0])
            and isinstance(ret, (PF, MF))
            and ret.shape[1] == 1
        ):
            ret = ret[:, 0]
        return mpp_wrap(ret)

    def __iter__(self):
        return iter(list(self._obj.columns))

    def __len__(self):
        return self._obj.height

    def __getattr__(self, item):
        if hasattr(self._plu, item):
            item_value = getattr(self._plu, item)
        elif hasattr(self._obj, item):
            item_value = getattr(self._obj, item)
        else:
            raise AttributeError(item)

        if inspect.ismethod(item_value):
            ret = chainable(item_value)
        else:
            ret = mpp_wrap(item_value)
        return ret

    @chainable
    def drop(self, columns: Strs) -> F:
        return self._plu.drop(columns=columns)

    @chainable
    def squeeze(self) -> tp.Union[PF, MF, PS, MS, S]:
        n_rows, n_cols = self.shape
        if n_rows == 1 and n_cols == 1:
            return self._obj[0, 0]
        elif n_cols == 1:
            return self._obj[:, 0]
        elif n_rows == 1:
            return S(self.to_pandas_().squeeze())
        else:
            return self

    @chainable
    def notna(self) -> F:
        return self._obj.select(pl.all().is_not_null().keep_name())

    @chainable
    def isna(self) -> F:
        return self._obj.select(pl.all().is_null().keep_name())

    def info(self) -> None:
        self._plu.info()

    @chainable
    def pipe(self, fn, *a, **kw) -> tp.Any:
        return fn(self, *a, **kw)

    @chainable
    def p(self, fn, *a, **kw) -> tp.Any:
        return fn(self, *a, **kw)

    @chainable
    def nunique(self) -> F:
        return self._obj.n_unique()

    @chainable
    def nlargest(self, n: int, subset: tp.List[tp.Union[str, PE]]) -> F:
        reverse = [True for _ in subset]
        largest_ixs: MS = self.select(
            pl.argsort_by(subset, reverse=reverse).slice(0, n).alias("largest_ixs")
        )["largest_ixs"]
        assert_isinstance(largest_ixs, MS)
        return self[largest_ixs]

    @chainable
    def nsmallest(self, n: int, subset: tp.List[tp.Union[str, PE]]) -> F:
        reverse = [False for _ in subset]
        smallest_ixs: MS = self.select(
            pl.argsort_by(subset, reverse=reverse).slice(0, n).alias("smallest_ixs")
        )["smallest_ixs"]
        assert_isinstance(smallest_ixs, MS)
        return self[smallest_ixs]

    @chainable
    def to_dict(self, *a, **kw):
        if not a and not kw and self._obj.height == 1:
            return {ps.name: ps[0] for ps in self._obj}
        else:
            return self._obj.to_dict(*a, **kw)

    #################
    # Not Chainable #
    #################
    def to_pandas_(self) -> DF:
        return self._plu.to_pandas_()

    def to_polars_(self) -> PF:
        return self._plu.to_polars_()

    def zip_columns_as_dict_(self, *a, **kw) -> tp.Dict[tp.Any, tp.Any]:
        return self._plu.zip_columns_as_dict_(*a, **kw)

    @property
    def dtypes_(self) -> Str2Str:
        columns = self._obj.columns
        dtypes = self._obj.dtypes
        has_nulls = {k: bool(v > 0) for k, v in self.null_count().to_dict().items()}
        return {
            c: polars_dtype_to_mpp_dtype_name(
                polars_dtype=d,
                nullable=has_nulls[c],
            )
            for c, d in zip(columns, dtypes)
        }


MF = MppFrame
F = tp.Union[MF, PF]


class MppFrameLoc:
    _cf: MF

    def __init__(self, cf: MF):
        self._cf = cf

    def __getitem__(self, item):
        return self._cf.__getitem__(item)
