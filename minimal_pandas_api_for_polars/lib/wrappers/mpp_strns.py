from __future__ import annotations

import inspect
import re
from re import Pattern
from typing import cast, Union

import polars as pl
from polars.internals.series import StringNameSpace

from minimal_pandas_api_for_polars.lib.helpers.types import PS, PF
from minimal_pandas_api_for_polars.lib.polars_utils import PolarsUtils
from minimal_pandas_api_for_polars.lib.wrappers.chainable import chainable
from minimal_pandas_api_for_polars.lib.wrappers.mpp_series import Ser
from minimal_pandas_api_for_polars.lib.wrappers.mpp_wrap import mpp_wrap
from minimal_pandas_api_for_polars.lib.helpers.testing_utils import assert_isinstance


class MppStrns:
    _obj: StringNameSpace
    _plu: PolarsUtils

    def __init__(self, obj: StringNameSpace):
        assert_isinstance(obj, StringNameSpace)
        self._obj = obj

    @property
    def _(self) -> StringNameSpace:
        return self._obj

    @property
    def ps(self) -> PS:
        return pl.wrap_s(self._obj._s)

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

    @chainable
    def cat(self, s: Union[str, PS], sep: str = "") -> Ser:
        from minimal_pandas_api_for_polars.lib.wrappers.mpp_series import MS

        def get_left() -> Ser:
            return self.ps.rename("left")

        def get_right(left: PS) -> Ser:
            if isinstance(s, str):
                return pl.repeat(s, n=len(left), name="right")
            elif isinstance(s, MS):
                return s._obj.rename("right")
            elif isinstance(s, PS):
                return s.rename("right")
            else:
                raise NotImplementedError(type(s))

        def get_pf() -> PF:
            left = get_left()
            right = get_right(left)
            return PF({"left": left, "right": right})

        exprs = [pl.col("left"), pl.col("right")]
        cat_expr = pl.concat_str(exprs, sep=sep).alias("cat")
        return get_pf().select(cat_expr)["cat"]

    @chainable
    def slice(
        self,
        start=None,
        stop=None,
        step=None,
    ) -> Ser:
        if step:
            raise NotImplementedError()
        assert start is not None, (start, stop, step)
        if stop is None:
            length = len(self.ps) - start
        else:
            length = stop - start
        return self._obj.slice(start, length=length)

    @chainable
    def extract(
        self,
        pattern: Union[str, Pattern],
        expand: bool = False,
        group_index: int = 1,
    ) -> Ser:
        if expand:
            raise NotImplementedError((pattern, expand))
        if isinstance(pattern, Pattern):
            pattern = pattern.pattern
        assert_isinstance(pattern, str)
        return self._obj.extract(pattern=pattern, group_index=group_index)

    @chainable
    def match(self, pat: str) -> Ser:
        if isinstance(pat, Pattern):
            pat = pat.pattern
        assert_isinstance(pat, str)
        if not pat.endswith("$"):
            pat = rf"{pat}$"
        if not pat.startswith("^"):
            pat = rf"^{pat}"
        return self.contains(pat)

    @chainable
    def endswith(self, pat: str, regex: bool = False) -> Ser:
        if isinstance(pat, Pattern):
            pat = pat.pattern
        assert_isinstance(pat, str)
        if not regex:
            pat = re.escape(pat)
        pat = rf"{pat}$"
        return self.contains(pat)

    @chainable
    def format(self, fstring: str, *others: PS) -> Ser:
        from minimal_pandas_api_for_polars.lib.wrappers.mpp_series import MS

        def get_pf() -> PF:
            main = self.ps.rename("main")
            others_dict = {}
            for ix, other in enumerate(others):
                if isinstance(other, MS):
                    other = other._obj
                elif isinstance(other, str):
                    other = pl.repeat(other, n=len(main))
                elif isinstance(other, PS):
                    other = other
                else:
                    raise NotImplementedError((type(other), ix))
                others_dict[f"other_{ix}"] = other.rename(f"other_{ix}")
            return PF({"main": main, **others_dict})

        if others:
            args = [pl.col("main"), *(f"other_{ix}" for ix in range(len(others)))]
        else:
            args = [pl.col("main") for _ in range(fstring.count("{}"))]
        fmt_expr = pl.format(fstring, *args).alias("fmt")
        return get_pf().select(fmt_expr)["fmt"]

    @chainable
    def isalpha(self) -> Ser:
        pat = r"^[a-zA-Z]+$"
        return self.contains(pat)

    @chainable
    def islower(self) -> Ser:
        pat = r"^[^A-Z]+$"
        return self.contains(pat)

    @chainable
    def isnumeric(self) -> Ser:
        pat = r"^[0-9]+$"
        return self.contains(pat)

    @chainable
    def isupper(self) -> Ser:
        pat = r"^[^a-z]+$"
        return self.contains(pat)

    @chainable
    def len(self) -> Ser:
        return self._obj.lengths()

    @chainable
    def lower(self) -> Ser:
        return self._obj.to_lowercase()

    @chainable
    def lstrip(self, to_strip: str = "", regex: bool = False) -> Ser:
        if not to_strip:
            return self._obj.lstrip()
        if not regex:
            to_strip = re.escape(to_strip)
        pat = rf"^({to_strip})+"
        return self.replace(pat, "")

    @chainable
    def rstrip(self, to_strip: str = "", regex: bool = False) -> Ser:
        if not to_strip:
            return self._obj.rstrip()
        if not regex:
            to_strip = re.escape(to_strip)
        pat = rf"({to_strip})+$"
        return self.replace(pat, "")

    @chainable
    def startswith(self, pat: str, regex: bool = False) -> Ser:
        if isinstance(pat, Pattern):
            pat = pat.pattern
        assert_isinstance(pat, str)
        if not regex:
            pat = re.escape(pat)
        pat = rf"^{pat}"
        return self.contains(pat)

    @chainable
    def strip(self, to_strip: str = "", regex: bool = False) -> Ser:
        intermediate = self.lstrip(to_strip, regex).str
        intermediate = cast(CSNS, intermediate)
        return intermediate.rstrip(to_strip, regex)

    @chainable
    def upper(self) -> Ser:
        return self._obj.to_uppercase()


CSNS = MppStrns
