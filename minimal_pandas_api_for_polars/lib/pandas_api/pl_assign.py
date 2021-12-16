from functools import reduce
from traceback import print_exc
from typing import Tuple, Union, Callable, Any
import pyarrow as pa
import polars as pl
from pandas.core.common import apply_if_callable
from pandas.core.dtypes.inference import is_scalar

from minimal_pandas_api_for_polars.lib.helpers.types import ND, S, I
from minimal_pandas_api_for_polars.lib.helpers.fqn import fqn
from minimal_pandas_api_for_polars.lib.helpers.types import PF


def pl_assign(pf: PF, fast: bool = True, **kwargs) -> PF:
    from minimal_pandas_api_for_polars.lib.wrappers.mpp_wrap import (
        mpp_wrap,
        mpp_unwrap,
    )

    pfx = f"In {fqn(pl_assign)} --"

    if fast:
        y2expr = factory_y2expr(pf)
        reducer = lambda x, y: x.with_column(y2expr(y))
        return reduce(reducer, kwargs.items(), pf)
    else:
        data = pf.clone()

        for k, v in kwargs.items():
            try:
                data[k] = mpp_unwrap(apply_if_callable(v, mpp_wrap(data)))
            except:
                print(
                    f"{pfx} Encountered error for k={repr(k)}. Will print traceback and re-raise.."
                )
                print_exc()
                raise
        return data


def factory_y2expr(pf: PF) -> Callable:
    from minimal_pandas_api_for_polars.lib.wrappers.mpp_series import MppSeries
    from minimal_pandas_api_for_polars.lib.wrappers.mpp_wrap import mpp_wrap

    def y2expr(y: Tuple[str, Union[pl.Expr, Callable, Any]]) -> pl.Expr:
        name, directive = y

        orig_directive_type = type(directive)

        if callable(directive):
            directive = directive(mpp_wrap(pf))

        if is_scalar(directive) and not isinstance(directive, pl.Expr):
            directive = pl.repeat(directive, n=pf.height)

        if isinstance(directive, MppSeries):
            directive = directive._
        elif isinstance(directive, (ND, S, list, tuple, I, pa.Array)):
            directive = pl.Series(directive)

        if not isinstance(directive, (pl.Expr, pl.Series)):
            raise NotImplementedError((type(directive), directive, orig_directive_type))

        return directive.alias(name)

    return y2expr
