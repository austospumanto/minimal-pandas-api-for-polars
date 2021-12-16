import typing as tp

from polars.internals.series import StringNameSpace

from minimal_pandas_api_for_polars.lib.helpers.types import PF, PS


def mpp_wrap(output: tp.Any) -> tp.Any:
    from minimal_pandas_api_for_polars.lib.wrappers.mpp_series import MppSeries
    from minimal_pandas_api_for_polars.lib.wrappers.mpp_frame import MppFrame
    from minimal_pandas_api_for_polars.lib.wrappers.mpp_strns import MppStrns

    if isinstance(output, PF):
        return MppFrame(output)
    elif isinstance(output, PS):
        return MppSeries(output)
    elif isinstance(output, StringNameSpace):
        return MppStrns(output)
    # elif isinstance(output, PE):
    #     return MppExpr(output)
    else:
        return output


def mpp_unwrap(output: tp.Any) -> tp.Any:
    from minimal_pandas_api_for_polars.lib.wrappers.mpp_series import MppSeries
    from minimal_pandas_api_for_polars.lib.wrappers.mpp_frame import MppFrame
    from minimal_pandas_api_for_polars.lib.wrappers.mpp_strns import MppStrns

    if isinstance(output, (MppFrame, MppSeries, MppStrns)):
        return output._
    else:
        return output


mw = mpp_wrap
mu = mpp_unwrap
