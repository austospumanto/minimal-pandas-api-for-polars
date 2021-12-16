from minimal_pandas_api_for_polars.lib.helpers.types import PF


def pl_dtypes(
    pf: PF,
    raw: bool = False,
) -> PF:
    return PF(
        {
            "column": pf.columns,
            "dtype": [(dt if raw else dt.__name__) for dt in pf.dtypes],
        }
    )
