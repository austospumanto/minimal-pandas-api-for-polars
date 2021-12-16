import pandas as pd
import pyarrow as pa

from minimal_pandas_api_for_polars.lib.helpers.types import DF, Strs, Str2Int, Str2Str
from minimal_pandas_api_for_polars.lib.helpers.types import PF
from minimal_pandas_api_for_polars.lib.wrappers.mpp_wrap import mpp_wrap


def pl2pd(pf: PF) -> DF:
    return (
        pf
        #
        .to_pandas(types_mapper=pl2pd_types_mapper)
        #
        .pipe(convert_nonnull_boolean_columns_to_bool_dtype, pf=pf)
    )


to_pandas = pl2pd
polars_to_pandas = pl2pd
polars2pandas = pl2pd
pl_topandas = pl2pd
pl_to_pandas = pl2pd


def pl2pd_types_mapper(typ):
    if typ in (pa.large_string(), pa.string()):
        return pd.StringDtype()
    elif typ == pa.bool_():
        return pd.BooleanDtype()
    elif typ == pa.int64():
        return pd.Int64Dtype()


def convert_nonnull_boolean_columns_to_bool_dtype(df: DF, pf: PF) -> DF:
    df_dtypes = df.dtypes.map(str).to_dict()
    boolean_cols: Strs = [c for c, d in df_dtypes.items() if d == "boolean"]
    Int64_cols: Strs = [c for c, d in df_dtypes.items() if d == "Int64"]
    if boolean_cols or Int64_cols:
        n_nulls_by_col: Str2Int = mpp_wrap(pf)[boolean_cols + Int64_cols].null_count().to_dict()
        dtyp: Str2Str = {
            **{c: "bool" for c in boolean_cols if n_nulls_by_col[c] == 0},
            **{c: "int64" for c in Int64_cols if n_nulls_by_col[c] == 0},
        }
        df = df.astype(dtyp)
    return df
