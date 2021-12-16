from typing import Any

import polars as pl

from minimal_pandas_api_for_polars.lib.helpers.types import DF, Str2Int, Strs, Str2Str, ND
from minimal_pandas_api_for_polars.lib.helpers.types import PF


def pd2pl(df: DF) -> PF:
    return (
        df
        #
        .pipe(make_convertible)
        #
        .pipe(pl.from_pandas)
    )


def isdictlike(v: Any) -> bool:
    return isinstance(v, dict)


def islistlike(v: Any) -> bool:
    return isinstance(v, (list, ND, tuple))


def make_convertible(df: DF) -> DF:
    df = df.convert_dtypes(
        infer_objects=True,
        convert_string=True,
        convert_integer=False,
        convert_boolean=False,
        convert_floating=False,
    )

    dtyp: Str2Str = {}

    df_notna: DF = df.notna()
    n_notnas_by_col: Str2Int = df_notna.sum().to_dict()
    all_null_cols: Strs = [cc for cc, nn in n_notnas_by_col.items() if nn == 0]
    dtyp.update({cc: "string" for cc in all_null_cols})

    # df_indices: ND = np.arange(0, len(df), dtype="int64")
    # asgn = {}
    # objcolnames: Strs = df.dtypes.map(str)[lambda x: (x == "object")].index.tolist()
    # for objcolname in objcolnames:
    #     if dtyp.get(objcolname) == "string":
    #         continue
    #     objcolvals_notna: ND = df_notna[objcolname].to_numpy()
    #     nonnull_objcolval_indices: ND = df_indices[objcolvals_notna]
    #     first_nonnull_objcolval_index: int = np.min(nonnull_objcolval_indices)
    #     first_nonnull_objcolval: Any = df.iloc[
    #         first_nonnull_objcolval_index, df.columns.tolist().index(objcolname)
    #     ]
    #     print(
    #         S(
    #             {
    #                 "objcolname": objcolname,
    #                 "first_nonnull_objcolval_index": first_nonnull_objcolval_index,
    #                 "first_nonnull_objcolval": first_nonnull_objcolval,
    #                 "type(first_nonnull_objcolval)": type(first_nonnull_objcolval),
    #             }
    #         )
    #     )
    #     if isdictlike(first_nonnull_objcolval) or islistlike(first_nonnull_objcolval):
    #         dtyp[objcolname] = "string"
    #         asgn[objcolname] = df[objcolname].map(ujson.dumps, na_action="ignore")
    # df_convertible = df.assign(**asgn).astype(dtyp)

    df_convertible = df.astype(dtyp)
    return df_convertible
