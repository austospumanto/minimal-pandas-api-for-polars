# Minimal Pandas API for Polars

## Install From PyPI:

```pip install minimal-pandas-api-for-polars```

## Example Usage (see tests/test_minimal_pandas_api_for_polars.py for more):

```
import seaborn as sns

from minimal_pandas_api_for_polars.lib.interop.pd_to_pl import pd2pl
from minimal_pandas_api_for_polars.lib.wrappers.mpp_wrap import mpp_wrap
from minimal_pandas_api_for_polars.lib.wrappers.mpp_frame import MF
from minimal_pandas_api_for_polars.lib.helpers.types import DF, PF

df1: DF = sns.load_dataset("titanic")
df2: DF = df1.astype({c: "string" for c, d in df1.dtypes.map(str).items() if d == "category"})
pf: PF = pd2pl(df2)
mf: MF = mpp_wrap(pf)

assert mf["adult_male"].cast(int).sum() == mf["adult_male"].astype("int64").sum()
assert (mf["age"] > 60).any() == (not ((mf["age"] <= 60).all()))
```
