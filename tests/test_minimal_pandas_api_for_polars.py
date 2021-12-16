from typing import TypeVar, Callable
import inspect
import pandas as pd
import pytest
import seaborn as sns
from minimal_pandas_api_for_polars import __version__
from minimal_pandas_api_for_polars.lib.helpers.types import DF
from minimal_pandas_api_for_polars.lib.interop.pd_to_pl import pd2pl
from minimal_pandas_api_for_polars.lib.wrappers.mpp_frame import MF
from minimal_pandas_api_for_polars.lib.wrappers.mpp_wrap import mpp_unwrap, mpp_wrap

FrameT = TypeVar("FrameT", DF, MF)
FnT = Callable[[FrameT], FrameT]


def get_input_data(input_data_name: str) -> DF:
    if input_data_name == "titanic":
        """
        <class 'pandas.core.frame.DataFrame'>
        RangeIndex: 891 entries, 0 to 890
        Data columns (total 15 columns):
         #   Column       Non-Null Count  Dtype
        ---  ------       --------------  -----
         0   survived     891 non-null    int64
         1   pclass       891 non-null    int64
         2   sex          891 non-null    object
         3   age          714 non-null    float64
         4   sibsp        891 non-null    int64
         5   parch        891 non-null    int64
         6   fare         891 non-null    float64
         7   embarked     889 non-null    object
         8   class        891 non-null    category
         9   who          891 non-null    object
         10  adult_male   891 non-null    bool
         11  deck         203 non-null    category
         12  embark_town  889 non-null    object
         13  alive        891 non-null    object
         14  alone        891 non-null    bool
        dtypes: bool(2), category(2), float64(2), int64(4), object(5)
        memory usage: 80.7+ KB
        """
        ret = sns.load_dataset("titanic")
    else:
        raise NotImplementedError(f"input_data_name={input_data_name} is not implemented")

    # TODO: Allow category dtypes
    return ret.convert_dtypes(
        infer_objects=False,
        convert_string=True,
        convert_integer=False,
        convert_boolean=False,
        convert_floating=False,
    ).pipe(convert_category_to_string)


def convert_category_to_string(df: DF) -> DF:
    return df.astype({c: "string" for c, d in df.dtypes.map(str).items() if d == "category"})


def assert_same_output(input_data_name: str, fn: FnT) -> None:
    input_data: DF = get_input_data(input_data_name)
    pandas2pandas_output: DF = fn(input_data).reset_index(drop=True)
    pandas2mpp2mpp2pandas_output: DF = input_data.pipe(pd2pl).pipe(mpp_wrap).pipe(fn).to_pandas_()
    pd.testing.assert_frame_equal(pandas2pandas_output, pandas2mpp2mpp2pandas_output)


def test_version():
    assert __version__ == "0.1.1"


def ma__is_adult(df):
    return df["age"].fillna(0.0) >= 18.0


def fn1(df):
    return df.drop(columns=["deck", "embark_town", "who", "class"])


def fn2(df):
    return df[ma__is_adult]


def fn3(df):
    return df.assign(is_adult=ma__is_adult)


def fn4(df):
    return df.assign(is_adult=ma__is_adult).drop(columns=["age"])


@pytest.mark.parametrize(
    "input_data_name,fn",
    [
        ("titanic", fn1),
        ("titanic", fn2),
        ("titanic", fn3),
        ("titanic", fn4),
    ],
)
def test_same_e2e_pandas_dataframe(input_data_name, fn):
    assert_same_output(input_data_name, fn)
