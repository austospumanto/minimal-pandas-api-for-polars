from typing import Type, Union, Callable, List, Dict

import numpy as np
import pandas as pd
import polars as pl
from polars.datatypes import DataType

Ints = List[int]
Strs = List[str]
Int2Str = Dict[int, str]
Str2Int = Dict[str, int]
Str2Float = Dict[str, float]
Int2Int = Dict[int, int]
Int2Float = Dict[int, float]
Str2Str = Dict[str, str]
Int2Strs = Dict[int, Strs]
Str2Strs = Dict[str, Strs]
Int2Ints = Dict[int, Ints]
Str2Ints = Dict[str, Ints]

ND = np.ndarray

S = pd.Series
DF = pd.DataFrame
T = pd.Timestamp
MI = pd.MultiIndex
TD = pd.Timedelta
I = pd.Index

PF = pl.DataFrame
PS = pl.Series
PE = pl.Expr

PandasDtypeDirective = str
PolarsDtype = Type[DataType]
PolarsDtypeDirective = Union[PolarsDtype, Type]
DtypeDirective = Union[PandasDtypeDirective, PolarsDtypeDirective]
PolarsDtypeDirectives = Dict[str, PolarsDtypeDirective]
DtypeDirectives = Dict[str, DtypeDirective]
Exprs = List[pl.Expr]
ColumnsRenamingFunc = Callable[[str], str]
ColumnsRenamingDict = Str2Str
ColumnsRenamingDirective = Union[ColumnsRenamingFunc, ColumnsRenamingDict]
