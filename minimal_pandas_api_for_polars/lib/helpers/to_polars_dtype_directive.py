from polars.datatypes import DTYPES

from minimal_pandas_api_for_polars.lib.helpers.convert_builtin_python_class_to_polars_dtype import (
    convert_builtin_python_class_to_polars_dtype,
)
from minimal_pandas_api_for_polars.lib.helpers.is_convertible_builtin_python_class import (
    is_convertible_builtin_python_class,
)
from minimal_pandas_api_for_polars.lib.helpers.is_polars_dtype_directive import (
    is_polars_dtype_directive,
)
from minimal_pandas_api_for_polars.lib.helpers.pandas_dtype_directive_to_polars_dtype import (
    pandas_dtype_directive_to_polars_dtype,
)
from minimal_pandas_api_for_polars.lib.helpers.types import (
    DtypeDirective,
    PolarsDtypeDirective,
)


def to_polars_dtype_directive(dtype_directive: DtypeDirective) -> PolarsDtypeDirective:
    if is_polars_dtype_directive(dtype_directive):
        return dtype_directive
    elif is_convertible_builtin_python_class(dtype_directive):
        return convert_builtin_python_class_to_polars_dtype(dtype_directive)
    elif isinstance(dtype_directive, str):
        if dtype_directive in POLARS_DATATYPE_BY_NAME:
            return POLARS_DATATYPE_BY_NAME[dtype_directive]
        else:
            return pandas_dtype_directive_to_polars_dtype(dtype_directive)
    else:
        raise NotImplementedError(dtype_directive)


POLARS_DATATYPE_BY_NAME = {dt.__name__: dt for dt in DTYPES}
