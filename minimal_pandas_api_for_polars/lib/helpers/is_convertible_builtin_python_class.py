from datetime import date, time, datetime
from typing import Type

CONVERTIBLE_BUILTIN_PYTHON_CLASSES = [
    str,
    bool,
    int,
    float,
    list,
    object,
    date,
    time,
    datetime,
]


def is_convertible_builtin_python_class(dtype_directive: Type) -> bool:
    return any((dtype_directive is t) for t in CONVERTIBLE_BUILTIN_PYTHON_CLASSES)
