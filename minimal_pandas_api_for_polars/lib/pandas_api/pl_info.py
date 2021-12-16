from __future__ import annotations

import os
import sys
from abc import ABC, abstractmethod
from typing import Iterable, Mapping, Sequence, IO, List, Iterator, cast

import pandas as pd
import polars as pl
from pandas import Index
from pandas._typing import FrameOrSeriesUnion, Dtype
from pandas.io.formats import format as fmt
from pandas.io.formats.info import _put_str
from pandas.io.formats.printing import pprint_thing

from minimal_pandas_api_for_polars.lib.helpers.types import S
from minimal_pandas_api_for_polars.lib.helpers.types import PF


def pl_info(pf: PF) -> None:
    verbose = True
    buf = None
    max_cols = None
    show_counts = True
    info = DataFrameInfo(
        data=pf,
    )
    info.render(
        buf=buf,
        max_cols=max_cols,
        verbose=verbose,
        show_counts=show_counts,
    )


class BaseInfo(ABC):
    # noinspection PyUnresolvedReferences
    """
    Base class for DataFrameInfo and SeriesInfo.

    Parameters
    ----------
    data : DataFrame or Series
        Either dataframe or series.
    """

    data: FrameOrSeriesUnion

    @property
    @abstractmethod
    def dtypes(self) -> Iterable[Dtype]:
        """
        Dtypes.

        Returns
        -------
        dtypes : sequence
            Dtype of each of the DataFrame's columns (or one series column).
        """

    @property
    @abstractmethod
    def dtype_counts(self) -> Mapping[str, int]:
        """Mapping dtype - number of counts."""

    @property
    @abstractmethod
    def non_null_counts(self) -> Sequence[int]:
        """Sequence of non-null counts for all columns or column (if series)."""

    # noinspection PyIncorrectDocstring
    @abstractmethod
    def render(
        self,
        *,
        buf: IO[str] | None,
        max_cols: int | None,
        verbose: bool | None,
        show_counts: bool | None,
    ) -> None:
        # noinspection PyUnresolvedReferences
        """
                Print a concise summary of a %(klass)s.

                This method prints information about a %(klass)s including
                the index dtype%(type_sub)s and non-null values.
                %(version_added_sub)s\

                Parameters
                ----------
                data : %(klass)s
                    %(klass)s to print information about.
                verbose : bool, optional
                    Whether to print the full summary. By default, the setting in
                    ``pandas.options.display.max_info_columns`` is followed.
                buf : writable buffer, defaults to sys.stdout
                    Where to send the output. By default, the output is printed to
                    sys.stdout. Pass a writable buffer if you need to further process
                    the output.
                %(max_cols_sub)s
                %(show_counts_sub)s

                Returns
                -------
                None
                    This method prints a summary of a %(klass)s and returns None.

                See Also
                --------
                %(see_also_sub)s

                Examples
                --------
                %(examples_sub)s
                """


class DataFrameInfo(BaseInfo):
    """
    Class storing dataframe-specific info.
    """

    def __init__(
        self,
        data: pl.DataFrame,
    ):
        self.data: pl.DataFrame = data

    @property
    def dtype_counts(self) -> Mapping[str, int]:
        return _get_dataframe_dtype_counts(self.data)

    @property
    def dtypes(self) -> Iterable[Dtype]:
        """
        Dtypes.

        Returns
        -------
        dtypes
            Dtype of each of the pl.DataFrame's columns.
        """
        from minimal_pandas_api_for_polars.lib.polars_utils import plu

        return S(self.data.pipe(plu).get_dtypes_(as_dict=True))

    @property
    def ids(self) -> Index:
        """
        Column names.

        Returns
        -------
        ids : Index
            pl.DataFrame's column names.
        """
        return pd.Index(self.data.columns)

    @property
    def row_count(self) -> int:
        """Number of rows."""
        return self.data.height

    @property
    def col_count(self) -> int:
        """Number of columns to be summarized."""
        return len(self.ids)

    @property
    def non_null_counts(self) -> Sequence[int]:
        """Sequence of non-null counts for all columns or column (if series)."""
        ret = (self.data.height - self.data.null_count().transpose()[:, 0]).to_list()
        return cast(Sequence[int], ret)

    def render(
        self,
        *,
        buf: IO[str] | None,
        max_cols: int | None,
        verbose: bool | None,
        show_counts: bool | None,
    ) -> None:
        printer = DataFrameInfoPrinter(
            info=self,
            max_cols=max_cols,
            verbose=verbose,
            show_counts=show_counts,
        )
        printer.to_buffer(buf)


class InfoPrinterAbstract:
    """
    Class for printing dataframe or series info.
    """

    def to_buffer(self, buf: IO[str] | None = None) -> None:
        """Save dataframe info into buffer."""
        table_builder = self._create_table_builder()
        lines = table_builder.get_lines()
        if buf is None:  # pragma: no cover
            buf = sys.stdout
        fmt.buffer_put_lines(buf, lines)

    @abstractmethod
    def _create_table_builder(self) -> TableBuilderAbstract:
        """Create instance of table builder."""


class DataFrameInfoPrinter(InfoPrinterAbstract):
    """
    Class for printing dataframe info.

    Parameters
    ----------
    info : DataFrameInfo
        Instance of DataFrameInfo.
    max_cols : int, optional
        When to switch from the verbose to the truncated output.
    verbose : bool, optional
        Whether to print the full summary.
    show_counts : bool, optional
        Whether to show the non-null counts.
    """

    data: pl.DataFrame

    def __init__(
        self,
        info: DataFrameInfo,
        max_cols: int | None = None,
        verbose: bool | None = None,
        show_counts: bool | None = None,
    ):
        self.info = info
        self.data = info.data
        self.verbose = verbose
        self.max_cols = self._initialize_max_cols(max_cols)
        self.show_counts = self._initialize_show_counts(show_counts)

    @property
    def max_rows(self) -> int:
        """Maximum info rows to be displayed."""
        return int(os.environ.get("POLARS_FMT_MAX_ROWS") or (len(self.data) + 1))

    @property
    def exceeds_info_cols(self) -> bool:
        """Check if number of columns to be summarized does not exceed maximum."""
        return bool(self.col_count > self.max_cols)

    @property
    def exceeds_info_rows(self) -> bool:
        """Check if number of rows to be summarized does not exceed maximum."""
        return bool(len(self.data) > self.max_rows)

    @property
    def col_count(self) -> int:
        """Number of columns to be summarized."""
        return self.info.col_count

    def _initialize_max_cols(self, max_cols: int | None) -> int:
        if max_cols is None:
            return int(os.environ.get("POLARS_FMT_MAX_COLS") or (self.col_count + 1))
        return max_cols

    def _initialize_show_counts(self, show_counts: bool | None) -> bool:
        if show_counts is None:
            return bool(not self.exceeds_info_cols and not self.exceeds_info_rows)
        else:
            return show_counts

    def _create_table_builder(self) -> DataFrameTableBuilder:
        """
        Create instance of table builder based on verbosity and display settings.
        """
        if self.verbose:
            return DataFrameTableBuilderVerbose(
                info=self.info,
                with_counts=self.show_counts,
            )
        elif self.verbose is False:  # specifically set to False, not necessarily None
            return DataFrameTableBuilderNonVerbose(info=self.info)
        else:
            if self.exceeds_info_cols:
                return DataFrameTableBuilderNonVerbose(info=self.info)
            else:
                return DataFrameTableBuilderVerbose(
                    info=self.info,
                    with_counts=self.show_counts,
                )


class TableBuilderAbstract(ABC):
    """
    Abstract builder for info table.
    """

    _lines: List[str]
    info: BaseInfo

    @abstractmethod
    def get_lines(self) -> List[str]:
        """Product in a form of list of lines (strings)."""

    @property
    def data(self) -> FrameOrSeriesUnion:
        return self.info.data

    @property
    def dtypes(self) -> Iterable[Dtype]:
        """Dtypes of each of the DataFrame's columns."""
        return self.info.dtypes

    @property
    def dtype_counts(self) -> Mapping[str, int]:
        """Mapping dtype - number of counts."""
        return self.info.dtype_counts

    @property
    def non_null_counts(self) -> Sequence[int]:
        return self.info.non_null_counts

    def add_object_type_line(self) -> None:
        """Add line with string representation of dataframe to the table."""
        self._lines.append(str(type(self.data)))

    def add_dtypes_line(self) -> None:
        """Add summary line with dtypes present in dataframe."""
        collected_dtypes = [f"{key}({val:d})" for key, val in sorted(self.dtype_counts.items())]
        self._lines.append(f"dtypes: {', '.join(collected_dtypes)}")


class DataFrameTableBuilder(TableBuilderAbstract):
    """
    Abstract builder for dataframe info table.

    Parameters
    ----------
    info : DataFrameInfo.
        Instance of DataFrameInfo.
    """

    def __init__(self, *, info: DataFrameInfo):
        self.info: DataFrameInfo = info

    def get_lines(self) -> List[str]:
        self._lines = []
        if self.col_count == 0:
            self._fill_empty_info()
        else:
            self._fill_non_empty_info()
        return self._lines

    def _fill_empty_info(self) -> None:
        """Add lines to the info table, pertaining to empty dataframe."""
        self.add_object_type_line()
        self._lines.append(f"Empty {type(self.data).__name__}")

    @abstractmethod
    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty dataframe."""

    @property
    def data(self) -> pl.DataFrame:
        """DataFrame."""
        return self.info.data

    @property
    def ids(self) -> Index:
        """Dataframe columns."""
        return self.info.ids

    @property
    def row_count(self) -> int:
        """Number of dataframe rows."""
        return self.info.row_count

    @property
    def col_count(self) -> int:
        """Number of dataframe columns to be summarized."""
        return self.info.col_count


class DataFrameTableBuilderNonVerbose(DataFrameTableBuilder):
    """
    Dataframe info table builder for non-verbose output.
    """

    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty dataframe."""
        self.add_object_type_line()
        # noinspection PyUnresolvedReferences
        self.add_num_rows_line()
        self.add_columns_summary_line()
        self.add_dtypes_line()

    def add_columns_summary_line(self) -> None:
        self._lines.append(self.ids._summary(name="Columns"))


class TableBuilderVerboseMixin(TableBuilderAbstract):
    """
    Mixin for verbose info output.
    """

    SPACING: str = " " * 2
    strrows: Sequence[Sequence[str]]
    gross_column_widths: Sequence[int]
    with_counts: bool

    @property
    @abstractmethod
    def headers(self) -> Sequence[str]:
        """Headers names of the columns in verbose table."""

    @property
    def header_column_widths(self) -> Sequence[int]:
        """Widths of header columns (only titles)."""
        return [len(col) for col in self.headers]

    def _get_gross_column_widths(self) -> Sequence[int]:
        """Get widths of columns containing both headers and actual content."""
        body_column_widths = self._get_body_column_widths()
        return [max(*widths) for widths in zip(self.header_column_widths, body_column_widths)]

    def _get_body_column_widths(self) -> Sequence[int]:
        """Get widths of table content columns."""
        strcols: Sequence[Sequence[str]] = list(zip(*self.strrows))
        return [max(len(x) for x in col) for col in strcols]

    def _gen_rows(self) -> Iterator[Sequence[str]]:
        """
        Generator function yielding rows content.

        Each element represents a row comprising a sequence of strings.
        """
        if self.with_counts:
            return self._gen_rows_with_counts()
        else:
            return self._gen_rows_without_counts()

    @abstractmethod
    def _gen_rows_with_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data with counts."""

    @abstractmethod
    def _gen_rows_without_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data without counts."""

    def add_header_line(self) -> None:
        header_line = self.SPACING.join(
            [
                _put_str(header, col_width)
                for header, col_width in zip(self.headers, self.gross_column_widths)
            ]
        )
        self._lines.append(header_line)

    def add_separator_line(self) -> None:
        separator_line = self.SPACING.join(
            [
                _put_str("-" * header_colwidth, gross_colwidth)
                for header_colwidth, gross_colwidth in zip(
                    self.header_column_widths, self.gross_column_widths
                )
            ]
        )
        self._lines.append(separator_line)

    def add_body_lines(self) -> None:
        for row in self.strrows:
            body_line = self.SPACING.join(
                [
                    _put_str(col, gross_colwidth)
                    for col, gross_colwidth in zip(row, self.gross_column_widths)
                ]
            )
            self._lines.append(body_line)

    def _gen_non_null_counts(self) -> Iterator[str]:
        """Iterator with string representation of non-null counts."""
        for count in self.non_null_counts:
            yield f"{count} non-null"

    def _gen_dtypes(self) -> Iterator[str]:
        """Iterator with string representation of column dtypes."""
        for dtype in self.dtypes:
            yield pprint_thing(dtype)


class DataFrameTableBuilderVerbose(DataFrameTableBuilder, TableBuilderVerboseMixin):
    """
    Dataframe info table builder for verbose output.
    """

    # noinspection PyMissingConstructor
    def __init__(
        self,
        *,
        info: DataFrameInfo,
        with_counts: bool,
    ):
        self.info = info
        self.with_counts = with_counts
        self.strrows: Sequence[Sequence[str]] = list(self._gen_rows())
        self.gross_column_widths: Sequence[int] = self._get_gross_column_widths()

    def _fill_non_empty_info(self) -> None:
        """Add lines to the info table, pertaining to non-empty dataframe."""
        self.add_object_type_line()
        self.add_num_rows_line()
        self.add_columns_summary_line()
        self.add_header_line()
        self.add_separator_line()
        self.add_body_lines()
        self.add_dtypes_line()

    @property
    def headers(self) -> Sequence[str]:
        """Headers names of the columns in verbose table."""
        if self.with_counts:
            return [" # ", "Column", "Non-Null Count", "Dtype"]
        return [" # ", "Column", "Dtype"]

    def add_columns_summary_line(self) -> None:
        self._lines.append(f"Data columns (total {self.col_count} columns):")

    def add_num_rows_line(self) -> None:
        self._lines.append(f"{self.row_count} rows")

    def _gen_rows_without_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data without counts."""
        yield from zip(
            self._gen_line_numbers(),
            self._gen_columns(),
            self._gen_dtypes(),
        )

    def _gen_rows_with_counts(self) -> Iterator[Sequence[str]]:
        """Iterator with string representation of body data with counts."""
        yield from zip(
            self._gen_line_numbers(),
            self._gen_columns(),
            self._gen_non_null_counts(),
            self._gen_dtypes(),
        )

    def _gen_line_numbers(self) -> Iterator[str]:
        """Iterator with string representation of column numbers."""
        for i, _ in enumerate(self.ids):
            yield f" {i}"

    def _gen_columns(self) -> Iterator[str]:
        """Iterator with string representation of column names."""
        for col in self.ids:
            yield pprint_thing(col)


def _get_dataframe_dtype_counts(df: pl.DataFrame) -> Mapping[str, int]:
    """
    Create mapping between datatypes and their number of occurrences.
    """
    # groupby dtype.name to collect e.g. Categorical columns
    from minimal_pandas_api_for_polars.lib.polars_utils import plu

    return S(df.pipe(plu).get_dtypes_(as_dict=True)).value_counts()
