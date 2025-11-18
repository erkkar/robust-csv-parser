"""Functions for parsing PyFlux output files"""

import functools
import gzip
import logging
import re
import warnings
from collections.abc import Iterable
from io import StringIO, TextIOBase
from pathlib import Path
from typing import Callable

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


class _FilepathOrBuffer:
    def __init__(self, filepath_or_buffer, encoding):
        if not isinstance(filepath_or_buffer, TextIOBase):
            if str(filepath_or_buffer).endswith("gz"):  # gzipped file
                open_func = gzip.open
            else:  # regular text file
                open_func = open
            self._fp = open_func(filepath_or_buffer, mode="rt", encoding=encoding)
        else:
            self._fp = filepath_or_buffer

    def __enter__(self):
        return self._fp

    def __exit__(self, *args):
        self._fp.close()


class RobustCSVParser:
    def __init__(
        self,
        sep=",",
        encoding="utf8",
        header_string: str | None = None,
        header_regex: str | None = None,
        column_regex: str | None = None,
        process_func: Callable | None = None,
        default_tz: str = "UTC",
        **csv_kwargs,
    ):
        """Robust data parser

        Args:
            sep (optional): CSV column separator. Defaults to ",".
            encoding (optional): File enconding. Defaults to "utf8".
            header_string (optional): String to detect a header row. Defaults to None.
            header_regex (optional): Regular expression to detect a header row. Defaults to None.
            column_regex (optional): Regular expression to filter columns with. Defaults to None.
            process_func (optional): Function to call for each DataFrame. Defaults to None.
            csv_kwargs: Other arguments are passed to `pandas.read_csv`.
        """
        self.sep = sep
        self.encoding = encoding
        if header_regex is not None and header_string is not None:
            warnings.warn(
                "Both `header_string` and `header_regex` given, "
                "only using the regular expression."
            )
        self.header_regex = header_regex
        self.header_string = header_string
        self.csv_kwargs = csv_kwargs
        self.column_regex = column_regex or r"."  # by default match anything
        self.process_func = process_func
        self.default_tz = default_tz

    def parse(
        self, filepath_or_buffer, logger=logger, log_level="WARNING"
    ) -> pd.DataFrame | None:
        """Parse a file

        Args:
            filepath_or_buffer: Path to a file or a file-like object
            logger: Logger to use, defaults to the module-level logger.
            log_level: Logging level, fefaults to 'WARNING'.
        """
        if logger is None:
            logger = logging.getLogger("RobustDataParser.parse")
        if not logger.hasHandlers():
            logger.setLevel(log_level)
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
            logger.addHandler(sh)

        logger.info("Reading file %s", filepath_or_buffer)
        with _FilepathOrBuffer(filepath_or_buffer, self.encoding) as fp:
            # Try to guess the header string from first field of first row if not given
            if self.header_string is None and self.header_regex is None:
                self.header_string = fp.readline().split(self.sep)[0]
                fp.seek(0)
            data = fp.read().replace("\0", "")  # Remove possible NUL values

        frames = []
        # Create an iterator to find all header rows
        regex = re.compile(
            # Allow a quotation mark (" or ') at the start of the header
            rf"['\"]?({self.header_regex or re.escape(self.header_string)}).*$",
            flags=re.MULTILINE,
        )
        headerrow_finder = re.finditer(regex, data)
        # Get start index of first header
        try:
            start_prev = next(headerrow_finder).start()
        except StopIteration:
            logger.error("No header found in %s", filepath_or_buffer)
            return None
        # Iterate over all found headers and finally from the last header to end (None)
        for start in [match.start() for match in headerrow_finder] + [None]:
            frames.append(
                self._parse_frame(
                    data[slice(start_prev, start)],  # parse data between indices
                    str(filepath_or_buffer),
                    logger,
                )
            )
            start_prev = start

        # Join data
        try:
            df = pd.concat(frames, join="outer")
        except ValueError:
            logger.error("All empty data in %s", filepath_or_buffer)
            return None
        df.attrs["source"] = str(filepath_or_buffer)
        if self.process_func is not None:
            try:
                return self.process_func(df)
            except Exception as err:
                logger.error(
                    "Unable to process file %s: %s", filepath_or_buffer, str(err)
                )
                return None
        return df

    def parse_multifile(
        self,
        filepaths: Iterable[Path | str],
        n_jobs: int = 1,
    ) -> pd.DataFrame:
        """Parse multiple files using multiple processes

        Args:
            filepaths: An iterable of file paths
            n_jobs: Number of parallel jobs to spawn
        """

        root_logger = logging.getLogger()
        if n_jobs > 1:
            logger.info("Starting read using %d workers", n_jobs)
            frames = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(self.parse)(
                    filepath, logger=None, log_level=root_logger.getEffectiveLevel()
                )
                for filepath in filepaths
            )
            logger.info("Done")
        else:
            frames = map(self.parse, filepaths)
        try:
            return pd.concat(frames, axis=0, join="outer")
        except ValueError:
            logger.error("All empty frames")
            return None

    def _parse_frame(self, data: str, source: str, logger: logging.Logger):
        csv_kwargs = self.csv_kwargs | dict(
            sep=self.sep,
            header=0,
            engine="c",
            on_bad_lines="warn",
        )
        if "parse_dates" in csv_kwargs:
            csv_kwargs["parse_dates"] = False
            csv_kwargs["dtype"] = None

        try:
            with warnings.catch_warnings(
                category=pd.errors.ParserWarning, action="ignore", record=True
            ) as w:
                df = (
                    pd.read_csv(StringIO(data), **csv_kwargs)
                    .dropna(axis=1, how="all")  # drop empty columns
                    .filter(regex=self.column_regex)  # filter columns
                )
                for message in w:
                    logger.warning(message)

        except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as err:
            logger.error("Failed reading file %s: %s", source, err)
            return None

        # Parse timestamps
        if self.csv_kwargs.get("parse_dates", None):
            # Try to guess the time zone
            index_name = df.index.name
            m = re.search(r"UTC(\+\d+)", index_name)
            if m:
                tz = f"Etc/GMT{-int(m.group(1))}"
            else:
                tz = self.default_tz
                logger.warning(
                    "Unable to detect time zone in %s, assuming %s",
                    source,
                    tz,
                )
            df.index = pd.DatetimeIndex(
                pd.to_datetime(
                    df.index,
                    format=self.csv_kwargs.get("date_format", None),
                    errors="coerce",
                ),
                tz=tz,
                name=df.index.name,
            ).tz_convert(self.default_tz)
            if "dtype" in self.csv_kwargs:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(message="overflow", action="ignore")
                        df = df.astype(self.csv_kwargs["dtype"])
                except (ValueError, RuntimeWarning) as err:
                    logger.error("Failed converting data in %s: %s", source, str(err))
                    return None

        return df
