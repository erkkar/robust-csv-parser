# Robust CSV parser

The robust CSV parser is intended for parsing malformed CSV files. 
Potential use cases:
* Multiple headers, some in the middle of the file
* Headers change in the middle of the file

## Examples

Parse a single file where a header row begins with ‘Period start’ or ‘Read time’.
The first column is treated as ISO8601 formatted time stamps, other data are 
numeric floats. Select only columns which start with ‘avg’. 
The default time zone is UTC if other cannot be inferred.

```Python
>>> from robust_csv_parser import RobustCSVParser
>>> parser = RobustCSVParser(
        sep=",",
        encoding="utf8",
        header_regex=r"Period start|Read time",
        column_regex=r"^avg",
        index_col=0,
        parse_dates=True,
        date_format="ISO8601",
        dtype=float,
        default_tz="UTC",
    )
>>> df = parser.parse('/path/to/file.csv')
```

File can also be a gzip archive:

```Python
>>> df = parser.parse('/path/to/archive.gz')
```

Parsing multiple files:

```Python
>>> from pathlib import Path
>>> df = parser.parse_multifile(Path('/path/to/data').glob('*.csv'))
```






