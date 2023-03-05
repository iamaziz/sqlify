# PySQLify

[![Open in Streamlit][share_badge]][share_link]

Upload a dataset to process (manipulate/analyze) it using SQL and Python, similar to running Jupyter Notebooks. Currently, supports `csv` files only.


[share_badge]: https://static.streamlit.io/badges/streamlit_badge_black_white.svg
[share_link]: https://sqlify.streamlit.app

<hr>

**TODO**
```
- [x] generate data profile with pandas_profiling
- [ ] support upload other file types: tsv, .xls, .xlsx
- [ ] caching / for scaling, use DuckDB
- [ ] parse data types, and change col types
- [ ] configs eg skip_rows, sep, column names .. etc
```

Credit. This is created with:

> [Streamlit](https://github.com/streamlit/streamlit) | [pandas](https://github.com/pandas-dev/pandas) | [duckdb](https://github.com/duckdb/duckdb) | [streamlit-ace](https://github.com/okld/streamlit-ace)