import polars as pl

from llmpipe import (
    read_data,  # Reads csv, tab separated (with header, .txt) or json lines (.jsonl) data from disk
    write_data  # Writes data as csv, tab separated (with header, .txt) or json lines (.jsonl) to disk
)

data_path = ...
samples = read_data(data_path)  # Infers file type

df = pl.from_dicts(samples)
...

write_data(to.to_dicts(), "path/to/output.jsonl")  # or csv or txt
