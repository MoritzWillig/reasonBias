# Causal Reasoning Bias in Humans and LMs


## Query data

1) Place your API keys in the respective `aleph_alpha` and `openai` files under the `keys/` directory.
2) Create datasets using `generate_causal_queries.py` and `generate_causal_queries_grid.py`.
  * Note: generating causal queries resets the `id_counters`. Make sure to run `rebuild_query_counts` before querying to keep prevent overwriting existing results.
4) Query APIs using `query_samples.py`. Comment in or out APIs in `active_apis` and select datasetss in `datasets`.
5) Use `A0_*` and `A1_*` files to summarize results in json, csv format and create plots.

Query results are stored to `queries/`. Plots and CSVs are saved to `evaluation/`.