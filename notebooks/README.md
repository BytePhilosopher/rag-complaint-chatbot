# Notebooks

Interactive exploration for the RAG complaint chatbot.

The Task 1 EDA + preprocessing logic lives in [`../src/eda_preprocessing.py`](../src/eda_preprocessing.py)
rather than a notebook, because the raw dataset (~5.6 GB) must be streamed in
chunks — running that interactively in a kernel on a low-RAM machine is
impractical. The script saves all statistics to `reports/eda_stats.json` and
plots to `reports/figures/`, which can be loaded into a notebook for narrative
write-up without re-reading the raw file.
