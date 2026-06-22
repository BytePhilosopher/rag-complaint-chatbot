# RAG-Powered Complaint Analysis Chatbot (CFPB)

A Retrieval-Augmented Generation (RAG) system over U.S. Consumer Financial
Protection Bureau (CFPB) complaint narratives. It lets a non-technical user ask
natural-language questions and get evidence-grounded answers drawn from real
consumer complaints across four financial products: **Credit Card, Personal
Loan, Savings Account, Money Transfer**.

## Project layout

```
rag-complaint-chatbot/
├── data/
│   └── filtered_complaints.csv     # cleaned, filtered output of Task 1
├── reports/
│   ├── eda_stats.json              # machine-readable EDA summary
│   └── figures/                    # generated EDA plots
├── vector_store/                   # persisted FAISS/ChromaDB index (Task 3)
├── notebooks/
├── src/
│   └── eda_preprocessing.py        # Task 1: chunked EDA + preprocessing
├── tests/
├── app.py                          # Gradio UI (Task 4)
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Task 1 — EDA & preprocessing

The raw `complaints.csv` (~5.6 GB, ~9.6M rows) is larger than typical RAM, so
the pipeline streams it in 200k-row chunks in a single pass. It:

1. Computes product distribution, narrative presence, and word-count stats.
2. Filters to the four target products (keyword-matched on the raw `Product`
   label) and drops rows with empty narratives.
3. Cleans narratives: lowercase, strip CFPB `XXXX` redactions and boilerplate
   openings, remove special characters, collapse whitespace.
4. Writes `data/filtered_complaints.csv` and EDA artifacts to `reports/`.

```bash
python src/eda_preprocessing.py
```

See [reports/EDA_summary.md](reports/EDA_summary.md) for the written findings.
