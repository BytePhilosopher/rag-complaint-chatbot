"""Task 1 — EDA and preprocessing for the CFPB complaint dataset.

The raw file (``complaints.csv``, ~5.6 GB, ~9.6M rows) is far larger than
available RAM, so every pass over it is *chunked*. A single streaming pass
computes the EDA statistics, collects narrative word counts, filters the four
target products, cleans the narratives, and appends the survivors to
``data/filtered_complaints.csv``.

Run:  python src/eda_preprocessing.py
"""
from __future__ import annotations

import re
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless / no display
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "complaints.csv"
OUT_CSV = ROOT / "data" / "filtered_complaints.csv"
FIG_DIR = ROOT / "reports" / "figures"
STATS_JSON = ROOT / "reports" / "eda_stats.json"
FIG_DIR.mkdir(parents=True, exist_ok=True)
STATS_JSON.parent.mkdir(parents=True, exist_ok=True)

CHUNKSIZE = 200_000

# --------------------------------------------------------------------------- #
# Product mapping: raw CFPB Product label -> target category
# Matched by lowercase keyword so historical label variants are captured.
# --------------------------------------------------------------------------- #
PRODUCT_KEYWORDS = [
    ("Credit Card", lambda p: "credit card" in p),                 # incl. "credit card or prepaid card"
    ("Personal Loan", lambda p: "personal loan" in p),             # payday/title/personal loan buckets
    ("Savings Account", lambda p: "savings" in p),                 # "checking or savings account"
    ("Money Transfer", lambda p: "money transfer" in p),           # "money transfer, virtual currency, ..."
]
# Columns kept in the cleaned output (renamed to be RAG-pipeline friendly).
KEEP_COLS = {
    "Complaint ID": "complaint_id",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Company": "company",
    "State": "state",
    "Date received": "date_received",
}

# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #
# Boilerplate / redaction patterns common in CFPB narratives.
BOILERPLATE = [
    r"i am writing to file a complaint( with the consumer financial protection bureau)?",
    r"i am writing to (you )?(in order )?to file a complaint",
    r"i would like to file a complaint",
    r"this is a complaint (against|regarding|about)",
    r"to whom it may concern",
    r"please be advised( that)?",
]
BOILERPLATE_RE = re.compile("|".join(BOILERPLATE))
# CFPB redacts PII with runs of X (e.g. "XXXX", "XX/XX/XXXX").
REDACTION_RE = re.compile(r"\b(?:x{2,}[\s/.-]*)+\b")
# Keep letters, digits and spaces only.
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip redactions/boilerplate/special chars, collapse space."""
    t = text.lower()
    t = REDACTION_RE.sub(" ", t)
    t = BOILERPLATE_RE.sub(" ", t)
    t = NON_ALNUM_RE.sub(" ", t)
    t = WS_RE.sub(" ", t).strip()
    return t


def map_product(raw: str) -> str | None:
    p = raw.lower()
    for category, match in PRODUCT_KEYWORDS:
        if match(p):
            return category
    return None


# --------------------------------------------------------------------------- #
# Main streaming pass
# --------------------------------------------------------------------------- #
def main() -> None:
    product_counts: Counter[str] = Counter()
    narr_present = 0
    narr_absent = 0
    total_rows = 0

    # Per-category word-count arrays (only narratives that survive the filter)
    wordcounts_kept: list[np.ndarray] = []
    # Word counts for ALL narratives (for the global length-distribution EDA)
    wordcounts_all: list[np.ndarray] = []

    kept_per_category: Counter[str] = Counter()
    dropped_empty_after_clean = 0
    rows_written = 0

    if OUT_CSV.exists():
        OUT_CSV.unlink()

    reader = pd.read_csv(
        RAW_CSV,
        usecols=list(KEEP_COLS) + ["Consumer complaint narrative"],
        dtype=str,
        chunksize=CHUNKSIZE,
        low_memory=False,
    )

    for i, chunk in enumerate(reader):
        total_rows += len(chunk)
        product_counts.update(chunk["Product"].fillna("<<NA>>"))

        narr = chunk["Consumer complaint narrative"]
        has_narr = narr.notna() & (narr.str.strip() != "")
        narr_present += int(has_narr.sum())
        narr_absent += int((~has_narr).sum())

        # global narrative word counts
        if has_narr.any():
            wc_all = narr[has_narr].str.split().str.len().to_numpy(dtype=np.int32)
            wordcounts_all.append(wc_all)

        # ---- filter: target product + non-empty narrative ----
        chunk = chunk[has_narr].copy()
        chunk["product_category"] = chunk["Product"].map(map_product)
        chunk = chunk[chunk["product_category"].notna()]
        if chunk.empty:
            continue

        # ---- clean ----
        cleaned = chunk["Consumer complaint narrative"].map(clean_text)
        nonempty = cleaned.str.len() > 0
        dropped_empty_after_clean += int((~nonempty).sum())
        chunk = chunk[nonempty]
        cleaned = cleaned[nonempty]

        wc_kept = cleaned.str.split().str.len().to_numpy(dtype=np.int32)
        wordcounts_kept.append(wc_kept)
        kept_per_category.update(chunk["product_category"])

        out = chunk[list(KEEP_COLS)].rename(columns=KEEP_COLS)
        out["product_category"] = chunk["product_category"].values
        out["narrative"] = cleaned.values

        out.to_csv(OUT_CSV, mode="a", header=(rows_written == 0), index=False)
        rows_written += len(out)

        if (i + 1) % 5 == 0:
            print(f"  ...processed {total_rows:,} rows, kept {rows_written:,}")

    # ----------------------------------------------------------------------- #
    # Aggregate + report
    # ----------------------------------------------------------------------- #
    wc_all = np.concatenate(wordcounts_all) if wordcounts_all else np.array([], np.int32)
    wc_kept = np.concatenate(wordcounts_kept) if wordcounts_kept else np.array([], np.int32)

    def describe(a: np.ndarray) -> dict:
        if a.size == 0:
            return {}
        return {
            "count": int(a.size),
            "min": int(a.min()),
            "max": int(a.max()),
            "mean": round(float(a.mean()), 2),
            "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)),
            "n_under_5_words": int((a < 5).sum()),
            "n_over_500_words": int((a > 500).sum()),
        }

    stats = {
        "total_rows": total_rows,
        "narratives_present": narr_present,
        "narratives_absent": narr_absent,
        "rows_written_filtered": rows_written,
        "dropped_empty_after_clean": dropped_empty_after_clean,
        "kept_per_category": dict(kept_per_category),
        "top_products_raw": dict(product_counts.most_common(15)),
        "word_count_all_narratives": describe(wc_all),
        "word_count_kept_narratives": describe(wc_kept),
    }
    STATS_JSON.write_text(json.dumps(stats, indent=2))

    print("\n" + "=" * 60)
    print(json.dumps(stats, indent=2))
    print("=" * 60)
    print(f"\nFiltered dataset -> {OUT_CSV}  ({rows_written:,} rows)")

    # ----------------------------------------------------------------------- #
    # Plots
    # ----------------------------------------------------------------------- #
    _plot_product_distribution(product_counts)
    _plot_narrative_presence(narr_present, narr_absent)
    if wc_all.size:
        _plot_wordcount_hist(wc_all, "all narratives", "wordcount_all.png")
    if wc_kept.size:
        _plot_wordcount_hist(wc_kept, "filtered (4 products, cleaned)", "wordcount_kept.png")
    if kept_per_category:
        _plot_kept_categories(kept_per_category)
    print(f"Figures -> {FIG_DIR}")


def _plot_product_distribution(counts: Counter) -> None:
    top = counts.most_common(15)
    labels = [k if len(k) < 40 else k[:37] + "..." for k, _ in top][::-1]
    vals = [v for _, v in top][::-1]
    plt.figure(figsize=(10, 7))
    plt.barh(labels, vals, color="#3b6ea5")
    plt.title("Top 15 raw Product categories (full dataset)")
    plt.xlabel("Number of complaints")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "product_distribution.png", dpi=120)
    plt.close()


def _plot_narrative_presence(present: int, absent: int) -> None:
    plt.figure(figsize=(6, 6))
    plt.pie([present, absent], labels=["with narrative", "without narrative"],
            autopct="%1.1f%%", colors=["#3b6ea5", "#c44e52"], startangle=90)
    plt.title("Complaints with vs. without a consumer narrative")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "narrative_presence.png", dpi=120)
    plt.close()


def _plot_wordcount_hist(arr: np.ndarray, label: str, fname: str) -> None:
    clip = np.clip(arr, 0, 600)
    plt.figure(figsize=(9, 5))
    plt.hist(clip, bins=60, color="#55a868", edgecolor="white")
    plt.axvline(np.median(arr), color="#c44e52", linestyle="--",
                label=f"median={np.median(arr):.0f}")
    plt.title(f"Narrative word-count distribution — {label}\n(clipped at 600)")
    plt.xlabel("words per narrative")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=120)
    plt.close()


def _plot_kept_categories(counts: Counter) -> None:
    items = counts.most_common()
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, vals, color="#8172b3")
    plt.title("Filtered dataset: complaints per target category")
    plt.ylabel("Number of complaints (with narrative)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "kept_categories.png", dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
