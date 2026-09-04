"""Loads T2-RAGBench (FinQA/ConvFinQA/TAT-DQA) from Hugging Face, assigning a
train/dev/test split to ConvFinQA -- the only subset that ships without one.
"""

from typing import Literal
from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd

CONVFINQA_SPLIT_SEED = 42
CONVFINQA_SPLIT_RATIOS = {"train": 0.8, "dev": 0.1, "test": 0.1}

SUBSET_FILES = {
    "ConvFinQA": ["data/ConvFinQA/turn_0.jsonl"],
    "FinQA": [
        "data/FinQA/train/metadata.jsonl",
        "data/FinQA/dev/metadata.jsonl",
        "data/FinQA/test/metadata.jsonl",
    ],
    "TAT-DQA": [
        "data/TAT-DQA/train/metadata.jsonl",
        "data/TAT-DQA/dev/metadata.jsonl",
        "data/TAT-DQA/test/metadata.jsonl",
    ],
}

SubsetName = Literal["ConvFinQA", "FinQA", "TAT-DQA", "all"]


def assign_convfinqa_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Assigns train/dev/test to ConvFinQA rows (originally all `split == "all"`).

    Mirrors FinQA's and TAT-DQA's ~80/10/10 row ratio. Splits by `context_id` (the source
    document), not by row, so a document's questions can't leak across splits; fixed seed,
    so re-running reproduces it.
    """
    doc_counts = df.groupby("context_id").size()
    doc_ids = doc_counts.index.to_numpy()
    rng = np.random.default_rng(CONVFINQA_SPLIT_SEED)
    rng.shuffle(doc_ids)

    cumulative_rows = doc_counts.loc[doc_ids].cumsum()
    total_rows = doc_counts.sum()
    train_cutoff = CONVFINQA_SPLIT_RATIOS["train"] * total_rows
    dev_cutoff = train_cutoff + CONVFINQA_SPLIT_RATIOS["dev"] * total_rows

    doc_to_split = {}
    for doc_id, rows_so_far in zip(doc_ids, cumulative_rows):
        if rows_so_far <= train_cutoff:
            doc_to_split[doc_id] = "train"
        elif rows_so_far <= dev_cutoff:
            doc_to_split[doc_id] = "dev"
        else:
            doc_to_split[doc_id] = "test"

    df = df.copy()
    df["split"] = df["context_id"].map(doc_to_split)
    return df


def load_t2_ragbench(subset: SubsetName = "all") -> pd.DataFrame:
    """Loads specified subset(s) of T2-RAGBench from Hugging Face into a pandas DataFrame."""
    if subset == "all":
        subsets_to_load = list(SUBSET_FILES.keys())
    elif subset in SUBSET_FILES:
        subsets_to_load = [subset]
    else:
        raise ValueError(f"Unknown subset: {subset}. Must be one of {list(SUBSET_FILES.keys())} or 'all'.")

    dfs = []
    for sub in subsets_to_load:
        for file_path in SUBSET_FILES[sub]:
            local_path = hf_hub_download(repo_id="G4KMU/t2-ragbench", filename=file_path, repo_type="dataset")
            df = pd.read_json(local_path, lines=True)
            df["subset_source"] = sub
            if sub == "ConvFinQA":
                df = assign_convfinqa_splits(df)
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)
