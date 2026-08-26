from rag_sec.dataset import load_t2_ragbench

if __name__ == "__main__":
    print("Loading T2-RAGBench dataset...")
    df = load_t2_ragbench("all")
    print(f"Total dataset loaded successfully: {len(df)} records across subsets.")
    print("Breakdown by subset source:")
    print(df["subset_source"].value_counts())