"""Arm 2: build the BM25 index (pg_search) over the existing `chunks.text`
column. No re-embedding needed -- this indexes text already loaded by
embed_local.py.
"""

from dotenv import load_dotenv

load_dotenv()

from rag_sec.store import BM25_INDEX_SQL, get_conn, init_schema


def main() -> None:
    init_schema()
    with get_conn(check=False) as conn:  # build-time: this script moves the counts
        print("Building BM25 index (pg_search)...")
        conn.execute(BM25_INDEX_SQL)
        conn.commit()
        n = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
    print(f"Done. BM25 index covers {n} chunks.")


if __name__ == "__main__":
    main()
