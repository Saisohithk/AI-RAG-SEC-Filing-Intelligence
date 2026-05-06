"""
ingest.py — Command-line tool to download and index SEC 10-K filings.

Usage:
  # Download filings from SEC EDGAR
  python scripts/ingest.py download --tickers AAPL MSFT GOOGL --num-filings 3

  # Embed and store in ChromaDB (local, no API key needed)
  python scripts/ingest.py embed --use-local

  # Embed and store in Pinecone (cloud)
  python scripts/ingest.py embed --use-pinecone

  # Full pipeline: download + embed in one command
  python scripts/ingest.py full --tickers AAPL MSFT --use-pinecone

  # Show vector store statistics
  python scripts/ingest.py stats

What this script does:
  1. download: Fetches 10-K PDFs/HTML from SEC EDGAR via sec-edgar-downloader
  2. embed: Reads saved filings, chunks them, embeds with Jina, stores in DB
  3. full: Both download and embed in sequence
  4. stats: Shows how many vectors are stored and which tickers are indexed
"""

# Ensure project root is on path when running as: python scripts/ingest.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
from tqdm import tqdm

# Set up logging to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),                         # Console output
        logging.FileHandler("ingestion.log", encoding="utf-8"),    # Log file (UTF-8)
    ],
)
# Force UTF-8 on Windows console so Unicode characters don't crash
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
logger = logging.getLogger(__name__)

# Default tickers for demos — the 5 most analyzed US companies
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]


def cmd_download(args) -> None:
    """
    Download 10-K filings from SEC EDGAR.
    Files are saved to ./data/sec_filings/TICKER/10-K/
    """
    from src.ingestion.downloader import download_10k_filings

    tickers = args.tickers or DEFAULT_TICKERS
    logger.info(f"Downloading {args.num_filings} filings each for: {tickers}")

    downloaded = download_10k_filings(
        tickers=tickers,
        num_filings=args.num_filings,
        output_dir="./data/sec_filings",
    )

    print(f"\n[OK] Downloaded filings for {len(downloaded)} tickers")
    for path in downloaded:
        files = list(path.rglob("*.htm")) + list(path.rglob("*.pdf"))
        print(f"  {path.parent.name}: {len(files)} files")


def cmd_embed(args) -> None:
    """
    Read downloaded filings, chunk them, embed with Jina, and store in vector DB.
    Assumes files are already downloaded (run 'download' first).
    """
    from src.ingestion.chunker import SECChunker
    from src.ingestion.downloader import extract_text_from_filing
    from src.ingestion.embedder import VectorStoreManager
    from src.core.config import settings

    use_local = getattr(args, "use_local", False) or not getattr(args, "use_pinecone", False)

    logger.info(f"Using {'ChromaDB (local)' if use_local else 'Pinecone (cloud)'}")

    vector_store = VectorStoreManager(use_local=use_local)
    chunker = SECChunker(settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)

    # Find all downloaded filing files
    filings_dir = Path("./data/sec_filings")
    if not filings_dir.exists():
        print("[FAIL] No downloaded filings found. Run 'python scripts/ingest.py download' first.")
        sys.exit(1)

    # sec-edgar-downloader 5.x saves as full-submission.txt (SGML format)
    # Also support legacy .htm and .pdf files
    filing_files = (
        list(filings_dir.rglob("full-submission.txt"))
        + list(filings_dir.rglob("*.htm"))
        + list(filings_dir.rglob("*.pdf"))
    )
    logger.info(f"Found {len(filing_files)} filing files to process")

    all_chunks = []

    # Process each file with a progress bar
    for filing_path in tqdm(filing_files, desc="Processing filings"):
        try:
            text, metadata = extract_text_from_filing(filing_path)
            chunks = chunker.chunk_document(text, metadata)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Error processing {filing_path.name}: {e}")
            continue

    if not all_chunks:
        print("[FAIL] No chunks created. Check the filing files.")
        sys.exit(1)

    logger.info(f"Total chunks ready to embed: {len(all_chunks)}")

    # Upsert to vector store with progress feedback
    print(f"\nEmbedding {len(all_chunks)} chunks (this may take a few minutes)...")
    stored = vector_store.upsert_chunks(all_chunks)

    print(f"\n[OK] Stored {stored} vectors in {'ChromaDB' if use_local else 'Pinecone'}")


def cmd_full(args) -> None:
    """Run full pipeline: download + embed in one step."""
    logger.info("Running full ingestion pipeline...")

    # Step 1: Download
    cmd_download(args)

    # Step 2: Embed (use whatever store was specified)
    cmd_embed(args)

    print("\n[OK] Full pipeline complete!")


def cmd_stats(args) -> None:
    """Show current vector store statistics."""
    from src.ingestion.embedder import VectorStoreManager
    from src.core.config import settings

    use_local = not getattr(args, "use_pinecone", False)
    vector_store = VectorStoreManager(use_local=use_local)
    stats = vector_store.get_collection_stats()

    print("\n--- Vector Store Statistics ---")
    print(f"  Store type:   {stats['store_type']}")
    print(f"  Index name:   {stats['index_name']}")
    print(f"  Vector count: {stats['vector_count']:,}")
    print(f"  Dimensions:   {stats['dimension']}")
    print("-------------------------------\n")


def main() -> None:
    """Parse arguments and dispatch to the right command function."""
    parser = argparse.ArgumentParser(
        description="SEC RAG — Ingestion Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/ingest.py download --tickers AAPL MSFT
  python scripts/ingest.py embed --use-local
  python scripts/ingest.py full --tickers AAPL --use-pinecone
  python scripts/ingest.py stats
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- download subcommand ---
    dl = subparsers.add_parser("download", help="Download 10-K filings from SEC EDGAR")
    dl.add_argument("--tickers", nargs="+", help="Stock tickers (default: AAPL MSFT GOOGL AMZN NVDA)")
    dl.add_argument("--num-filings", type=int, default=3, help="Filings per ticker (default: 3)")

    # --- embed subcommand ---
    em = subparsers.add_parser("embed", help="Embed and store downloaded filings")
    em_group = em.add_mutually_exclusive_group()
    em_group.add_argument("--use-local", action="store_true", help="Use ChromaDB (local, no API key)")
    em_group.add_argument("--use-pinecone", action="store_true", help="Use Pinecone (cloud)")

    # --- full subcommand ---
    fl = subparsers.add_parser("full", help="Download + embed in one step")
    fl.add_argument("--tickers", nargs="+", help="Stock tickers to ingest")
    fl.add_argument("--num-filings", type=int, default=3, help="Filings per ticker")
    fl_group = fl.add_mutually_exclusive_group()
    fl_group.add_argument("--use-local", action="store_true", help="Use ChromaDB (local)")
    fl_group.add_argument("--use-pinecone", action="store_true", help="Use Pinecone (cloud)")

    # --- stats subcommand ---
    st = subparsers.add_parser("stats", help="Show vector store statistics")
    st.add_argument("--use-pinecone", action="store_true", help="Show Pinecone stats (default: ChromaDB)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Dispatch to the right function
    commands = {
        "download": cmd_download,
        "embed": cmd_embed,
        "full": cmd_full,
        "stats": cmd_stats,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
