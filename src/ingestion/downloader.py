"""
downloader.py — Downloads 10-K filings from SEC EDGAR and extracts their text.

What this file does:
  1. Uses sec-edgar-downloader to fetch annual 10-K filings for any stock ticker
  2. sec-edgar-downloader 5.x saves files as full-submission.txt (EDGAR SGML format)
     These contain the HTML/text filing embedded inside SGML wrapper tags
  3. Extracts the actual 10-K HTML from inside the SGML envelope
  4. Attaches metadata (ticker, date, form type) to each extracted document

EDGAR file format:
  full-submission.txt is an SGML container:
    <SEC-DOCUMENT>
      <DOCUMENT>
        <TYPE>10-K
        <TEXT>
          <html>...actual filing content here...</html>
        </TEXT>
      </DOCUMENT>
    </SEC-DOCUMENT>
  We extract just the <TEXT> section from the 10-K document.
"""

import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader

logger = logging.getLogger(__name__)


def download_10k_filings(
    tickers: list[str],
    num_filings: int = 3,
    output_dir: str = "./data/sec_filings",
) -> list[Path]:
    """
    Download 10-K annual filings from SEC EDGAR for a list of stock tickers.

    Files are saved to: output_dir/sec-edgar-filings/TICKER/10-K/ACCESSION/full-submission.txt

    Args:
        tickers: Stock ticker symbols, e.g. ["AAPL", "MSFT"]
        num_filings: How many years of filings to download per ticker (default: 3)
        output_dir: Root directory to save downloaded filings

    Returns:
        List of paths to ticker-level 10-K directories that were successfully downloaded

    Example:
        >>> paths = download_10k_filings(["AAPL"], num_filings=2)
        >>> print(paths[0])
        ./data/sec_filings/sec-edgar-filings/AAPL/10-K
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # sec-edgar-downloader 5.x: parameter is download_folder (not save_path)
    downloader = Downloader(
        company_name="SEC RAG Research",
        email_address="research@secrag.local",
        download_folder=str(output_path),
    )

    downloaded_paths: list[Path] = []

    for ticker in tickers:
        try:
            logger.info(f"Downloading {num_filings} 10-K filings for {ticker}...")
            downloader.get("10-K", ticker, limit=num_filings)

            # sec-edgar-downloader 5.x saves to: download_folder/sec-edgar-filings/TICKER/10-K/
            ticker_dir = output_path / "sec-edgar-filings" / ticker / "10-K"

            if ticker_dir.exists():
                filing_count = sum(1 for d in ticker_dir.iterdir() if d.is_dir())
                downloaded_paths.append(ticker_dir)
                logger.info(f"  OK {ticker}: {filing_count} filings at {ticker_dir}")
            else:
                logger.warning(f"  FAIL {ticker}: directory not found at {ticker_dir}")

        except Exception as e:
            logger.error(f"  FAIL {ticker}: download error — {e}")
            continue

    logger.info(
        f"Download complete: {len(downloaded_paths)}/{len(tickers)} tickers succeeded"
    )
    return downloaded_paths


def extract_text_from_filing(filing_path: Path) -> tuple[str, dict]:
    """
    Extract readable text from a SEC filing file.

    Handles two formats:
    - full-submission.txt: EDGAR SGML envelope (from sec-edgar-downloader 5.x)
      Contains embedded HTML that we extract from the <TEXT> block
    - .htm / .html: Plain HTML filing
    - .pdf: PDF filing

    Args:
        filing_path: Path to the filing file

    Returns:
        Tuple of (extracted_text, metadata_dict)
        metadata contains: ticker, filing_date, form_type, source_path

    Example:
        >>> text, meta = extract_text_from_filing(Path(".../full-submission.txt"))
        >>> print(meta["ticker"])
        AAPL
    """
    suffix = filing_path.suffix.lower()
    name = filing_path.name.lower()

    if name == "full-submission.txt":
        # EDGAR SGML format — extract the HTML/text inside the <TEXT> block
        text = _extract_from_sgml(filing_path)

    elif suffix in (".htm", ".html"):
        raw_html = filing_path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup(["script", "style", "header", "footer"]):
            tag.decompose()
        text = soup.get_text(separator="\n")

    elif suffix == ".pdf":
        import fitz  # PyMuPDF

        doc = fitz.open(str(filing_path))
        text = "\n".join(page.get_text() for page in doc)
        doc.close()

    else:
        raise ValueError(f"Unsupported file format: {suffix} ({filing_path.name})")

    text = _clean_text(text)
    metadata = _extract_metadata_from_path(filing_path)

    logger.debug(f"Extracted {len(text):,} chars from {filing_path.name}")
    return text, metadata


def _extract_from_sgml(filing_path: Path) -> str:
    """
    Extract the 10-K document text from inside an EDGAR full-submission.txt file.

    The SGML format looks like:
      <DOCUMENT>
      <TYPE>10-K
      ...
      <TEXT>
      <html>...filing content...</html>
      </TEXT>
      </DOCUMENT>

    We find the first <DOCUMENT> block with TYPE=10-K and extract its <TEXT>.
    """
    # Read file — EDGAR files can be large (5-50MB)
    try:
        raw = filing_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.error(f"Could not read {filing_path}: {e}")
        return ""

    # Find the 10-K document block
    # Pattern: <DOCUMENT>\n<TYPE>10-K\n...<TEXT>\ncontent\n</TEXT>
    doc_pattern = re.compile(
        r"<DOCUMENT>\s*<TYPE>10-K.*?<TEXT>(.*?)</TEXT>",
        re.DOTALL | re.IGNORECASE,
    )
    match = doc_pattern.search(raw)

    if not match:
        # Fallback: just take everything between the first <TEXT> tags
        text_pattern = re.compile(r"<TEXT>(.*?)</TEXT>", re.DOTALL | re.IGNORECASE)
        match = text_pattern.search(raw)

    if not match:
        logger.warning(
            f"No <TEXT> block found in {filing_path.name}, using raw content"
        )
        return raw[:500_000]  # Return first 500K chars as fallback

    text_content = match.group(1).strip()

    # If the content looks like HTML, parse it with BeautifulSoup
    if "<html" in text_content.lower() or "<p" in text_content.lower():
        soup = BeautifulSoup(text_content, "html.parser")
        for tag in soup(["script", "style", "header", "footer"]):
            tag.decompose()
        return soup.get_text(separator="\n")

    return text_content


def _clean_text(text: str) -> str:
    """Remove excess whitespace and blank lines from extracted text."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line for line in text.split("\n") if line.strip()]
    return "\n".join(lines)


def _extract_metadata_from_path(filing_path: Path) -> dict:
    """
    Parse the file path to extract ticker, form type, and filing date.

    sec-edgar-downloader 5.x path structure:
      .../sec-edgar-filings/TICKER/10-K/ACCESSION_NUMBER/full-submission.txt
      Accession format: XXXXXXXXXX-YY-NNNNNN  (YY = 2-digit year)
    """
    parts = filing_path.parts
    ticker = "UNKNOWN"
    form_type = "10-K"
    filing_date = "unknown"

    # Walk the path parts to find the ticker (part before "10-K")
    for i, part in enumerate(parts):
        if part == "10-K" and i > 0:
            ticker = parts[i - 1]
            form_type = part
            break

    # Extract year from accession number: XXXXXXXXXX-YY-NNNNNN
    for part in parts:
        match = re.search(r"\d{10}-(\d{2})-\d{6}", part)
        if match:
            filing_date = f"20{match.group(1)}"
            break

    return {
        "ticker": ticker,
        "filing_date": filing_date,
        "form_type": form_type,
        "source_path": str(filing_path),
    }
