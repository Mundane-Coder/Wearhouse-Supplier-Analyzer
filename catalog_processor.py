import pdfplumber
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extracts text and tables from a PDF file.
    Returns a list of pages with text and tables.
    """
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            page_data = {
                "page": page_num,
                "text": page.extract_text() or "",
                "tables": []
            }

            # Extract tables if any
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    # Remove empty rows
                    clean_table = [
                        row for row in table
                        if any(cell and str(cell).strip() for cell in row)
                    ]
                    if clean_table:
                        page_data["tables"].append(clean_table)

            pages.append(page_data)

    return pages


def table_to_chunks(table: list[list], page_num: int) -> list[dict]:
    """
    Converts a table into chunks.
    The first row is considered the header (SKU, name, price, etc.)
    Each subsequent row = one product = one chunk.
    """
    if len(table) < 2:
        return []

    headers = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(table[0])]
    chunks = []

    for row in table[1:]:
        # Collect row data into a dict {header: value}
        row_data = {}
        for i, cell in enumerate(row):
            if i < len(headers):
                row_data[headers[i]] = str(cell).strip() if cell else ""

        # Form chunk text - understandable for embedding
        text_parts = [f"{k}: {v}" for k, v in row_data.items() if v]
        chunk_text = " | ".join(text_parts)

        if not chunk_text.strip():
            continue

        # Metadata for filtering
        metadata = {
            "page": page_num,
            "source_type": "table",
        }

        # Try to extract standard catalog fields
        for key, value in row_data.items():
            key_lower = key.lower()
            if any(x in key_lower for x in ["sku", "article", "ref"]):
                metadata["sku"] = value
            elif any(x in key_lower for x in ["price", "cost"]):
                metadata["price"] = value
            elif any(x in key_lower for x in ["category", "type"]):
                metadata["category"] = value
            elif any(x in key_lower for x in ["name", "product"]):
                metadata["name"] = value

        chunks.append({
            "text": chunk_text,
            "metadata": metadata
        })

    return chunks


def text_to_chunks(text: str, page_num: int, chunk_size: int = 500) -> list[dict]:
    """
    Splits regular text into chunks (for pages without tables).
    """
    if not text.strip():
        return []

    # Simple split by paragraphs, then by size
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) <= chunk_size:
            current_chunk += " " + para if current_chunk else para
        else:
            if current_chunk:
                chunks.append({
                    "text": current_chunk,
                    "metadata": {"page": page_num, "source_type": "text"}
                })
            current_chunk = para

    if current_chunk:
        chunks.append({
            "text": current_chunk,
            "metadata": {"page": page_num, "source_type": "text"}
        })

    return chunks


def process_catalog(pdf_path: str) -> list[dict]:
    """
    Main function: takes a PDF path,
    returns a list of chunks ready for embeddings.
    """
    pages = extract_text_from_pdf(pdf_path)
    all_chunks = []

    for page_data in pages:
        page_num = page_data["page"]

        # Priority to tables - they are more structured and important for the catalog
        if page_data["tables"]:
            for table in page_data["tables"]:
                chunks = table_to_chunks(table, page_num)
                all_chunks.extend(chunks)
        else:
            # For pages without tables - regular text chunking
            chunks = text_to_chunks(page_data["text"], page_num)
            all_chunks.extend(chunks)

    return all_chunks


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        chunks = process_catalog(pdf_file)
        print(f"Extracted chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"\n--- Chunk {i} ---")
            print(f"Text: {chunk['text'][:100]}...")
            print(f"Metadata: {chunk['metadata']}")
    else:
        print("Usage: python catalog_processor.py path/to/catalog.pdf")