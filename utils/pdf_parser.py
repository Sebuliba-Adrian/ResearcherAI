"""
PDF Parser - Extract text and metadata from PDF files
====================================================

Uses PyPDF2 for text extraction and basic metadata parsing
"""

import os
import re
import logging
from datetime import datetime
from typing import Dict, List, Optional
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)


class PDFParser:
    """Extract text and metadata from PDF research papers"""

    def __init__(self):
        self.supported_extensions = ['.pdf']

    def parse_pdf(self, file_path: str) -> Dict:
        """
        Parse a PDF file and extract text + metadata

        Args:
            file_path: Path to PDF file

        Returns:
            Dictionary with paper metadata:
            {
                "id": "local_pdf_123",
                "title": "Extracted title",
                "abstract": "Extracted abstract",
                "authors": ["Author 1", "Author 2"],
                "published": "2025-01-01",
                "url": "file:///path/to/paper.pdf",
                "source": "Local PDF Upload",
                "topics": ["keywords"],
                "full_text": "Complete PDF text"
            }
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if not file_path.lower().endswith('.pdf'):
            raise ValueError(f"File must be a PDF: {file_path}")

        try:
            # Read PDF
            reader = PdfReader(file_path)

            # Extract metadata from PDF properties
            pdf_metadata = reader.metadata if reader.metadata else {}

            # Extract all text
            full_text = ""
            for page in reader.pages:
                full_text += page.extract_text() + "\n"

            # Clean text
            full_text = self._clean_text(full_text)

            # Extract structured metadata
            metadata = self._extract_metadata(full_text, pdf_metadata, file_path)

            # Add full text
            metadata["full_text"] = full_text
            metadata["num_pages"] = len(reader.pages)

            logger.info(f"Successfully parsed PDF: {metadata['title']}")
            return metadata

        except Exception as e:
            logger.error(f"Failed to parse PDF {file_path}: {e}")
            raise

    def parse_pdf_bytes(self, file_bytes: bytes, filename: str) -> Dict:
        """
        Parse PDF from bytes (for file uploads)

        Args:
            file_bytes: PDF file content as bytes
            filename: Original filename

        Returns:
            Dictionary with paper metadata
        """
        # Save temporarily
        temp_dir = "./volumes/uploads"
        os.makedirs(temp_dir, exist_ok=True)

        temp_path = os.path.join(temp_dir, filename)

        try:
            # Write bytes to temp file
            with open(temp_path, 'wb') as f:
                f.write(file_bytes)

            # Parse
            result = self.parse_pdf(temp_path)

            # Update URL to reflect temp location
            result["url"] = f"file://{temp_path}"
            result["filename"] = filename

            return result

        except Exception as e:
            # Clean up on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove page numbers (common patterns)
        text = re.sub(r'\b\d+\s*$', '', text, flags=re.MULTILINE)

        # Remove headers/footers (heuristic)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Skip very short lines (likely headers/footers)
            if len(line.strip()) < 10:
                continue
            cleaned_lines.append(line)

        text = '\n'.join(cleaned_lines)

        return text.strip()

    def _extract_metadata(self, text: str, pdf_metadata: Dict, file_path: str) -> Dict:
        """
        Extract structured metadata from PDF text and metadata

        Uses heuristics to find:
        - Title (usually first large text block or from PDF metadata)
        - Authors (patterns like "by Author Name" or PDF metadata)
        - Abstract (section labeled "Abstract")
        - Keywords/Topics
        - Date
        """
        metadata = {
            "id": self._generate_id(file_path),
            "source": "Local PDF Upload",
            "url": f"file://{os.path.abspath(file_path)}"
        }

        # 1. Extract title
        metadata["title"] = self._extract_title(text, pdf_metadata)

        # 2. Extract authors
        metadata["authors"] = self._extract_authors(text, pdf_metadata)

        # 3. Extract abstract
        metadata["abstract"] = self._extract_abstract(text)

        # 4. Extract keywords/topics
        metadata["topics"] = self._extract_keywords(text)

        # 5. Extract date
        metadata["published"] = self._extract_date(text, pdf_metadata, file_path)

        return metadata

    def _generate_id(self, file_path: str) -> str:
        """Generate unique ID for local PDF"""
        filename = os.path.basename(file_path)
        # Use filename without extension + timestamp
        name_part = os.path.splitext(filename)[0]
        # Sanitize
        name_part = re.sub(r'[^a-zA-Z0-9_-]', '_', name_part)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"local_pdf_{name_part}_{timestamp}"

    def _extract_title(self, text: str, pdf_metadata: Dict) -> str:
        """Extract paper title"""
        # Try PDF metadata first
        if pdf_metadata:
            if '/Title' in pdf_metadata and pdf_metadata['/Title']:
                return str(pdf_metadata['/Title']).strip()

        # Heuristic: First substantial line (50+ chars) is usually the title
        lines = text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if 50 <= len(line) <= 200:  # Titles are usually 50-200 chars
                # Skip if it looks like a header
                if not re.match(r'^(page|abstract|introduction|\d+)', line, re.IGNORECASE):
                    return line

        # Fallback: Use first line
        if lines:
            return lines[0].strip()

        return "Unknown Title"

    def _extract_authors(self, text: str, pdf_metadata: Dict) -> List[str]:
        """Extract author names"""
        authors = []

        # Try PDF metadata first
        if pdf_metadata and '/Author' in pdf_metadata:
            author_str = str(pdf_metadata['/Author']).strip()
            if author_str:
                # Split by common delimiters
                authors = re.split(r',|;|\band\b', author_str)
                authors = [a.strip() for a in authors if a.strip()]
                if authors:
                    return authors

        # Heuristic: Look for author patterns in first 500 chars
        first_section = text[:500]

        # Pattern 1: "by Author Name"
        match = re.search(r'by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', first_section)
        if match:
            authors.append(match.group(1))

        # Pattern 2: Multiple capitalized names after title
        # (This is a simplified heuristic)
        lines = first_section.split('\n')
        for i, line in enumerate(lines[1:5]):  # Check lines 2-5
            # Look for lines with multiple capitalized words
            if re.match(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)$', line.strip()):
                authors.append(line.strip())

        # Deduplicate
        authors = list(dict.fromkeys(authors))

        return authors if authors else ["Unknown Author"]

    def _extract_abstract(self, text: str) -> str:
        """Extract abstract section"""
        # Pattern: "Abstract" followed by text until next section
        abstract_match = re.search(
            r'abstract[:\s]+(.*?)(?=\n\s*(?:introduction|keywords|1\.|i\.|background))',
            text,
            re.IGNORECASE | re.DOTALL
        )

        if abstract_match:
            abstract = abstract_match.group(1).strip()
            # Limit to reasonable length (1500 chars max)
            if len(abstract) > 1500:
                abstract = abstract[:1500] + "..."
            return abstract

        # Fallback: Use first 300 chars after title as abstract
        # Skip first 200 chars (likely title + authors)
        if len(text) > 200:
            abstract = text[200:500].strip()
            return abstract

        return "No abstract found"

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords/topics"""
        keywords = []

        # Pattern: "Keywords:" followed by comma-separated terms
        keyword_match = re.search(
            r'keywords?[:\s]+(.*?)(?=\n\s*(?:introduction|abstract|1\.))',
            text,
            re.IGNORECASE | re.DOTALL
        )

        if keyword_match:
            keyword_str = keyword_match.group(1).strip()
            # Split by comma or semicolon
            keywords = re.split(r'[,;]', keyword_str)
            keywords = [k.strip().lower() for k in keywords if k.strip()]
            # Limit to 10 keywords
            keywords = keywords[:10]

        # If no keywords found, extract from title/abstract
        if not keywords:
            # Simple extraction: capitalized technical terms
            terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text[:1000])
            # Filter common words
            stopwords = {'The', 'A', 'An', 'In', 'On', 'At', 'To', 'For', 'Of', 'And', 'Or', 'But'}
            keywords = [t.lower() for t in terms if t not in stopwords]
            # Get unique and limit
            keywords = list(dict.fromkeys(keywords))[:10]

        return keywords

    def _extract_date(self, text: str, pdf_metadata: Dict, file_path: str) -> str:
        """Extract publication date"""
        # Try PDF metadata first
        if pdf_metadata and '/CreationDate' in pdf_metadata:
            creation_date = str(pdf_metadata['/CreationDate'])
            # Parse PDF date format (D:YYYYMMDDHHmmSS)
            match = re.match(r'D:(\d{4})(\d{2})(\d{2})', creation_date)
            if match:
                year, month, day = match.groups()
                return f"{year}-{month}-{day}"

        # Heuristic: Look for year patterns in first 500 chars
        first_section = text[:500]
        year_match = re.search(r'\b(20\d{2}|19\d{2})\b', first_section)
        if year_match:
            year = year_match.group(1)
            return f"{year}-01-01"  # Default to Jan 1

        # Fallback: Use file modification time
        if os.path.exists(file_path):
            mtime = os.path.getmtime(file_path)
            return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

        # Last resort: Current date
        return datetime.now().strftime("%Y-%m-%d")


# ============================================================================
# Convenience functions
# ============================================================================

def parse_pdf_file(file_path: str) -> Dict:
    """
    Parse a PDF file (convenience function)

    Args:
        file_path: Path to PDF file

    Returns:
        Paper metadata dictionary
    """
    parser = PDFParser()
    return parser.parse_pdf(file_path)


def parse_pdf_upload(file_bytes: bytes, filename: str) -> Dict:
    """
    Parse uploaded PDF file (convenience function)

    Args:
        file_bytes: PDF content as bytes
        filename: Original filename

    Returns:
        Paper metadata dictionary
    """
    parser = PDFParser()
    return parser.parse_pdf_bytes(file_bytes, filename)
