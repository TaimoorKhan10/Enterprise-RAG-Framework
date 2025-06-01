"""
Document processing module for handling various document formats.
"""

import logging
import os
from typing import Dict, List, Optional, Union, Any
import hashlib
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Processes documents of various formats into chunks suitable for embedding and retrieval.
    Supports PDF, DOCX, HTML, text, markdown, images (with OCR), audio (with transcription), and Excel.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the document processor with configuration options.

        Args:
            config: Configuration dictionary with processing options
        """
        self.config = config or {
            "chunk_size": 500,
            "chunk_overlap": 100,
            "chunking_strategy": "fixed",  # Options: fixed, semantic, recursive
            "extract_metadata": True,
            "ocr_enabled": False,
            "audio_transcription_enabled": False,
        }

        logger.info("Document processor initialized with config: %s", self.config)

        # Initialize format-specific processors as needed
        self._initialize_processors()

    def _initialize_processors(self) -> None:
        """Initialize format-specific document processors based on configuration."""
        self.processors = {}

        # Text processor (always enabled)
        from .text_processor import TextProcessor
        self.processors["text"] = TextProcessor(self.config)

        try:
            # PDF processor
            from .pdf_processor import PDFProcessor
            self.processors["pdf"] = PDFProcessor(self.config)
        except ImportError:
            logger.warning("PDF processing libraries not found, PDF processing disabled")

        try:
            # DOCX processor
            from .docx_processor import DocxProcessor
            self.processors["docx"] = DocxProcessor(self.config)
        except ImportError:
            logger.warning("DOCX processing libraries not found, DOCX processing disabled")

        try:
            # HTML processor
            from .html_processor import HTMLProcessor
            self.processors["html"] = HTMLProcessor(self.config)
        except ImportError:
            logger.warning("HTML processing libraries not found, HTML processing disabled")

        try:
            # Excel processor
            from .excel_processor import ExcelProcessor
            self.processors["excel"] = ExcelProcessor(self.config)
        except ImportError:
            logger.warning("Excel processing libraries not found, Excel processing disabled")

        # Image processor with OCR (optional)
        if self.config.get("ocr_enabled", False):
            try:
                from .image_processor import ImageProcessor
                self.processors["image"] = ImageProcessor(self.config)
                logger.info("Image processing with OCR enabled")
            except ImportError:
                logger.warning("OCR libraries not found, image processing disabled")

        # Audio transcription processor (optional)
        if self.config.get("audio_transcription_enabled", False):
            try:
                from .audio_processor import AudioProcessor
                self.processors["audio"] = AudioProcessor(self.config)
                logger.info("Audio transcription enabled")
            except ImportError:
                logger.warning("Audio processing libraries not found, audio transcription disabled")

    def process_documents(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process a list of documents into chunks suitable for embedding and retrieval.

        Args:
            documents: List of document dictionaries with at minimum text/content and metadata

        Returns:
            List of processed document chunks with text and metadata
        """
        processed_docs = []

        for doc in documents:
            try:
                processed_chunks = self.process_document(doc)
                processed_docs.extend(processed_chunks)
                logger.debug(
                    "Processed document into %d chunks: %s",
                    len(processed_chunks),
                    doc.get("metadata", {}).get("filename", "unnamed"),
                )
            except Exception as e:
                logger.error(
                    "Error processing document %s: %s",
                    doc.get("metadata", {}).get("filename", "unnamed"),
                    str(e),
                )
                continue

        logger.info(
            "Processed %d documents into %d chunks", len(documents), len(processed_docs)
        )
        return processed_docs

    def process_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process a single document into chunks.

        Args:
            document: Document dictionary with text/content and metadata

        Returns:
            List of processed document chunks with text and metadata
        """
        # Determine document type and select appropriate processor
        doc_type = self._determine_document_type(document)

        if doc_type in self.processors:
            processor = self.processors[doc_type]
            chunks = processor.process(document)
        else:
            # Fallback to text processor
            logger.warning(
                "No processor found for document type %s, using text processor",
                doc_type,
            )
            processor = self.processors["text"]
            chunks = processor.process(document)

        # Add common metadata and IDs to all chunks
        for i, chunk in enumerate(chunks):
            # Ensure metadata exists
            if "metadata" not in chunk:
                chunk["metadata"] = {}

            # Copy original document metadata if it exists
            if "metadata" in document:
                for key, value in document["metadata"].items():
                    if key not in chunk["metadata"]:
                        chunk["metadata"][key] = value

            # Add chunk-specific metadata
            chunk["metadata"]["chunk_id"] = i
            chunk["metadata"]["chunk_count"] = len(chunks)
            chunk["metadata"]["processed_at"] = datetime.now().isoformat()

            # Generate unique ID for the chunk
            if "id" not in chunk:
                chunk_text = chunk.get("text", "")
                chunk_id = hashlib.md5(
                    f"{chunk_text}_{i}_{time.time()}".encode()
                ).hexdigest()
                chunk["id"] = chunk_id

        return chunks

    def _determine_document_type(self, document: Dict[str, Any]) -> str:
        """
        Determine the document type based on metadata and content.

        Args:
            document: Document dictionary

        Returns:
            Document type string (pdf, docx, html, text, image, audio, excel)
        """
        # Check if type is explicitly provided
        if "type" in document:
            return document["type"]

        # Check metadata for filename or mimetype
        metadata = document.get("metadata", {})

        if "mimetype" in metadata:
            mimetype = metadata["mimetype"]
            if mimetype.startswith("application/pdf"):
                return "pdf"
            elif mimetype.startswith("application/vnd.openxmlformats-officedocument.wordprocessingml"):
                return "docx"
            elif mimetype.startswith("application/vnd.openxmlformats-officedocument.spreadsheetml"):
                return "excel"
            elif mimetype.startswith("text/html"):
                return "html"
            elif mimetype.startswith("text/"):
                return "text"
            elif mimetype.startswith("image/"):
                return "image"
            elif mimetype.startswith("audio/"):
                return "audio"

        # Check filename extension
        if "filename" in metadata:
            filename = metadata["filename"].lower()
            if filename.endswith(".pdf"):
                return "pdf"
            elif filename.endswith(".docx") or filename.endswith(".doc"):
                return "docx"
            elif filename.endswith(".xlsx") or filename.endswith(".xls"):
                return "excel"
            elif filename.endswith(".html") or filename.endswith(".htm"):
                return "html"
            elif filename.endswith(".txt") or filename.endswith(".md"):
                return "text"
            elif any(filename.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]):
                return "image"
            elif any(filename.endswith(ext) for ext in [".mp3", ".wav", ".ogg", ".m4a"]):
                return "audio"

        # Default to text
        logger.warning("Could not determine document type, defaulting to text")
        return "text"

    def load_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Load all supported documents from a directory.

        Args:
            directory_path: Path to directory containing documents

        Returns:
            List of document dictionaries with text and metadata
        """
        if not os.path.isdir(directory_path):
            raise ValueError(f"Directory not found: {directory_path}")

        documents = []

        for root, _, files in os.walk(directory_path):
            for file in files:
                try:
                    file_path = os.path.join(root, file)
                    doc = self.load_document(file_path)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    logger.error("Error loading file %s: %s", file_path, str(e))

        logger.info(
            "Loaded %d documents from directory %s", len(documents), directory_path
        )
        return documents

    def load_document(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load a document from a file path.

        Args:
            file_path: Path to document file

        Returns:
            Document dictionary with text and metadata, or None if unsupported
        """
        if not os.path.isfile(file_path):
            raise ValueError(f"File not found: {file_path}")

        # Get file extension and metadata
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()

        # Basic metadata
        metadata = {
            "filename": os.path.basename(file_path),
            "filepath": file_path,
            "file_size": os.path.getsize(file_path),
            "created_at": datetime.fromtimestamp(
                os.path.getctime(file_path)
            ).isoformat(),
            "modified_at": datetime.fromtimestamp(
                os.path.getmtime(file_path)
            ).isoformat(),
        }

        # Determine document type based on extension
        if file_extension == ".pdf" and "pdf" in self.processors:
            doc_type = "pdf"
            content = self.processors["pdf"].extract_text(file_path)
        elif file_extension in [".docx", ".doc"] and "docx" in self.processors:
            doc_type = "docx"
            content = self.processors["docx"].extract_text(file_path)
        elif file_extension in [".html", ".htm"] and "html" in self.processors:
            doc_type = "html"
            content = self.processors["html"].extract_text(file_path)
        elif file_extension in [".txt", ".md"]:
            doc_type = "text"
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        elif (
            file_extension in [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
            and "image" in self.processors
        ):
            doc_type = "image"
            content = self.processors["image"].extract_text(file_path)
        elif (
            file_extension in [".mp3", ".wav", ".ogg", ".m4a"]
            and "audio" in self.processors
        ):
            doc_type = "audio"
            content = self.processors["audio"].extract_text(file_path)
        else:
            logger.warning("Unsupported file type: %s", file_path)
            return None

        return {
            "content": content,
            "metadata": metadata,
            "type": doc_type,
        }
