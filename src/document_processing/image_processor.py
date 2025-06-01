"""
Image processing module with OCR capabilities.
"""

import logging
import os
from typing import Dict, List, Optional, Union, Any
import io
import base64

import numpy as np
from PIL import Image
import pytesseract

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Processes image files and extracts text content using OCR.
    Supports PNG, JPG, JPEG, TIFF and other common image formats.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the image processor with configuration options.

        Args:
            config: Configuration dictionary with processing options
        """
        self.config = config or {}
        
        # Set up OCR configuration
        self.ocr_config = self.config.get("ocr_config", {
            "lang": "eng",  # Default language
            "config": "--psm 3",  # Page segmentation mode: Fully automatic page segmentation
            "timeout": 30,  # Timeout in seconds
        })
        
        # Configure pytesseract path if provided
        tesseract_cmd = self.config.get("tesseract_cmd")
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            
        logger.info("Image processor initialized with OCR config: %s", self.ocr_config)

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process an image file and extract text using OCR.

        Args:
            file_path: Path to the image file

        Returns:
            Dictionary with text content and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")

        start_time = time.time()
        logger.info(f"Processing image file: {file_path}")

        try:
            # Load the image
            img = Image.open(file_path)
            
            # Extract basic metadata
            metadata = self._extract_metadata(img, file_path)
            
            # Perform OCR
            text = self._perform_ocr(img)
            
            processing_time = time.time() - start_time
            logger.info(f"Image processing completed in {processing_time:.2f}s")
            
            return {
                "text": text,
                "metadata": metadata,
                "processing_stats": {
                    "processing_time_seconds": processing_time,
                    "ocr_engine": "pytesseract",
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing image {file_path}: {str(e)}")
            raise

    def process_base64(self, base64_string: str, filename: str = "unknown.jpg") -> Dict[str, Any]:
        """
        Process an image from a base64 string.

        Args:
            base64_string: Base64-encoded image data
            filename: Virtual filename for metadata

        Returns:
            Dictionary with text content and metadata
        """
        try:
            # Decode base64 image
            imgdata = base64.b64decode(base64_string)
            img = Image.open(io.BytesIO(imgdata))
            
            # Extract basic metadata
            metadata = self._extract_metadata(img, filename)
            
            # Perform OCR
            text = self._perform_ocr(img)
            
            return {
                "text": text,
                "metadata": metadata,
                "processing_stats": {
                    "ocr_engine": "pytesseract",
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing base64 image: {str(e)}")
            raise

    def _perform_ocr(self, img: Image.Image) -> str:
        """
        Perform OCR on the image.

        Args:
            img: PIL Image object

        Returns:
            Extracted text string
        """
        # Preprocess image if needed
        if self.config.get("preprocess_image", True):
            img = self._preprocess_image(img)
        
        # Perform OCR
        text = pytesseract.image_to_string(
            img,
            lang=self.ocr_config.get("lang", "eng"),
            config=self.ocr_config.get("config", "--psm 3"),
            timeout=self.ocr_config.get("timeout", 30)
        )
        
        return text.strip()

    def _extract_metadata(self, img: Image.Image, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from the image.

        Args:
            img: PIL Image object
            file_path: Path to the image file or virtual filename

        Returns:
            Dictionary with metadata
        """
        filename = os.path.basename(file_path)
        file_ext = os.path.splitext(filename)[1].lower()
        
        metadata = {
            "source": "image",
            "source_type": file_ext.replace(".", ""),
            "filename": filename,
            "width": img.width,
            "height": img.height,
            "mode": img.mode,
            "format": img.format,
        }
        
        # Extract EXIF data if available
        if hasattr(img, "_getexif") and callable(img._getexif):
            exif = img._getexif()
            if exif:
                # Only include selected EXIF fields to avoid overwhelming metadata
                exif_fields = {
                    "DateTimeOriginal": "creation_date",
                    "Make": "camera_make",
                    "Model": "camera_model",
                    "GPSInfo": "gps_info",
                    "ImageDescription": "description",
                }
                
                for exif_tag, meta_key in exif_fields.items():
                    if exif_tag in exif:
                        metadata[meta_key] = str(exif[exif_tag])
        
        return metadata

    def _preprocess_image(self, img: Image.Image) -> Image.Image:
        """
        Preprocess the image to improve OCR results.

        Args:
            img: PIL Image object

        Returns:
            Preprocessed PIL Image object
        """
        # Convert to grayscale if color image
        if img.mode not in ('L', '1'):
            img = img.convert('L')
            
        # Apply additional preprocessing as needed
        # This could include noise removal, contrast enhancement, etc.
        
        return img
