import os
from typing import List, Dict, Any, Tuple
from pathlib import Path
import logging
from datetime import datetime
import psutil
import gc

from unstructured.partition.pdf import partition_pdf
from database import Document, get_session

from utils.logging_config import setup_detailed_logging

logger = setup_detailed_logging()

class PDFProcessor:
    def __init__(self, output_dir: str = "extracted_images", engine=None):
        self.logger = logger
        self.logger.info("Initializing PDFProcessor with output_dir=%s", output_dir)

        self.image_output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Ensure cache directories exist with proper permissions
        cache_dir = os.path.join(os.getcwd(), ".cache", "huggingface")
        paddle_dir = os.path.join(os.getcwd(), ".paddleocr")
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(paddle_dir, exist_ok=True)

        # Set PaddleOCR environment variable to use the correct path
        os.environ["PADDLE_HOME"] = paddle_dir
        os.environ["HOME"] = os.getcwd()  # Ensure HOME is set correctly

        self.engine = engine
        self.current_doc_id = None

        self.logger.info("PDFProcessor initialized successfully")

    def set_current_document(self, file_path: str) -> int:
        """Create or get document record and set current_doc_id."""
        session = get_session(self.engine)
        try:
            doc = (
                session.query(Document)
                .filter_by(readable_filename=os.path.basename(file_path))
                .first()
            )

            if not doc:
                doc = Document(
                    readable_filename=os.path.basename(file_path),
                    created_at=datetime.utcnow(),
                    partition_status="processing",
                )
                session.add(doc)
                session.flush()

            self.current_doc_id = doc.id
            session.commit()
            return doc.id
        except Exception as e:
            session.rollback()
            self.logger.error(f"Failed to set current document: {str(e)}")
            raise
        finally:
            session.close()

    def process_pdf(self, file_path: str) -> Tuple[List[Any], List[str]]:
        """Process a single PDF file."""
        self.logger.info("Starting PDF processing for file: %s", file_path)
        try:
            doc_id = self.set_current_document(file_path)
            self.logger.info("Set document ID: %d", doc_id)

            file_size = os.path.getsize(file_path)
            self.logger.info("File size: %.2f MB", file_size / 1024 / 1024)

            if file_size > 100_000_000:
                self.logger.error("File too large: %.2f MB", file_size / 1024 / 1024)
                raise ValueError(f"File too large ({file_size} bytes)")

            process = psutil.Process()
            initial_memory = process.memory_info().rss
            self.logger.info("Initial memory usage: %.2f MB", initial_memory / 1024 / 1024)

            # Set OCR agent
            os.environ["OCR_AGENT"] = "unstructured.partition.utils.ocr_models.paddle_ocr.OCRAgentPaddle"

            # Create file-specific image directory
            file_specific_dir = Path(self.image_output_dir) / Path(file_path).stem
            file_specific_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Verify file exists and is readable
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"PDF file not found: {file_path}")

                # Verify file is actually a PDF
                with open(file_path, "rb") as f:
                    header = f.read(4)
                    if header != b"%PDF":
                        raise ValueError(f"File {file_path} is not a valid PDF")

                # Set environment variables for image padding
                os.environ["EXTRACT_IMAGE_BLOCK_CROP_HORIZONTAL_PAD"] = "20"
                os.environ["EXTRACT_IMAGE_BLOCK_CROP_VERTICAL_PAD"] = "10"

                elements = partition_pdf(
                    filename=file_path,
                    mode="elements",
                    strategy="hi_res",
                    hi_res_model_name="yolox",
                    infer_table_structure=True,
                    extract_images_in_pdf=True,
                    extract_image_block_types=["Image", "Table"],
                    extract_image_block_to_payload=False,
                    extract_image_block_output_dir=str(file_specific_dir),
                    include_metadata=True,
                    include_original_elements=True,
                )

                if not elements:
                    self.logger.warning(f"No elements extracted from PDF {file_path}")
                    return [], []

                self.logger.info(f"Extracted {len(elements)} elements from PDF {file_path}")

                from unstructured.staging.base import elements_to_dicts
                dics_elements = elements_to_dicts(elements=elements)

                # Collect image paths
                image_paths = []
                if Path(self.image_output_dir).exists():
                    image_paths = [
                        str(f)
                        for f in Path(self.image_output_dir).glob("**/*.[jJ][pP][gG]")
                    ]
                    self.logger.info(f"Found {len(image_paths)} images in {self.image_output_dir}")

                self.logger.info("PDF processing completed successfully")
                self.logger.debug("Elements extracted: %d, Image paths: %d",
                                len(elements), len(image_paths))
                return dics_elements, image_paths

            except Exception as e:
                self.logger.error(f"Error processing PDF {file_path}: {str(e)}", exc_info=True)
                raise Exception(f"Failed to process PDF {file_path}: {str(e)}") from e

            finally:
                final_memory = process.memory_info().rss
                self.logger.info(f"Completed processing with {final_memory/1024/1024:.1f}MB memory usage")
                gc.collect()

        except Exception as e:
            self.logger.error(f"Failed to process {file_path}: {str(e)}")
            raise
