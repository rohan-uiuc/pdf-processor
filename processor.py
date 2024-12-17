import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple, Optional
import json
from datetime import datetime
from pathlib import Path
import logging
import sys
import base64
import psutil
import gc
import nltk
import ssl

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.chunking.basic import chunk_elements
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from unstructured.documents.elements import Table
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy import cast, text, func
from database import Document, Segment, Chunk, init_db, get_session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pdf_processor.log"),
    ],
)
logger = logging.getLogger(__name__)
logging.getLogger("unstructured").setLevel(logging.DEBUG)
logging.getLogger("processor").setLevel(logging.DEBUG)


# def setup_nltk():
#     """Setup NLTK data"""
#     try:
#         # Try to create unverified HTTPS context
#         ssl._create_default_https_context = ssl._create_unverified_context
#         nltk.download('punkt')
#         nltk.download('averaged_perceptron_tagger')
#     except Exception as e:
#         logger.warning(f"Failed to download NLTK data: {e}")


def setup_detailed_logging():
    """Configure detailed logging with custom formatting"""
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - [%(levelname)s] - %(funcName)s:%(lineno)d - %(message)s"
    )

    # Update existing handlers with new formatter
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    # Add memory usage to log messages
    def memory_usage_mb():
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    # Add custom filter to include memory usage
    class MemoryFilter(logging.Filter):
        def filter(self, record):
            record.memory_mb = f"{memory_usage_mb():.2f}MB"
            return True

    logger.addFilter(MemoryFilter())


setup_detailed_logging()


class PDFProcessor:
    def __init__(self, output_dir: str = "extracted_images", engine=None):
        # First assign logger
        self.logger = logger
        self.logger.info("Initializing PDFProcessor with output_dir=%s", output_dir)

        # Then continue with other initializations
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

        self.client = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")
        self.engine = engine
        self.current_doc_id = None

        # setup_nltk()

        self.logger.info("PDFProcessor initialized successfully")

    def set_current_document(self, file_path: str) -> int:
        """Create or get document record and set current_doc_id."""
        session = get_session(self.engine)
        try:
            # Check if document already exists
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

            self.logger.info(f"Starting PDF processing with {initial_memory/1024/1024:.1f}MB memory usage")

            # Set OCR agent
            os.environ["OCR_AGENT"] = "unstructured.partition.utils.ocr_models.paddle_ocr.OCRAgentPaddle"

            # Create file-specific image directory
            file_specific_dir = Path(self.image_output_dir) / Path(file_path).stem
            file_specific_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Starting PDF processing for {file_path}")
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
                    hi_res_model_name="yolox",  # Use YOLO for better table detection
                    infer_table_structure=True,
                    extract_images_in_pdf=True,
                    extract_image_block_types=["Image", "Table"],
                    extract_image_block_to_payload=False,
                    extract_image_block_output_dir=str(file_specific_dir),
                    include_metadata=True,
                    include_original_elements=True,
                    # languages=["en"],  # Specify languages for OCR
                )

                if not elements:
                    self.logger.warning(f"No elements extracted from PDF {file_path}")
                    return [], []

                self.logger.info(f"Extracted {len(elements)} elements from PDF {file_path}")

                from unstructured.staging.base import elements_to_dicts
                dics_elements = elements_to_dicts(
                    elements=elements
                )
                # self.logger.info(f"Elements in dicts: {(dics_elements)}")

                # # Convert elements to JSON to preserve all metadata
                # from unstructured.staging.base import elements_to_json
                
                # # Convert all elements to JSON but only print first 10 for logging
                # json_elements = elements_to_json(
                #     elements=elements,
                #     indent=2
                # )
                # self.logger.info(f"Elements in JSON: {json_elements}")

                # Store the full JSON elements in processing_artifacts
                # session = get_session(self.engine)
                # try:
                #     doc = session.query(Document).get(doc_id)
                #     if doc:
                #         doc.processing_artifacts = {
                #             'elements': dics_elements,
                #             'processing_stats': {
                #                 'total_elements': len(elements),
                #                 'processed_at': datetime.utcnow().isoformat()
                #             }
                #         }
                #         session.commit()
                # finally:
                #     session.close()

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

    def chunk_elements(
        self, elements_str: List[Any], doc_id: int = None
    ) -> List[Dict[str, Any]]:
        """Chunk elements by title and return structured chunks with metadata."""
        self.logger.info("Starting element chunking with %d elements", len(elements_str))

        try:
            if doc_id is not None:
                self.current_doc_id = doc_id
                self.logger.info("Using provided doc_id: %d", doc_id)

            # from unstructured.staging.base import elements_from_json
            # elements = elements_from_json(elements_str)
            # self.logger.info(f"Elements from JSON: {elements}")

            from unstructured.staging.base import elements_from_dicts
            elements = elements_from_dicts(elements_str)
            # self.logger.info(f"Elements from dicts: {elements}")

            # Log element types before chunking
            element_types = {}
            for element in elements:
                element_type = type(element).__name__
                element_types[element_type] = element_types.get(element_type, 0) + 1

            self.logger.info(f"Element type distribution before chunking: {element_types}")

            # Create chunks while preserving table elements
            chunks = chunk_by_title(
                elements,
                multipage_sections=True,
                combine_text_under_n_chars=500,
                new_after_n_chars=1500,
                max_characters=2000,
                # preserve_table_structure=True
            )

            # Log chunk types after chunking
            chunk_types = {}
            for chunk in chunks:
                chunk_type = type(chunk).__name__
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

            self.logger.info(f"Chunk type distribution after chunking: {chunk_types}")

            # Structure chunks with metadata
            structured_chunks = []
            session = get_session(self.engine)
            try:
                for idx, chunk in enumerate(chunks, 1):
                    # Convert chunk to JSON to preserve all metadata
                    from unstructured.staging.base import elements_to_json
                    chunk_json = elements_to_json([chunk], indent=2)

                    # Create structured chunk data
                    chunk_data = {
                        'chunk_number': idx,
                        'content': chunk_json,
                        'type': type(chunk).__name__,
                    }
                    structured_chunks.append(chunk_data)

                    if idx % 100 == 0:
                        self.logger.info(f"Processed {idx} chunks...")

                self.logger.info(f"Successfully chunked {len(structured_chunks)} chunks")
                return structured_chunks

            except Exception as e:
                session.rollback()
                self.logger.error(f"Failed to process chunks: {str(e)}")
                raise
            finally:
                session.close()

        except Exception as e:
            self.logger.error(f"Error chunking elements: {str(e)}")
            raise

    async def process_tables(
        self, elements: List[Any], image_paths: List[str]
    ) -> List[Dict[str, Any]]:
        """Process tables from elements and images with improved metadata handling."""
        self.logger.info("Starting table processing")
        table_results = []
        try:
            # Get all table elements with their HTML and metadata
            table_elements = []
            for e in elements:
                if isinstance(e, Table) or (hasattr(e, "type") and e.type == "Table"):
                    table_data = {
                        "element": e,
                        "html": (
                            getattr(e, "table_html", None)
                            or getattr(e.metadata, "text_as_html", "")
                            if hasattr(e, "metadata")
                            else ""
                        ),
                        "metadata": getattr(e, "metadata", {}),
                        "coordinates": (
                            getattr(e.metadata, "coordinates", None)
                            if hasattr(e, "metadata")
                            else None
                        ),
                    }
                    table_elements.append(table_data)

            # Get all potential table images
            table_images = [p for p in image_paths if "table" in p.lower()]

            self.logger.info(
                f"Found {len(table_elements)} table elements and {len(table_images)} table images"
            )

            # Process each table with its corresponding image
            for idx, table_data in enumerate(table_elements):
                table_result = {
                    "table_index": idx,
                    "html": table_data["html"],
                    "coordinates": table_data["coordinates"],
                    "extracted_data": None,
                    "image_path": None,
                }

                # Find matching image if exists
                matching_image = None
                if table_images:
                    # TODO: Implement better matching logic using coordinates
                    matching_image = table_images.pop(0)
                    table_result["image_path"] = matching_image

                if matching_image:
                    analysis = await self.process_table_image(
                        matching_image, table_data["html"]
                    )
                    if analysis:
                        table_result["extracted_data"] = analysis

                table_results.append(table_result)

            self.logger.info(
                f"Completed table processing. Processed {len(table_results)} tables"
            )
            return table_results

        except Exception as e:
            self.logger.error(f"Error processing tables: {str(e)}")
            raise

    def save_to_db(
        self,
        filename: str,
        elements: List[Any],
        chunks: List[Dict[str, Any]],
        table_results: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> None:
        """Save processed data to database with improved metadata handling and chunk storage."""
        self.logger.info("Starting database save for %s", filename)
        session = get_session(self.engine)

        try:
            # Create or get existing document
            doc = (
                session.query(Document)
                .filter_by(readable_filename=filename)
                .first()
            )
            
            if not doc:
                self.logger.info("Creating new document record")
                doc = Document(
                    readable_filename=filename,
                    processing_artifacts={
                        "elements": [str(e) for e in elements],
                        "table_results": table_results,
                        "processing_stats": {
                            "total_elements": len(elements),
                            "total_chunks": len(chunks),
                            "total_tables": len(table_results),
                            "processed_at": datetime.utcnow().isoformat(),
                        },
                    },
                    metadata_schema=metadata.get("schema", {}),
                    created_at=datetime.utcnow(),
                    partition_status="completed",
                    chunk_status="completed",
                )
                session.add(doc)
                session.flush()
            
            self.logger.info("Created/Retrieved document record with ID %d", doc.id)

            current_segment = None
            segment_number = 0
            chunk_number = 0

            # Process segments and chunks
            for chunk in chunks:
                try:
                    chunk_content = chunk.get("content", "")
                    chunk_type = chunk.get("type", "text").lower()
                    
                    # Check if this chunk represents a title or new section
                    if chunk_type == "title":
                        segment_number += 1
                        current_segment = Segment(
                            document_id=doc.id,
                            segment_number=segment_number,
                            title=chunk_content,
                            created_at=datetime.utcnow(),
                        )
                        session.add(current_segment)
                        session.flush()
                        self.logger.info(
                            f"Created new segment {segment_number}: {chunk_content[:50]}..."
                        )

                    chunk_number += 1
                    
                    # Extract table-specific data if present
                    table_html = None
                    table_data = None
                    if chunk_type == "table":
                        table_html = chunk.get("table_html")
                        table_data = chunk.get("table_data")

                    # Create chunk record with proper metadata
                    chunk_metadata = {
                        "chunk_number": chunk_number,
                        "original_type": chunk.get("type"),
                        "extraction_method": chunk.get("extraction_method", "default"),
                        "confidence_score": chunk.get("confidence_score", 100),
                    }

                    db_chunk = Chunk(
                        document_id=doc.id,
                        segment_id=current_segment.id if current_segment else None,
                        chunk_number=chunk_number,
                        content=chunk_content,
                        chunk_type=chunk_type,
                        table_html=table_html,
                        table_data=table_data,
                        chunk_metadata=chunk_metadata,
                        created_at=datetime.utcnow(),
                    )
                    session.add(db_chunk)

                    # Periodically flush to avoid memory issues with large documents
                    if chunk_number % 100 == 0:
                        session.flush()
                        self.logger.info(f"Processed {chunk_number} chunks...")

                except Exception as e:
                    self.logger.error(
                        f"Error processing chunk {chunk_number}: {str(e)}"
                    )
                    raise

            session.commit()
            self.logger.info(
                "Successfully saved document %s with %d chunks across %d segments",
                filename,
                chunk_number,
                segment_number,
            )

        except Exception as e:
            session.rollback()
            self.logger.error("Database save failed: %s", str(e), exc_info=True)
            raise
        finally:
            session.close()

    async def process_table_image(
        self, image_path: str, table_html: str
    ) -> Optional[str]:
        """Process table image using OpenAI API."""
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode()

                # Construct a more detailed prompt
                prompt = (
                    "Analyze this table image and its html representation and extract its data in a structured format. "
                    "Return the data as a html table. Include all columns, headers, and rows "
                    "exactly as they appear in the image. If there are merged cells or special "
                    "formatting, preserve that in the output."
                )

                message = HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            },
                        },
                    ]
                )

                self.logger.info(
                    f"Sending table image {image_path} to OpenAI for analysis"
                )
                response = await self.client.ainvoke([message])

                if not response or not response.content:
                    self.logger.warning(
                        f"No response content for table image {image_path}"
                    )
                    return None

                self.logger.info(f"Successfully processed table image {image_path}")
                return response.content

        except Exception as e:
            self.logger.error(f"Error processing table image {image_path}: {str(e)}")
            return None

    async def define_schema(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Define document schema using trustcall."""
        try:
            # Create schema extractor if not exists
            if not hasattr(self, "schema_extractor"):
                from trustcall import create_extractor
                from pydantic import BaseModel
                from typing import List

                class FieldDefinition(BaseModel):
                    field_name: str
                    description: str
                    type: str
                    required: bool = False
                    example: str = ""

                class DocumentSchemaDefinition(BaseModel):
                    schema_type: str
                    schema_version: str
                    fields: List[FieldDefinition]
                    description: str

                self.schema_extractor = create_extractor(
                    self.client,
                    tools=[DocumentSchemaDefinition],
                    tool_choice="DocumentSchemaDefinition",
                    enable_inserts=True,
                )

            # Analyze document content to determine schema
            chunks = metadata.get("chunks", [])
            chunk_texts = [
                chunk["content"] for chunk in chunks if chunk["type"] != "Title"
            ]
            sample_content = "\n\n".join(
                chunk_texts[:3]
            )  # Use first 3 chunks as sample

            prompt = f"""Based on the following document content, define a schema for extracting metadata.
			The schema should capture key information that would be relevant for a course document.

			Sample content:
			{sample_content}

			Define a schema that includes fields for course-related information like:
			- Course name, code, department
			- Instructor information
			- Course schedule, prerequisites
			- Learning objectives
			- And any other relevant fields based on the content

			Return as a DocumentSchemaDefinition with appropriate field definitions.
			"""

            result = await self.schema_extractor.ainvoke(
                {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a schema definition expert.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "existing": {},
                }
            )

            if not result or not result.get("responses"):
                raise ValueError("No schema definition generated")

            schema_def = result["responses"][0]
            return schema_def.model_dump()

        except Exception as e:
            self.logger.error(f"Error defining schema: {str(e)}")
            raise

    async def extract_metadata(
        self, doc_id: int, schema: Dict[str, Any], chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract metadata from chunks using trustcall based on schema."""
        try:
            # Create metadata extractor if not exists
            if not hasattr(self, "metadata_extractor"):
                from trustcall import create_extractor
                from pydantic import BaseModel
                from typing import List, Any

                class MetadataValue(BaseModel):
                    field_name: str
                    value: Any
                    confidence_score: int
                    source_chunk_id: Optional[int] = None
                    extraction_method: str = "gpt-4"

                class DocumentMetadataExtraction(BaseModel):
                    document_id: int
                    metadata_values: List[MetadataValue]

                self.metadata_extractor = create_extractor(
                    self.client,
                    tools=[DocumentMetadataExtraction],
                    tool_choice="DocumentMetadataExtraction",
                    enable_inserts=True,
                )

            # Process chunks in batches to extract metadata
            batch_size = 5
            all_metadata = []

            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                chunk_texts = [
                    f"[Chunk {chunk.get('chunk_number', i)}]\n{chunk['content']}"
                    for chunk in batch
                ]

                schema_str = json.dumps(schema, indent=2)
                chunks_str = "\n\n".join(chunk_texts)

                prompt = (
                    "Extract metadata according to the following schema:\n"
                    f"{schema_str}\n\n"
                    "From these document chunks:\n"
                    f"{chunks_str}\n\n"
                    "Return a DocumentMetadataExtraction with appropriate metadata values.\n"
                    "Include confidence scores (0-100) for each extraction.\n"
                    "If a field's value is found across multiple chunks, combine the information."
                )

                result = await self.metadata_extractor.ainvoke(
                    {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a metadata extraction expert.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "existing": {"document_id": doc_id},
                    }
                )

                if result and result.get("responses"):
                    metadata_extraction = result["responses"][0]
                    all_metadata.extend(metadata_extraction.metadata_values)

            return [m.model_dump() for m in all_metadata]

        except Exception as e:
            self.logger.error(f"Error extracting metadata: {str(e)}")
            raise

    async def process_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple PDF files with complete pipeline."""
        results = []
        for file_path in file_paths:
            try:
                # Step 1: Process PDF
                elements, image_paths = self.process_pdf(file_path)

                # Step 2: Chunk Content
                structured_chunks = self.chunk_elements(elements)

                # Step 3: Process Tables
                table_results = await self.process_tables(elements, image_paths)

                # Step 4: Define Schema
                metadata = {
                    "elements": [str(e) for e in elements],
                    "image_paths": image_paths,
                    "chunks": structured_chunks,
                    "table_results": table_results,
                    "original_filename": os.path.basename(file_path),
                }

                schema = await self.define_schema(metadata)

                # Step 5: Extract Metadata
                extracted_metadata = await self.extract_metadata(
                    0,  # temporary doc_id, will be updated in save_to_db
                    schema,
                    structured_chunks,
                )

                # Save everything to database
                self.save_to_db(
                    os.path.basename(file_path),
                    elements,
                    structured_chunks,
                    table_results,
                    {"schema": schema, "extracted_metadata": extracted_metadata},
                )

                results.append(
                    {
                        "filename": os.path.basename(file_path),
                        "status": "success",
                        "chunks": len(structured_chunks),
                        "tables_processed": len(table_results),
                        "metadata_fields": len(extracted_metadata),
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "filename": os.path.basename(file_path),
                        "status": "error",
                        "error": str(e),
                    }
                )

        return results

    def load_elements(self, element_strings: List[str]) -> List[Any]:
        """Reconstruct elements from stored strings."""
        # This is a simplified version - you might need to enhance it based on your needs
        from unstructured.documents.elements import Text, Title, Table
        elements = []
        for e_str in element_strings:
            if "Title:" in e_str:
                elements.append(Title(text=e_str.replace("Title: ", "")))
            elif "Table:" in e_str:
                elements.append(Table(text=e_str.replace("Table: ", "")))
            else:
                elements.append(Text(text=e_str))
        return elements


async def pipeline_process_document(file_path: str):
    """Complete pipeline: upload, partition, table process, chunk, define schema, extract metadata."""
    session = get_session()
    try:
        doc = Document(
            readable_filename=Path(file_path).name,
            partition_status='pending',
            table_status='pending',
            chunk_status='pending',
            db_save_status='pending',
            created_at=datetime.utcnow()
        )
        session.add(doc)
        session.commit()

        doc_id = doc.id
        processor = PDFProcessor()

        # Step 1: Partition PDF
        doc.partition_status = 'processing'
        session.commit()

        elements, image_paths = processor.process_pdf(file_path)
        doc.partition_status = 'completed'
        doc.processing_artifacts = {
            'elements': [str(e) for e in elements],
            'image_paths': image_paths,
            'processing_stats': {
                'total_elements': len(elements),
                'total_images': len(image_paths),
                'processed_at': datetime.utcnow().isoformat()
            }
        }
        session.commit()

        # Step 2: Process tables
        doc.table_status = 'processing'
        session.commit()

        table_results = await processor.process_tables(elements, image_paths)
        
        # Step 3: Chunking
        doc.chunk_status = 'processing'
        session.commit()

        chunks = processor.chunk_elements(elements)
        
        # Step 4: Define schema
        schema = await processor.define_schema({"chunks": chunks})
        
        # Step 5: Extract metadata
        metadata = await processor.extract_metadata(doc_id, schema, chunks)
        
        # Final step: Save everything using the fixed save_to_db method
        processor.save_to_db(
            filename=Path(file_path).name,
            elements=elements,
            chunks=chunks,
            table_results=table_results,
            metadata={
                "schema": schema,
                "extracted_metadata": metadata
            }
        )
        
        doc.db_save_status = 'completed'
        session.commit()
        
        logger.info("Pipeline completed for document: %s", file_path)
        
    except Exception as e:
        logger.error(f"Pipeline failed for document {file_path}: {str(e)}")
        doc.last_error = str(e)
        doc.db_save_status = 'failed'
        session.commit()
        raise
    finally:
        session.close()
