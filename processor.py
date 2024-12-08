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

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from unstructured.documents.elements import Table

from database import Document, Segment, Chunk, init_db, get_session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pdf_processor.log')
    ]
)
logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, output_dir: str = "extracted_images"):
        self.image_output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.client = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")
        self.engine = init_db()
        self.logger = logging.getLogger(__name__)
        
    def process_pdf(self, file_path: str) -> Tuple[List[Any], List[str]]:
        """Process a single PDF file."""
        try:
            # Verify file size before processing
            file_size = os.path.getsize(file_path)
            if file_size > 100_000_000:  # 100MB limit
                raise ValueError(f"File too large ({file_size} bytes)")
            
            # Add memory monitoring
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            self.logger.info(f"Starting PDF processing with {initial_memory/1024/1024:.1f}MB memory usage")
            
            os.environ['OCR_AGENT'] = 'unstructured.partition.utils.ocr_models.paddle_ocr.OCRAgentPaddle'
            # Create file-specific image directory
            file_specific_dir = Path(self.image_output_dir) / Path(file_path).stem
            file_specific_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Starting PDF processing for {file_path}")
            try:
                # Verify file exists and is readable
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"PDF file not found: {file_path}")
                
                # Verify file is actually a PDF
                with open(file_path, 'rb') as f:
                    header = f.read(4)
                    if header != b'%PDF':
                        raise ValueError(f"File {file_path} is not a valid PDF")

                chunks = partition_pdf(
                    filename=file_path,
                    mode="elements", 
                    strategy="hi_res", 
                    hi_res_model_name="yolox",
                    infer_table_structure=True,
                    # extract_images_in_pdf=True
                    # extract_image_block_to_payload=True,
                    # strategy="hi_res",
                    # # Set environment variables for image padding if needed
                    # # os.environ["EXTRACT_IMAGE_BLOCK_CROP_HORIZONTAL_PAD"] = "20"
                    # # os.environ["EXTRACT_IMAGE_BLOCK_CROP_VERTICAL_PAD"] = "10"
                    # extract_images_in_pdf=True,
                    extract_images_in_pdf=True,                            # mandatory to set as ``True``
                    extract_image_block_types=["Image", "Table"],          # optional
                    extract_image_block_to_payload=False,
                    extract_image_block_output_dir=str(file_specific_dir),
                    # extract_image_block_types=["Image", "Table"],
                    # infer_table_structure=True,
                    include_metadata=True,
                    include_original_elements=True,
                    flatten_metadata=True,
                    chunk_multipage_sections=True,
                    chunk_combine_text_under_n_chars=500,
                    chunk_new_after_n_chars=1500,
                    chunk_max_characters=2000
                )
                # elements = partition_pdf(
                #     filename=file_path,
                #     mode="elements",
                #     strategy="hi_res",
                #     infer_table_structure=True,
                #     extract_images_in_pdf=True,
                #     extract_image_block_types=["Image", "Table"],
                #     extract_image_block_to_payload=False,
                #     extract_image_block_output_dir=str(file_specific_dir),
                #     include_metadata=True,
                # )
                
                if not chunks:
                    self.logger.warning(f"No elements extracted from PDF {file_path}")
                    return [], []
                
                logger.info(f"Extracted {len(chunks)} elements from PDF {file_path}")
                for element in chunks:
                    if hasattr(element.metadata, 'text_as_html') and element.metadata.text_as_html:
                        print(element.to_dict())
                        break
                # Process and store table HTML in element metadata
                table_count = 0
                for element in chunks:
                    logger.info(f"Processing element of type: {type(element).__name__}")
                    
                    # Check if element is a Table instance
                    if isinstance(element, Table):
                        # Safely access metadata attributes
                        metadata = getattr(element, 'metadata', None)
                        text_as_html = getattr(metadata, 'text_as_html', None)
                        
                        if text_as_html:
                            # Store the HTML in table_html attribute
                            element.table_html = text_as_html
                            self.logger.info(f"Found table {table_count + 1} with content: {element.text[:100]}...")
                            table_count += 1
                            
                            # Log the table HTML for debugging 
                            self.logger.debug(f"Table HTML: {element.table_html[:200]}...")
                
                self.logger.info(f"Found {table_count} table elements in PDF {file_path}")
                # Fixed image path collection
                image_paths = []
                if Path(self.image_output_dir).exists():
                    image_paths = [
                        str(f) for f in Path(self.image_output_dir).glob("**/*.[jJ][pP][gG]")
                    ]
                    self.logger.info(f"Found {len(image_paths)} images in {self.image_output_dir}")
                
                self.logger.info(f"Successfully processed PDF. Found {len(chunks)} elements and {len(image_paths)} images")
                return chunks, image_paths
            
            except Exception as e:
                self.logger.error(f"Error processing PDF {file_path}: {str(e)}", exc_info=True)
                # Re-raise with more context
                raise Exception(f"Failed to process PDF {file_path}: {str(e)}") from e
            
            finally:
                final_memory = process.memory_info().rss
                self.logger.info(f"Completed processing with {final_memory/1024/1024:.1f}MB memory usage")
                gc.collect()
            
        except Exception as e:
            self.logger.error(f"Failed to process {file_path}: {str(e)}")
            raise
    
    async def process_tables(self, elements: List[Any], image_paths: List[str]) -> List[Dict[str, Any]]:
        """Process tables from elements and images."""
        self.logger.info("Starting table processing")
        table_results = []
        try:
            # Get all table elements with their HTML
            table_elements = [
                {
                    'element': e,
                    'html': getattr(e, 'table_html', None) or getattr(e, 'metadata', {}).get('text_as_html', '')
                }
                for e in elements 
                if hasattr(e, 'type') and e.type == 'Table'
            ]
            
            # Get all potential table images
            table_images = [p for p in image_paths if 'table' in p.lower()]
            
            self.logger.info(f"Found {len(table_elements)} table elements and {len(table_images)} table images")
            
            # Process each table image independently
            for image_path in table_images:
                if os.path.exists(image_path):
                    self.logger.info(f"Processing table from {image_path}")
                    # Find closest matching table element based on position or content
                    matching_element = None
                    table_html = ""
                    if table_elements:
                        # Take the next available table element and its HTML
                        element_data = table_elements.pop(0)
                        matching_element = element_data['element']
                        table_html = element_data['html']
                    
                    table_analysis = await self.process_table_image(image_path, table_html)
                    
                    if table_analysis:  # Only add if we got a result
                        table_results.append({
                            'image_path': image_path,
                            'analysis': table_analysis,
                            'html': table_html  # Store the HTML representation
                        })
            
            self.logger.info(f"Completed table processing. Processed {len(table_results)} tables")
            return table_results
        except Exception as e:
            self.logger.error(f"Error processing tables: {str(e)}")
            raise
    
    def chunk_elements(self, elements: List[Any]) -> List[Any]:
        """Chunk elements by title."""
        self.logger.info("Starting element chunking")
        try:
            chunks = chunk_by_title(
                elements,
                multipage_sections=True,
                combine_text_under_n_chars=500,
                new_after_n_chars=1500,
                max_characters=2000
            )
            self.logger.info(f"Successfully chunked elements into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            self.logger.error(f"Error chunking elements: {str(e)}")
            raise
    
    async def process_table_image(self, image_path: str, table_html: str) -> Optional[str]:
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
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                )
                
                self.logger.info(f"Sending table image {image_path} to OpenAI for analysis")
                response = await self.client.ainvoke([message])
                
                if not response or not response.content:
                    self.logger.warning(f"No response content for table image {image_path}")
                    return None
                    
                self.logger.info(f"Successfully processed table image {image_path}")
                return response.content
                
        except Exception as e:
            self.logger.error(f"Error processing table image {image_path}: {str(e)}")
            return None

    def save_to_db(self, filename: str, chunks: List[str], metadata: Dict[str, Any]) -> None:
        """Save processed data to database."""
        session = get_session(self.engine)
        try:
            # Create document record
            doc = Document(
                readable_filename=filename,
                metadata_schema=metadata,
                created_at=datetime.utcnow()
            )
            session.add(doc)
            session.flush()

            # Process segments and chunks
            current_segment = None
            segment_number = 0
            chunk_number = 0

            for chunk_text in chunks:
                # Check if this chunk represents a title by looking at the string representation
                if "Title:" in chunk_text:
                    segment_number += 1
                    current_segment = Segment(
                        document_id=doc.id,
                        segment_number=segment_number,
                        title=chunk_text.replace("Title:", "").strip(),
                        created_at=datetime.utcnow()
                    )
                    session.add(current_segment)
                    session.flush()
                
                chunk_number += 1
                db_chunk = Chunk(
                    document_id=doc.id,
                    segment_id=current_segment.id if current_segment else None,
                    chunk_number=chunk_number,
                    content=chunk_text,
                    created_at=datetime.utcnow()
                )
                session.add(db_chunk)

            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    async def process_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple PDF files."""
        results = []
        for file_path in file_paths:
            try:
                # Process PDF
                elements, image_paths = self.process_pdf(file_path)
                
                # Process tables
                table_results = await self.process_tables(elements, image_paths)
                
                # Save to database
                metadata = {
                    'table_results': table_results,
                    'original_filename': os.path.basename(file_path)
                }
                self.save_to_db(os.path.basename(file_path), elements, metadata)
                
                results.append({
                    'filename': os.path.basename(file_path),
                    'status': 'success',
                    'chunks': len(elements),
                    'tables_processed': len(table_results)
                })
            except Exception as e:
                results.append({
                    'filename': os.path.basename(file_path),
                    'status': 'error',
                    'error': str(e)
                })
        
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
