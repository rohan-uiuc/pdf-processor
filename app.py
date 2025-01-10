import os
import sys
import traceback
import gradio as gr
import asyncio
from typing import List, Dict, Any, Optional, TypedDict
from database import Document, get_session, init_db, delete_document as db_delete_document, Chunk
from processors import PDFProcessor, ChunkProcessor, TableProcessor, SchemaProcessor, MetadataProcessor
from dotenv import load_dotenv
import logging
from pathlib import Path
import json
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from unstructured.staging.base import elements_from_base64_gzipped_json
from sqlalchemy.orm.attributes import flag_modified

from processors.metadata_processor import DocumentMetadataExtraction


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

force_recreate = os.environ.get("FORCE_RECREATE", "false").lower() == "true"
# Initialize database
engine = init_db(force_recreate=force_recreate)  # Force recreate on first run

class TableResult(TypedDict):
    image_path: str
    analysis: str

class ProcessingState:
    def __init__(self):
        self.pdf_processor = PDFProcessor(engine=engine)
        self.chunk_processor = ChunkProcessor(engine=engine)
        self.table_processor = TableProcessor(engine=engine)
        self.schema_processor = SchemaProcessor(engine=engine)
        self.metadata_processor = MetadataProcessor(engine=engine)

state = ProcessingState()

def save_files(files) -> List[str]:
    """Save uploaded files and return their paths."""
    file_paths = []
    upload_dir = Path("uploaded_files")
    upload_dir.mkdir(exist_ok=True)
    print("Upload dir:", upload_dir)
    
    try:
        for file in files:
            if file is None:
                continue
                
            if isinstance(file, (str, Path)):
                # Handle string/Path input
                file_path = Path(file)
                if file_path.suffix.lower() == '.pdf':
                    file_paths.append(str(file_path))
                    
            elif isinstance(file, bytes):
                # Handle bytes data
                file_path = upload_dir / f"uploaded_file_{len(file_paths)}.pdf"
                with open(file_path, 'wb') as f:
                    f.write(file)
                file_paths.append(str(file_path))
                
            else:
                # Handle Gradio file object
                try:
                    if hasattr(file, 'name'):
                        original_name = Path(file.name).name
                        file_path = upload_dir / original_name
                        
                        # Ensure file is PDF
                        if file_path.suffix.lower() != '.pdf':
                            continue
                            
                        # Read and save file content
                        content = file.read() if hasattr(file, 'read') else None
                        if content:
                            with open(file_path, 'wb') as f:
                                f.write(content)
                            file_paths.append(str(file_path))
                except Exception as e:
                    logger.error(f"Error processing file {getattr(file, 'name', 'unknown')}: {str(e)}")
                    continue
        print("File paths:", file_paths)
        return file_paths
        
    except Exception as e:
        logger.error(f"Error in save_files: {str(e)}")
        return []

async def partition_pdf(files) -> tuple[str, list]:
    """First step: Partition PDF and extract images."""
    if not files:
        return "No PDF files uploaded.", []
    
    session = None
    try:
        file_paths = save_files(files)
        if not file_paths:
            return "No valid PDF files found.", []
        
        logger.info(f"Saved files to: {file_paths}")
        output = "Partitioning Results:\n\n"
        
        session = get_session(engine)
        
        for file_path in file_paths:
            doc = None
            try:
                # Check if document exists and its status
                doc = session.query(Document).filter_by(
                    readable_filename=Path(file_path).name
                ).first()
                
                if doc and doc.partition_status == 'completed':
                    output += f"File: {file_path} (Already partitioned)\n"
                    continue

                # Create or update document record
                if not doc:
                    doc = Document(
                        readable_filename=Path(file_path).name,
                        partition_status='processing'
                    )
                    session.add(doc)
                    try:
                        session.flush()
                    except Exception as e:
                        session.rollback()
                        logger.error(f"Error adding document: {str(e)}")
                        raise
                else:
                    doc.partition_status = 'processing'
                
                try:
                    session.commit()
                except Exception as e:
                    session.rollback()
                    logger.error(f"Error committing initial status: {str(e)}")
                    raise

                try:
                    results = state.pdf_processor.process_pdf(file_path)
                except MemoryError:
                    print("Not enough memory to process file")
                    traceback.print_exc()
                    sys.exit(1)
                except Exception as e:
                    print(f"Processing failed: {str(e)}")
                    traceback.print_exc()
                    sys.exit(1)

                elements, image_paths = results
                
                # Update document status
                doc.partition_status = 'completed'
                logger.info(f"Saving Elements: {elements}")
                doc.processing_artifacts = {
                    'elements': elements,
                    'image_paths': image_paths,
                    'processing_stats': {
                        'total_elements': len(elements),
                        'total_images': len(image_paths),
                        'processed_at': datetime.now(timezone.utc).isoformat()
                    }
                }
                
                try:
                    session.commit()
                except Exception as e:
                    session.rollback()
                    logger.error(f"Error committing final status: {str(e)}")
                    raise
                
                output += f"File: {file_path}\n"
                output += f"Elements extracted: {len(elements)}\n"
                output += f"Images extracted: {len(image_paths)}\n"
                output += "-" * 50 + "\n"
                
            except Exception as e:
                if doc:
                    doc.partition_status = 'failed'
                    doc.last_error = str(e)
                    try:
                        session.commit()
                    except:
                        session.rollback()
                logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
                output += f"Error processing {file_path}: {str(e)}\n"
                output += "-" * 50 + "\n"
                continue
        
        # Get updated status data
        docs = session.query(Document).order_by(Document.created_at.desc()).all()
        status_data = [[
            doc.readable_filename,
            doc.partition_status,
            doc.chunk_status,
            doc.table_status,
            doc.db_save_status,
            doc.last_error or ""
        ] for doc in docs]
        
        return output, status_data
        
    except Exception as e:
        logger.error(f"Error in partition_pdf: {str(e)}", exc_info=True)
        return f"Error during partitioning: {str(e)}", []
    finally:
        if session:
            session.close()

async def chunk_content() -> tuple[str, list]:
    """Second step: Chunk the content."""
    session = get_session(engine)
    try:
        output = "Chunking Results:\n\n"
        
        # Get all documents that have been partitioned but not chunked
        docs = session.query(Document).filter(
            Document.partition_status == 'completed',
            Document.chunk_status.in_(['pending', 'failed'])
        ).all()
        
        if not docs:
            return "No documents ready for chunking.", []
        
        for doc in docs:
            try:
                doc.chunk_status = 'processing'
                session.commit()
                
                artifacts = doc.processing_artifacts or {}
                elements_dicts = artifacts['elements']
                logger.info(f"Processing document: {doc.readable_filename}")
                logger.info(f"Number of elements to chunk: {len(elements_dicts)}")
                
                # Process chunks with document ID
                structured_chunks = state.chunk_processor.chunk_elements(elements_dicts, doc_id=doc.id)
                logger.info(f"Number of chunks created: {len(structured_chunks)}")
                
                # Update document with chunk results
                artifacts['chunks'] = structured_chunks
                artifacts['processing_stats']['total_chunks'] = len(structured_chunks)
                doc.processing_artifacts = artifacts
                
                # Save chunks to database
                import json
                from unstructured.staging.base import elements_from_base64_gzipped_json, elements_to_json
                
                for chunk_idx, chunk_data in enumerate(structured_chunks, 1):
                    try:
                        # Extract the actual content from the JSON structure
                        content_json = json.loads(chunk_data['content'])
                        
                        # Get the text content
                        text_content = ""
                        if isinstance(content_json, list) and len(content_json) > 0:
                            text_content = content_json[0].get('text', '')
                            logger.debug(f"Chunk {chunk_idx} text length: {len(text_content)}")
                        
                        # Get metadata and handle orig_elements
                        metadata = {}
                        table_image_paths = []
                        
                        if isinstance(content_json, list) and len(content_json) > 0:
                            chunk_metadata = content_json[0].get('metadata', {})
                            orig_elements_raw = chunk_metadata.get('orig_elements')
                            
                            if orig_elements_raw:
                                try:
                                    # Convert from base64 gzipped JSON if needed
                                    orig_elements = elements_from_base64_gzipped_json(orig_elements_raw)
                                    
                                    # Process each unstructured element
                                    for element in orig_elements:
                                        try:
                                            # Get element type using the class name
                                            element_type = element.__class__.__name__
                                            logger.debug(f"Processing element of type: {element_type}")
                                            
                                            # Check if element is a table type
                                            if element_type in ['Table', 'TableChunk']:
                                                try:
                                                    # First check if the element has a metadata attribute
                                                    if hasattr(element, 'metadata'):
                                                        # If metadata exists, try to get the image_path
                                                        if hasattr(element.metadata, 'image_path'):
                                                            image_path = element.metadata.image_path
                                                            logger.debug(f"Found image path: {image_path}")
                                                            table_image_paths.append(image_path)
                                                        else:
                                                            logger.debug("No image_path in element metadata")
                                                    else:
                                                        logger.debug("No metadata attribute found on element")
                                                except Exception as e:
                                                    logger.error(f"Error accessing element metadata: {str(e)}")
                                                    
                                        except Exception as e:
                                            logger.error(f"Error processing element: {str(e)}")
                                            raise
                                            
                                except Exception as e:
                                    logger.error(f"Error processing orig_elements: {str(e)}")
                                    raise
                            
                            metadata = {k: v for k, v in chunk_metadata.items() if k != 'orig_elements'}
                            table_html = chunk_metadata.get('text_as_html')
                            # Original elements json
                            orig_elements_json = elements_to_json(orig_elements)
                            # logger.info(f"Orig elements json: {orig_elements_json}")
                            
                            # Filter out None values and convert to strings
                            table_image_paths = [str(path) for path in table_image_paths if path is not None]
                            logger.info(f"Found {len(table_image_paths)} table image paths")
                            logger.info(f"Table image paths: {table_image_paths}")
                        
                        # Create chunk record
                        chunk = Chunk(
                            document_id=doc.id,
                            chunk_number=chunk_data['chunk_number'],
                            chunk_type=chunk_data['type'],
                            content=text_content,
                            table_html=table_html,
                            table_image_paths=table_image_paths,
                            chunk_metadata=metadata, 
                            orig_elements=orig_elements_json, 
                            created_at=datetime.now(timezone.utc)
                        )
                        session.add(chunk)
                        
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk_idx} for document {doc.readable_filename}: {str(e)}")
                        raise
                
                session.commit()
                doc.chunk_status = 'completed'
                session.commit()
                logger.info(f"Successfully processed all chunks for document: {doc.readable_filename}")
                
                output += f"File: {doc.readable_filename}\n"
                output += f"Chunks created: {len(structured_chunks)}\n"
                output += "-" * 50 + "\n"
                
            except Exception as e:
                session.rollback()  # Explicitly rollback on error
                doc.chunk_status = 'failed'
                doc.last_error = str(e)
                session.commit()
                logger.error(f"Error chunking {doc.readable_filename}: {str(e)}", exc_info=True)
                output += f"Error chunking {doc.readable_filename}: {str(e)}\n"
                output += "-" * 50 + "\n"
                continue
        
        # Get updated status data
        docs = session.query(Document).order_by(Document.created_at.desc()).all()
        status_data = [[
            doc.readable_filename,
            doc.partition_status,
            doc.chunk_status,
            doc.table_status,
            doc.db_save_status,
            doc.last_error or ""
        ] for doc in docs]
        
        return output, status_data
        
    except Exception as e:
        logger.error(f"Error in chunk_content: {str(e)}", exc_info=True)
        return f"Error during chunking: {str(e)}", []
    finally:
        if session:
            session.close()

async def process_tables() -> tuple[str, list]:
    """Third step: Process tables from extracted images."""
    session = get_session(engine)
    try:
        output = "Table Processing Results:\n\n"
        
        # Process tables - no need to pass elements and image_paths anymore
        table_results = await state.table_processor.process_tables()
        
        if not table_results:
            return "No tables were processed.", get_document_status()
            
        # Format output message
        for result in table_results:
            if 'error' in result:
                output += f"Error processing chunk {result['chunk_id']}: {result['error']}\n"
            else:
                output += f"Chunk {result['chunk_id']}: Processed {result.get('processed_images', 0)} images\n"
            output += "-" * 50 + "\n"
        
        # Get updated status data
        status_data = get_document_status()
        
        return output, status_data
        
    except Exception as e:
        logger.error(f"Error in process_tables: {str(e)}", exc_info=True)
        return f"Error processing tables: {str(e)}", []
    finally:
        if session:
            session.close()

async def extract_schema() -> tuple[str, list]:
    """Fourth step: Extract and define schema for documents."""
    session = get_session(engine)
    try:
        output = "Schema Extraction Results:\n\n"
        
        # Get documents ready for schema extraction
        docs = session.query(Document).filter(
            Document.table_status == 'completed',
            Document.db_save_status == 'pending'
        ).all()
        
        if not docs:
            return "No documents ready for schema extraction.", []
        
        for doc in docs:
            try:
                # fetch related chunks here
                chunks = session.query(Chunk).filter(Chunk.document_id == doc.id).all()

                metadata = doc.processing_artifacts

                # Define schema using trustcall
                schema = await state.schema_processor.define_schema(chunks)
                logger.info(f"EXTRACTED SCHEMA: {schema}")
                
                # Update document with schema
                metadata['schema'] = schema[0]
                
                doc.processing_artifacts = metadata
                flag_modified(doc, "processing_artifacts")  # Mark the field as modified
                doc.db_save_status = 'processing'
                session.commit()
                
                output += f"File: {doc.readable_filename}\n"
                output += f"Schema extracted successfully\n"
                output += "-" * 50 + "\n"
                
            except Exception as e:
                doc.db_save_status = 'failed'
                doc.last_error = str(e)
                session.commit()
                output += f"Error extracting schema for {doc.readable_filename}: {str(e)}\n"
                continue
        
        # Get updated status data
        docs = session.query(Document).order_by(Document.created_at.desc()).all()
        status_data = [[
            doc.readable_filename,
            doc.partition_status,
            doc.chunk_status,
            doc.table_status,
            doc.db_save_status,
            doc.last_error or ""
        ] for doc in docs]
        
        return output, status_data
    except Exception as e:
        logger.error(f"Error in extract_schema: {str(e)}", exc_info=True)
        return f"Error extracting schema: {str(e)}", []
    finally:
        session.close()

async def extract_metadata() -> tuple[str, list]:
    """Fifth step: Extract metadata based on schema."""
    session = get_session(engine)
    try:
        output = "Metadata Extraction Results:\n\n"
        
        # Get documents ready for metadata extraction
        docs = session.query(Document).filter(
            Document.db_save_status == 'processing'
        ).all()
        
        if not docs:
            return "No documents ready for metadata extraction.", []
        
        for doc in docs:
            try:
                # fetch related chunks here
                chunks = session.query(Chunk).filter(Chunk.document_id == doc.id).all()

                metadata = doc.processing_artifacts
                schema = metadata.get('schema')
                
                if not schema:
                    continue

                logger.info(f"Extracting metadata for document: {doc.id} and {doc.readable_filename}")
                
                # Extract metadata using trustcall
                extracted_metadata: DocumentMetadataExtraction = await state.metadata_processor.extract_metadata(
                    doc.id, schema, chunks
                )
                logger.info(f"EXTRACTED METADATA: {extracted_metadata}")
                
                # Update document with extracted metadata

                metadata['extracted_metadata'] = extracted_metadata[0]["metadata_list"]
                doc.processing_artifacts = metadata
                flag_modified(doc, "processing_artifacts")  # Mark the field as modified
                doc.db_save_status = 'completed'
                session.commit()
                
                output += f"File: {doc.readable_filename}\n"
                output += f"Metadata extracted successfully\n"
                output += f"Metadata: {metadata['extracted_metadata']}\n"
                output += "-" * 50 + "\n"
                
            except Exception as e:
                doc.db_save_status = 'failed'
                doc.last_error = str(e)
                session.commit()
                output += f"Error extracting metadata for {doc.readable_filename}: {str(e)}\n"
                continue
        
        # Get updated status data
        docs = session.query(Document).order_by(Document.created_at.desc()).all()
        status_data = [[
            doc.readable_filename,
            doc.partition_status,
            doc.chunk_status,
            doc.table_status,
            doc.db_save_status,
            doc.last_error or ""
        ] for doc in docs]
        
        return output, status_data
    except Exception as e:
        logger.error(f"Error in extract_metadata: {str(e)}", exc_info=True)
        return f"Error extracting metadata: {str(e)}", []
    finally:
        session.close()

async def delete_document(filename: str) -> tuple[str, list]:
    """Delete a document and its associated data."""
    if not filename:
        return "No document selected for deletion.", []
        
    session = get_session(engine)
    try:
        # Delete from database first
        success, message = db_delete_document(session, filename)
        if not success:
            return message, get_document_status()
            
        # If database deletion was successful, delete files
        try:
            # Delete uploaded PDF if it exists
            upload_dir = Path("uploaded_files")
            file_path = upload_dir / filename
            if file_path.exists():
                file_path.unlink()
                
            # Delete associated images
            image_dir = Path("extracted_images") / Path(filename).stem
            if image_dir.exists():
                import shutil
                shutil.rmtree(image_dir)
        except Exception as e:
            logger.warning(f"Error cleaning up files for {filename}: {str(e)}")
            # Don't fail if file cleanup fails
            
        return f"Successfully deleted document: {filename}", get_document_status()
        
    except Exception as e:
        logger.error(f"Error in delete_document: {str(e)}", exc_info=True)
        return f"Error deleting document: {str(e)}", get_document_status()
    finally:
        session.close()

def get_document_status():
    session = get_session(engine)
    try:
        docs = session.query(Document).order_by(Document.created_at.desc()).all()
        status_data = []
        for doc in docs:
            status_data.append([
                doc.readable_filename,
                doc.partition_status or "pending",
                doc.chunk_status or "pending",
                doc.table_status or "pending",
                doc.db_save_status or "pending",
                doc.last_error or "",
                "🗑️ Delete"  # Moved delete button to the end
            ])
        return status_data
    finally:
        session.close()

def handle_delete_click(evt: gr.SelectData, status_table) -> tuple[str, list]:
    """Handle click events on the status table for document deletion."""
    try:
        row_index = evt.index[0]
        col_index = evt.index[1]
        
        if col_index == 6:  # Last column is now delete
            filename = status_table[row_index][0]  # Filename is still first column
            return asyncio.run(delete_document(filename))
    except Exception as e:
        logger.error(f"Error in delete handler: {str(e)}")
        return f"Error deleting document: {str(e)}", get_document_status()
    return None, get_document_status()

def create_gradio_interface():
    """Create the Gradio interface for the PDF processor."""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 📄 PDF Document Processor
            Upload PDF documents and process them through our pipeline.
            """
        )
        
        # Main processing section
        with gr.Column():
            file_input = gr.File(
                label="Upload PDF Document(s)",
                file_count="multiple",
                file_types=[".pdf"],
                elem_classes="file-input"
            )
            
            with gr.Row():
                partition_btn = gr.Button("1. Partition PDF", variant="primary")
                chunk_btn = gr.Button("2. Chunk Content", variant="primary")
                table_btn = gr.Button("3. Process Tables", variant="primary")
                schema_btn = gr.Button("4. Extract Schema", variant="primary")
                metadata_btn = gr.Button("5. Extract Metadata", variant="primary")
            
            output_text = gr.Textbox(
                label="Processing Output",
                lines=10,
                show_label=True
            )
        
        # Status section
        gr.Markdown("---")  # Divider
        with gr.Column():
            with gr.Row():
                gr.Markdown("### 📊 Document Status")
                refresh_btn = gr.Button("🔄 Refresh", variant="secondary", scale=0)
            
            # Updated status table with fixed column headers
            status_table = gr.Dataframe(
                headers=[
                    "Document",
                    "Partition Status",
                    "Chunk Status",
                    "Table Status",
                    "DB Status",
                    "Last Error",
                    "Actions"  # Moved to end
                ],
                datatype=["str", "str", "str", "str", "str", "str", "str"],
                value=get_document_status,
                interactive=True,
                wrap=True
            )
        
        # Connect buttons to their respective functions
        partition_btn.click(
            fn=partition_pdf,
            inputs=[file_input],
            outputs=[output_text, status_table]
        )
        chunk_btn.click(
            fn=chunk_content,
            inputs=[],
            outputs=[output_text, status_table]
        )
        table_btn.click(
            fn=process_tables,
            inputs=[],
            outputs=[output_text, status_table]
        )
        schema_btn.click(
            fn=extract_schema,
            inputs=[],
            outputs=[output_text, status_table]
        )
        metadata_btn.click(
            fn=extract_metadata,
            inputs=[],
            outputs=[output_text, status_table]
        )
        refresh_btn.click(
            fn=get_document_status,
            inputs=[],
            outputs=[status_table]
        )
        
        # Handle delete button clicks in table
        status_table.select(
            fn=handle_delete_click,
            inputs=[status_table],
            outputs=[output_text, status_table]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
