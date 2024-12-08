import os
import sys
import traceback
import gradio as gr
import asyncio
from typing import List, Dict, Any, Optional, TypedDict
from database import Document, get_session
from processor import PDFProcessor
from dotenv import load_dotenv
import logging
from pathlib import Path
import json
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TableResult(TypedDict):
    image_path: str
    analysis: str

class ProcessingState:
    def __init__(self):
        self.processor = PDFProcessor()

state = ProcessingState()

def save_files(files) -> List[str]:
    """Save uploaded files and return their paths."""
    file_paths = []
    upload_dir = Path("uploaded_files")
    upload_dir.mkdir(exist_ok=True)
    
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
        
        return file_paths
        
    except Exception as e:
        logger.error(f"Error in save_files: {str(e)}")
        return []

async def partition_pdf(files) -> str:
    """First step: Partition PDF and extract images."""
    if not files:
        return "No PDF files uploaded."
    
    session = None
    try:
        file_paths = save_files(files)
        if not file_paths:
            return "No valid PDF files found."
        
        logger.info(f"Saved files to: {file_paths}")
        output = "Partitioning Results:\n\n"
        
        session = get_session(state.processor.engine)
        
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
                        session.flush()  # Test if we can add the document
                    except Exception as e:
                        session.rollback()
                        logger.error(f"Error adding document: {str(e)}")
                        raise
                else:
                    doc.partition_status = 'processing'
                
                try:
                    session.commit()  # Commit the initial status
                except Exception as e:
                    session.rollback()
                    logger.error(f"Error committing initial status: {str(e)}")
                    raise

                try:
                    processor = PDFProcessor()
                    results = processor.process_pdf(file_path)
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
                doc.metadata_schema = {
                    'elements': [str(e) for e in elements],
                    'image_paths': image_paths,
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
        
        return output
        
    except Exception as e:
        logger.error(f"Error in partition_pdf: {str(e)}", exc_info=True)
        return f"Error during partitioning: {str(e)}"
    finally:
        if session:
            session.close()

async def process_tables() -> str:
    """Second step: Process tables from extracted images."""
    session = get_session(state.processor.engine)
    try:
        output = "Table Processing Results:\n\n"
        
        # Get all documents that have been partitioned but not processed for tables
        docs = session.query(Document).filter(
            Document.partition_status == 'completed',
            Document.table_status.in_(['pending', 'failed'])
        ).all()
        
        if not docs:
            return "No documents ready for table processing."
        
        for doc in docs:
            try:
                doc.table_status = 'processing'
                session.commit()
                
                metadata = doc.metadata_schema
                elements = state.processor.load_elements(metadata['elements'])
                image_paths = metadata['image_paths']
                
                table_results = await state.processor.process_tables(elements, image_paths)
                
                # Update document with table results
                metadata['table_results'] = table_results
                doc.metadata_schema = metadata
                doc.table_status = 'completed'
                session.commit()
                
                output += f"File: {doc.readable_filename}\n"
                output += f"Tables processed: {len(table_results)}\n"
                output += "-" * 50 + "\n"
                
            except Exception as e:
                doc.table_status = 'failed'
                doc.last_error = str(e)
                session.commit()
                output += f"Error processing {doc.readable_filename}: {str(e)}\n"
                continue
        
        return output
    except Exception as e:
        logger.error(f"Error in process_tables: {str(e)}", exc_info=True)
        return f"Error processing tables: {str(e)}"
    finally:
        session.close()

async def chunk_content() -> str:
    """Third step: Chunk the content."""
    session = get_session(state.processor.engine)
    try:
        output = "Chunking Results:\n\n"
        
        # Get all documents that have been processed for tables but not chunked
        docs = session.query(Document).filter(
            Document.table_status == 'completed',
            Document.chunk_status.in_(['pending', 'failed'])
        ).all()
        
        if not docs:
            return "No documents ready for chunking. Please process tables first."
        
        for doc in docs:
            try:
                doc.chunk_status = 'processing'
                session.commit()
                
                # Get elements from metadata
                metadata = doc.metadata_schema
                elements = state.processor.load_elements(metadata['elements'])
                
                # Chunk the elements
                chunks = state.processor.chunk_elements(elements)
                
                # Store chunks in metadata
                metadata['chunks'] = [str(chunk) for chunk in chunks]
                doc.metadata_schema = metadata
                doc.chunk_status = 'completed'
                session.commit()
                
                output += f"File: {doc.readable_filename}\n"
                output += f"Total chunks created: {len(chunks)}\n"
                output += "-" * 50 + "\n"
                
            except Exception as e:
                doc.chunk_status = 'failed'
                doc.last_error = str(e)
                session.commit()
                logger.error(f"Error chunking {doc.readable_filename}: {str(e)}")
                output += f"Error processing {doc.readable_filename}: {str(e)}\n"
                continue
        
        return output
        
    except Exception as e:
        logger.error(f"Error in chunk_content: {str(e)}")
        return f"Error chunking content: {str(e)}"
    finally:
        session.close()

async def save_to_database() -> str:
    """Final step: Save everything to database."""
    if not state.current_files:
        return "No files have been processed. Please partition PDF first."
    
    try:
        output = "Saving to database:\n\n"
        for file_path, file_state in state.current_files.items():
            # Skip if content already saved to database
            if 'saved_to_db' in file_state:
                output += f"File: {file_path} (Content already saved to database)\n"
                continue

            metadata = {
                'table_results': file_state['table_results'],
                'image_paths': file_state['image_paths']
            }
            state.processor.save_to_db("processed_document.pdf", file_state['chunks'], metadata)
            
            # Update state
            file_state['saved_to_db'] = True
            state.save_file_state(file_path, file_state)
            
            output += f"File: {file_path}\n"
            output += "Successfully saved to database!\n"
            output += "-" * 50 + "\n"
        
        return output
    except Exception as e:
        logger.error(f"Error in save_to_database: {str(e)}")
        return f"Error saving to database: {str(e)}"

async def get_processing_status() -> str:
    """Get current processing status of all documents."""
    session = get_session(state.processor.engine)
    try:
        docs = session.query(Document).all()
        if not docs:
            return "No documents in the system."
        
        output = "Document Processing Status:\n\n"
        for doc in docs:
            output += f"Document: {doc.readable_filename}\n"
            output += f"Partition Status: {doc.partition_status}\n"
            output += f"Table Status: {doc.table_status}\n"
            output += f"Chunk Status: {doc.chunk_status}\n"
            output += f"DB Save Status: {doc.db_save_status}\n"
            if doc.last_error:
                output += f"Last Error: {doc.last_error}\n"
            output += "-" * 50 + "\n"
        
        return output
    finally:
        session.close()

async def get_pending_documents() -> List[Dict[str, Any]]:
    """Get list of documents with their processing status."""
    session = get_session(state.processor.engine)
    try:
        docs = session.query(Document).all()
        return [
            {
                "id": doc.id,
                "filename": doc.readable_filename,
                "partition_status": doc.partition_status,
                "table_status": doc.table_status,
                "chunk_status": doc.chunk_status,
                "db_save_status": doc.db_save_status,
                "last_error": doc.last_error
            }
            for doc in docs
        ]
    finally:
        session.close()

# Add this new function near the other database functions
async def delete_document(doc_name: str) -> str:
    """Delete a document and its associated data from the database."""
    if not doc_name:
        return "No document selected."
    
    # Extract filename from dropdown text (removes status info in parentheses)
    filename = doc_name.split(" (")[0]
    
    session = get_session(state.processor.engine)
    try:
        doc = session.query(Document).filter_by(readable_filename=filename).first()
        if not doc:
            return f"Document '{filename}' not found in database."
        
        # Delete the document (cascade will handle related records)
        session.delete(doc)
        session.commit()
        return f"Successfully deleted document: {filename}"
    except Exception as e:
        session.rollback()
        logger.error(f"Error deleting document {filename}: {str(e)}")
        return f"Error deleting document: {str(e)}"
    finally:
        session.close()

# Create Gradio interface
with gr.Blocks(title="PDF Processor") as app:
    gr.Markdown("# PDF Document Processor")
    gr.Markdown("""
    Process PDF files step by step:
    1. Partition PDF and extract images
    2. Process tables
    3. Chunk content
    4. Save to database
    
    You can either upload new documents or continue processing existing ones.
    """)
    
    with gr.Row():
        with gr.Column():
            file_input = gr.File(
                file_count="multiple",
                file_types=[".pdf"],
                label="Upload New PDF Files",
                type="binary"
            )
            
        with gr.Column():
            with gr.Row():
                doc_dropdown = gr.Dropdown(
                    label="Or Select Existing Document",
                    choices=[],
                    interactive=True,
                    allow_custom_value=False
                )
                refresh_btn = gr.Button("🔄 Refresh")
            with gr.Row():
                delete_btn = gr.Button("🗑️ Delete Selected Document", variant="secondary")
    
    with gr.Row():
        partition_btn = gr.Button("1. Partition PDF")
        process_tables_btn = gr.Button("2. Process Tables")
        chunk_btn = gr.Button("3. Chunk Content")
        save_btn = gr.Button("4. Save to Database")
        status_btn = gr.Button("Check Processing Status")
    
    with gr.Row():
        output = gr.Textbox(
            label="Processing Results",
            lines=15,
            max_lines=30
        )

    # Function to update dropdown choices
    async def update_dropdown():
        docs = await get_pending_documents()
        choices = []
        for doc in docs:
            status_info = []
            if doc["partition_status"] != "completed":
                status_info.append("needs partitioning")
            elif doc["table_status"] != "completed":
                status_info.append("needs table processing")
            elif doc["chunk_status"] != "completed":
                status_info.append("needs chunking")
            elif doc["db_save_status"] != "completed":
                status_info.append("needs saving")
            
            status_str = f" ({', '.join(status_info)})" if status_info else " (completed)"
            choices.append(f"{doc['filename']}{status_str}")
        
        return gr.Dropdown(choices=choices)

    # Connect refresh button
    refresh_btn.click(
        fn=update_dropdown,
        inputs=[],
        outputs=[doc_dropdown]
    )

    # Update the processing functions to handle both new uploads and existing documents
    async def handle_partition(files, selected_doc):
        if files:
            return await partition_pdf(files)
        elif selected_doc:
            doc_name = selected_doc.split(" (")[0]  # Extract filename from dropdown
            return await partition_pdf([doc_name])
        return "Please either upload new files or select an existing document."

    async def handle_tables(selected_doc):
        return await process_tables()

    async def handle_chunks(selected_doc):
        return await chunk_content()

    async def handle_save(selected_doc):
        return await save_to_database()

    # Connect the buttons with updated handlers
    partition_btn.click(
        fn=handle_partition,
        inputs=[file_input, doc_dropdown],
        outputs=[output]
    )
    
    process_tables_btn.click(
        fn=handle_tables,
        inputs=[doc_dropdown],
        outputs=[output]
    )
    
    chunk_btn.click(
        fn=handle_chunks,
        inputs=[doc_dropdown],
        outputs=[output]
    )
    
    save_btn.click(
        fn=handle_save,
        inputs=[doc_dropdown],
        outputs=[output]
    )
    
    status_btn.click(
        fn=get_processing_status,
        inputs=[],
        outputs=[output]
    )

    # Automatically load the document list when the interface starts
    app.load(
        fn=update_dropdown,
        inputs=[],
        outputs=[doc_dropdown]
    )

    # Add the delete button click handler
    delete_btn.click(
        fn=delete_document,
        inputs=[doc_dropdown],
        outputs=[output]
    )

if __name__ == "__main__":
    app.launch(
        debug=True,
        show_error=True,
        server_name="0.0.0.0",
        server_port=7860
    )
