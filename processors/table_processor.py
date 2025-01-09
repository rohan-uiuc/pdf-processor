import base64
from typing import List, Dict, Any, Optional
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from sqlalchemy.orm import Session
from database import Chunk, Document
import os
from datetime import datetime

from utils.logging_config import setup_detailed_logging

logger = setup_detailed_logging()

class TableProcessor:
    def __init__(self, engine=None):
        
        self.logger = logger
        self.engine = engine
        self.client = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

    async def process_tables(self) -> List[Dict[str, Any]]:
        """Process tables from database chunks and update with structured data."""
        self.logger.info("Starting table processing")
        
        if not self.engine:
            raise ValueError("Database engine not initialized")

        session = Session(self.engine)
        try:
            # Get all documents with completed chunking but pending table processing
            docs = session.query(Document).filter(
                Document.chunk_status == 'completed',
                Document.table_status.in_(['pending', 'failed'])
            ).all()

            results = []
            for doc in docs:
                try:
                    # Get all table chunks for this document
                    table_chunks = session.query(Chunk).filter(
                        Chunk.document_id == doc.id,
                        Chunk.chunk_type == 'Table' or Chunk.chunk_type == 'TableChunk'
                    ).all()

                    if not table_chunks or len(table_chunks) == 0:
                        self.logger.info(f"No table chunks found for document {doc.readable_filename}")
                        continue

                    self.logger.info(f"Processing {len(table_chunks)} table chunks for document {doc.readable_filename}")

                    for chunk in table_chunks:
                        try:
                            # Validate table image paths
                            if not chunk.table_image_paths or not isinstance(chunk.table_image_paths, list):
                                self.logger.warning(f"No valid image paths found in chunk {chunk.id}")
                                continue

                            # Process each image path for the table
                            chunk_results = []
                            for image_path in chunk.table_image_paths:
                                if not image_path or not os.path.exists(image_path):
                                    self.logger.warning(f"Image path {image_path} not found for chunk {chunk.id}")
                                    continue

                                if chunk.table_html:
                                    structured_data = await self.process_table_with_vision(
                                        image_path,
                                        chunk.table_html
                                    )

                                    if structured_data:
                                        chunk_results.append(structured_data)

                            if chunk_results:
                                # Store all results in the chunk's table_data
                                chunk.table_data = {
                                    'processed_at': datetime.utcnow().isoformat(),
                                    'results': chunk_results
                                }
                                session.commit()

                                results.append({
                                    'chunk_id': chunk.id,
                                    'processed_images': len(chunk_results),
                                    'success': True
                                })
                            else:
                                self.logger.warning(f"No successful results for chunk {chunk.id}")
                                results.append({
                                    'chunk_id': chunk.id,
                                    'processed_images': 0,
                                    'success': False
                                })

                        except Exception as e:
                            self.logger.error(f"Error processing table chunk {chunk.id}: {str(e)}")
                            results.append({
                                'chunk_id': chunk.id,
                                'error': str(e)
                            })
                            continue

                    # Update document status only if all chunks were processed
                    doc.table_status = 'completed'
                    session.commit()

                except Exception as e:
                    self.logger.error(f"Error processing document {doc.readable_filename}: {str(e)}")
                    doc.table_status = 'failed'
                    doc.last_error = str(e)
                    session.commit()

            return results

        except Exception as e:
            self.logger.error(f"Error in process_tables: {str(e)}")
            raise
        finally:
            session.close()

    async def process_table_with_vision(self, image_path: str, table_html: str) -> Optional[Dict]:
        """Process table using both image and HTML with GPT-4V."""
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode()

                prompt = """Analyze this table image and extract structured data.
                The goal is to create a precise JSON representation of the table data.
                
                Guidelines:
                1. Preserve the exact column headers and row structure
                2. Maintain any merged cells or special formatting
                3. Ensure all text values match exactly what appears in the table
                4. Include any relevant metadata about the table structure
                
                Return the data as a JSON object with:
                - headers: array of column headers
                - rows: array of row objects with column values
                - metadata: any additional table structure information
                
                The HTML representation is provided for additional context."""

                message = HumanMessage(
                    content=[
                        {
                            "type": "text",
                            #"text": f"{prompt}\n\nHTML Representation:\n{table_html}"
                            "text": f"{prompt}\n\n"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                )

                self.logger.info(f"Processing table with image {image_path}")
                response = await self.client.ainvoke([message])

                if not response or not response.content:
                    self.logger.warning(f"No response content for table {image_path}")
                    return None

                # Parse the response into structured data
                # The response should be a JSON string that we can directly store
                return response.content

        except Exception as e:
            self.logger.error(f"Error processing table with vision for {image_path}: {str(e)}")
            return None
