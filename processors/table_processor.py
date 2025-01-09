import base64
from typing import List, Dict, Any, Optional, Union
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from sqlalchemy.orm import Session
from database import Chunk, Document
import os
from datetime import datetime, timezone
from trustcall import create_extractor

from utils.logging_config import setup_detailed_logging

logger = setup_detailed_logging()

class MergedCell(BaseModel):
    start_row: int = Field(description="Starting row index (0-based)")
    end_row: int = Field(description="Ending row index (0-based)")
    start_col: int = Field(description="Starting column index (0-based)")
    end_col: int = Field(description="Ending column index (0-based)")
    value: str = Field(description="Content of the merged cell")

class TableStructure(BaseModel):
    merged_cells: List[MergedCell] = Field(default_factory=list, description="List of merged cells in the table")
    header_rows: int = Field(default=1, description="Number of header rows in the table")
    header_hierarchy: Dict[str, List[str]] = Field(default_factory=dict, description="Hierarchical structure of headers if present")
    total_rows: int = Field(description="Total number of rows including headers")
    total_cols: int = Field(description="Total number of columns")
    column_spans: List[Dict[str, Any]] = Field(default_factory=list, description="Column span information")
    row_spans: List[Dict[str, Any]] = Field(default_factory=list, description="Row span information")

class TableData(BaseModel):
    headers: List[Union[str, List[str]]] = Field(description="Array of column headers, can be nested for multi-level headers")
    rows: List[Dict[str, Any]] = Field(default_factory=list, description="Array of row objects with column values")
    structure: TableStructure = Field(description="Detailed table structure information")
    raw_data: List[List[Any]] = Field(description="Raw table data as a 2D array, preserving exact cell positions")

    class Config:
        extra = "allow"  # Allow extra fields in the response

class TableProcessor:
    def __init__(self, engine=None):
        self.logger = logger
        self.engine = engine
        self.client = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), 
            model="gpt-4o",
            temperature=0
        )
        # Initialize Trustcall extractor with our TableData schema
        self.extractor = create_extractor(
            self.client,
            tools=[TableData],
            tool_choice="TableData"
        )

    async def process_table_with_vision(self, image_path: str, table_html: str) -> Optional[TableData]:
        """Process table using image with GPT-4V."""
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode()

                prompt = """
                You are an expert at analyzing tables and extracting structured data.
                Extract ALL structural details from the table image, including headers, rows, merged cells, and hierarchies.

Instructions:
1. Headers (including multi-level headers):
   - Extract all column headers
   - If headers are nested, represent them as arrays
   - Keep header text exactly as shown
   - Identify merged header cells and their spans

2. Rows and Cells:
   - Extract each row as a dictionary with column headers as keys
   - Use exact header text as keys
   - Keep all cell values in their original format
   - Include empty cells as empty strings
   - Track merged cells across rows and columns

3. Table Structure:
   - Count total rows and columns
   - Note any merged cells with their exact positions and content
   - Record header structure and hierarchy
   - Track all cell spans and merges

4. Raw Data:
   - Create a 2D array representing the table exactly as shown
   - Preserve all empty cells
   - Keep original text formatting
   - Maintain merged cell positions

Example structure for a complex table:
{
    "headers": [
        ["Category", "Specifications", "Specifications", "Specifications"],
        ["", "Model A", "Model B", "Model C"]
    ],
    "rows": [
        {
            "Category": "Engine",
            "Model A": "2.0L",
            "Model B": "2.5L",
            "Model C": "3.0L"
        },
        {
            "Category": "Performance",
            "Model A": "200hp",
            "Model B": "250hp",
            "Model C": "300hp"
        }
    ],
    "structure": {
        "total_rows": 4,
        "total_cols": 4,
        "header_rows": 2,
        "merged_cells": [
            {
                "start_row": 0,
                "end_row": 0,
                "start_col": 1,
                "end_col": 3,
                "value": "Specifications"
            },
            {
                "start_row": 2,
                "end_row": 3,
                "start_col": 0,
                "end_col": 0,
                "value": "Performance Data"
            }
        ],
        "header_hierarchy": {
            "Specifications": ["Model A", "Model B", "Model C"]
        },
        "column_spans": [
            {"row": 0, "col_start": 1, "col_end": 3, "value": "Specifications"}
        ],
        "row_spans": [
            {"col": 0, "row_start": 2, "row_end": 3, "value": "Performance Data"}
        ]
    },
    "raw_data": [
        ["Category", "Specifications", "Specifications", "Specifications"],
        ["", "Model A", "Model B", "Model C"],
        ["Performance Data", "200hp", "250hp", "300hp"],
        ["", "Fast", "Faster", "Fastest"]
    ]
}

Important:
- Capture ALL merged cells with their exact positions and content
- Track both horizontal (column) and vertical (row) spans
- Preserve the hierarchy of headers
- Keep all text exactly as shown in the table
- Include empty cells in both rows and raw_data
- Record the total number of header rows accurately"""

                # Create the message with image
                message = {
                    "messages": [
                        {
                            "role": "system",
                            "content": prompt
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Extract the table data from this image, preserving all structural information and exact text values."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    }
                                }
                            ]
                        }
                    ]
                }

                self.logger.info(f"Processing table with image {image_path}")
                try:
                    # Use Trustcall to extract table data
                    result = await self.extractor.ainvoke(message)
                    
                    # Get the first response and convert to dict
                    if result and "responses" in result and len(result["responses"]) > 0:
                        return TableData(**result["responses"][0].model_dump(exclude_none=True))
                    return None

                except Exception as e:
                    self.logger.error(f"Error extracting table data: {str(e)}")
                    return None

        except Exception as e:
            self.logger.error(f"Error processing table with vision for {image_path}: {str(e)}")
            return None

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

                                structured_data = await self.process_table_with_vision(
                                    image_path,
                                    ""  # Removed table_html parameter
                                )

                                if structured_data:
                                    chunk_results.append(structured_data)

                            if chunk_results:
                                # Store all results in the chunk's table_data
                                chunk.table_data = {
                                    'processed_at': datetime.now(timezone.utc).isoformat(),
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
