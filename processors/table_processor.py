import base64
from typing import List, Dict, Any, Optional, Union
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from database import Chunk, Document
import os
from datetime import datetime, timezone
from trustcall import create_extractor

from utils.logging_config import setup_detailed_logging

logger = setup_detailed_logging()

# class MergedCell(BaseModel):
#     start_row: int = Field(description="Starting row index (0-based)")
#     end_row: int = Field(description="Ending row index (0-based)")
#     start_col: int = Field(description="Starting column index (0-based)")
#     end_col: int = Field(description="Ending column index (0-based)")
#     value: str = Field(description="Content of the merged cell")

# class TableStructure(BaseModel):
#     merged_cells: List[MergedCell] = Field(default_factory=list, description="List of merged cells in the table")
#     header_rows: int = Field(default=1, description="Number of header rows in the table")
#     header_hierarchy: Dict[str, List[str]] = Field(default_factory=dict, description="Hierarchical structure of headers if present")
#     total_rows: int = Field(description="Total number of rows including headers")
#     total_cols: int = Field(description="Total number of columns")
#     column_spans: List[Dict[str, Any]] = Field(default_factory=list, description="Column span information")
#     row_spans: List[Dict[str, Any]] = Field(default_factory=list, description="Row span information")

# class TableData(BaseModel):
#     headers: List[Union[str, List[str]]] = Field(description="Array of column headers, should be nested for multi-level headers")
#     rows: List[Dict[str, Any]] = Field(default_factory=list, description="Array of row objects with column values")
#     structure: TableStructure = Field(description="Detailed table structure information")
#     raw_data: List[List[Any]] = Field(description="Raw table data as a 2D array, preserving exact cell positions")

#     class Config:
#         extra = "allow"  # Allow extra fields in the response

class TableCell(BaseModel):
    value: str = Field(description="Cell content")
    row_index: int = Field(description="0-based row index")
    col_index: int = Field(description="0-based column index")
    is_header: bool = Field(description="Whether this cell is a header")
    is_merged: bool = Field(description="Whether this cell is part of a merged group")
    parent_section: Optional[str] = Field(description="Parent section/category this cell belongs to")
    data_type: str = Field(description="Type of data in cell (text, number, etc.)")
    
class TableSection(BaseModel):
    name: str = Field(description="Section name (e.g., 'POWER', 'TRANSMISSION')")
    start_row: int = Field(description="Starting row of section")
    end_row: int = Field(description="Ending row of section")
    properties: List[str] = Field(description="List of property names in this section")
    
class EnhancedTableStructure(BaseModel):
    sections: List[TableSection] = Field(description="Table sections/categories")
    cell_matrix: List[List[TableCell]] = Field(description="2D matrix of all cells")
    primary_headers: List[str] = Field(description="Top-level headers")
    sub_headers: Dict[str, List[str]] = Field(description="Nested header relationships")
    section_relationships: Dict[str, List[str]] = Field(description="How sections relate to headers")
    data_hierarchy: Dict[str, Any] = Field(description="Tree structure of data relationships")

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
            tools=[EnhancedTableStructure],
            tool_choice="EnhancedTableStructure"
        )

    async def process_table_with_vision(self, image_path: str) -> Optional[Dict]:
        """Process table using image with GPT-4V."""
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode()

                prompt = """
                You are an expert at analyzing tables and extracting structured data.
                Extract ALL structural details from the table image, including headers, rows, merged cells, and hierarchies sequentially.
                You must return a valid JSON object that captures ALL structural details. Follow these instructions religiously!
                Make sure to capture all cells, rows, columns, headers, merged cells, and hierarchies, and relationships between them, DO NOT skip any.
Instructions:
1. SECTION IDENTIFICATION
- First identify distinct table sections (groups of related rows)
- Note where each section starts and ends
- Identify the hierarchy of sections

2. HEADER ANALYSIS
- Identify all column headers at each level
- Map relationships between header levels
- Note which headers apply to which sections

3. MERGED CELL PROCESSING
- Identify all merged cells (both horizontal and vertical)
- Determine if merged cells are headers or data
- Map merged cells to their respective sections
- Repeat the values for individual cells that are part of a merged group such that all the values mapping to a particular row and column are accurately captured.

4. DATA STRUCTURE MAPPING
- Create a cell matrix preserving all relationships
- Map each cell to its correct headers and section
- Preserve the parent-child relationships

5. SEMANTIC GROUPING
- Group related data fields
- Identify dependent fields
- Map field relationships within sections

For each cell, provide:
{
    "value": "actual content",
    "row_index": 0,
    "col_index": 0,
    "is_header": false,
    "is_merged": false,
    "parent_section": "section name",
    "data_type": "text/number/etc"
}

For each section, provide:
{
    "name": "section name",
    "start_row": 0,
    "end_row": 5,
    "properties": ["prop1", "prop2"]
}

Important:
- Capture ALL merged cells with their exact positions and content
- Track both horizontal (column) and vertical (row) spans
- Preserve the hierarchy of headers
- Keep all text exactly as shown in the table
- Record the total number of header rows accurately

Here is an example of the output format you should return:
{
    "sections": [
        {
            "name": "POWER",
            "start_row": 0,
            "end_row": 5,
            "properties": [
                "Rated PTO power hp (SAE) at 2,100 engine rpm",
                "Rated Engine power PS (hp ISO) at 2100 engine rpm (97/68/EC)",
                "Max Engine power PS (hp ISO) at 1900 engine rpm (97/68/EC)",
                "Intelligent Power Management (Available)",
                "PTO torque rise",
                "PTO power bulge"
            ]
        }
    ],
"""

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
                        return EnhancedTableStructure(**result["responses"][0].model_dump(exclude_none=True)).model_dump()
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
                                    image_path
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
