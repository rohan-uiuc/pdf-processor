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

# class TableCell(BaseModel):
#     value: str = Field(description="Cell content")
#     row_index: int = Field(description="0-based row index")
#     col_index: int = Field(description="0-based column index")
#     is_header: bool = Field(description="Whether this cell is a header")
#     is_merged: bool = Field(description="Whether this cell is part of a merged group")
#     parent_section: Optional[str] = Field(description="Parent section/category this cell belongs to")
#     data_type: str = Field(description="Type of data in cell (text, number, etc.)")
    
# class TableSection(BaseModel):
#     name: str = Field(description="Section name (e.g., 'POWER', 'TRANSMISSION')")
#     start_row: int = Field(description="Starting row of section")
#     end_row: int = Field(description="Ending row of section")
#     properties: List[str] = Field(description="List of property names in this section")
    
# class EnhancedTableStructure(BaseModel):
#     sections: List[TableSection] = Field(description="Table sections/categories")
#     cell_matrix: List[List[TableCell]] = Field(description="2D matrix of all cells")
#     primary_headers: List[str] = Field(description="Top-level headers")
#     # sub_headers: Dict[str, List[str]] = Field(description="Nested header relationships")
#     section_relationships: Dict[str, List[str]] = Field(description="How sections relate to headers")
#     data_hierarchy: Dict[str, Any] = Field(description="Tree structure of data relationships")

class TableValue(BaseModel):
    value: str = Field(description="The actual value for this field")

class TableData(BaseModel):
    """
    A generic nested structure where:
    - First level is the primary column values (could be model numbers, years, categories, etc.)
    - Second level is sections/groups in the table
    - Third level is the actual field-value pairs
    """
    data: Dict[str, Dict[str, Dict[str, str]]] = Field(
        description="Nested structure: primary_key -> section -> {field: value}"
    )

    class Config:
        extra = "allow"
        json_schema_extra = {
            "example": {
                "data": {
                    "Model A": {  # Could be any primary key (model, year, category)
                        "Section 1": {
                            "Field 1": "Value 1",
                            "Field 2": "Value 2"
                        },
                        "Section 2": {
                            "Field 3": "Value 3"
                        }
                    },
                    "Model B": {
                        "Section 1": {
                            "Field 1": "Value 4",
                            "Field 2": "Value 5"
                        }
                    }
                }
            }
        }

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

    async def process_table_with_vision(self, image_path: str) -> Optional[Dict]:
        """Process table using image with GPT-4V."""
        try:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode()

                prompt = """
You are an expert at analyzing tables and extracting structured data into clean, hierarchical formats.
Your task is to convert the table into a nested structure that preserves all values while being easy to query.

Output Format:
{
    "data": {
        "<primary_key>": {            # This is the column header/identifier (like model number, year, etc.)
            "<section_name>": {        # Table sections or categories (like POWER, TRANSMISSION)
                "<field_name>": "<value>"  # Actual field-value pairs
            }
        }
    }
}

Important Instructions:
1. PRIMARY KEY IDENTIFICATION
   - Identify the main column headers that serve as primary keys
   - These are usually model numbers, years, or categories that differentiate columns

2. SECTION IDENTIFICATION
   - Identify distinct table sections (usually with different background colors or bold headers)
   - Each section should group related fields together

3. VALUE MAPPING
   - For each primary key (column) and section:
     * Map every field to its exact value
     * Keep text exactly as shown
     * Include units where present
     * Preserve all numerical values and formats

4. MERGED CELL HANDLING
   - If a cell is merged across columns:
     * Repeat the value for each applicable primary key
   - If a cell is merged across rows:
     * Include it in each relevant field-value pair

Example Output:
{
    "data": {
        "8320RT": {
            "POWER": {
                "Rated PTO power hp (SAE) at 2,100 engine rpm": "264 hp (196 kW)",
                "Rated Engine power PS (hp ISO) at 2100 engine rpm (97/68/EC)": "320 hp (235 kW)",
                "Intelligent Power Management (Available)": "35 additional engine horsepower at 2,100 rpm rated speed",
                "PTO torque rise": "40%"
            },
            "TRANSMISSION": {
                "Type": "Standard (42 km/h at 1,550 ECO erpm)"
            }
        },
        "8345RT": {
            "POWER": {
                "Rated PTO power hp (SAE) at 2,100 engine rpm": "286 hp (213 kW)",
                "Rated Engine power PS (hp ISO) at 2100 engine rpm (97/68/EC)": "345 hp (254 kW)",
                "Intelligent Power Management (Available)": "35 additional engine horsepower at 2,100 rpm rated speed",
                "PTO torque rise": "40%"
            }
        }
    }
}

Critical Requirements:
- Preserve exact text and values
- Include all fields and values
- Maintain section grouping
- Handle merged cells by repeating values
- Keep numerical values and units intact
- Don't skip any rows or columns
- Don't summarize or modify values
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
                        return TableData(**result["responses"][0].model_dump(exclude_none=True)).model_dump()
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
