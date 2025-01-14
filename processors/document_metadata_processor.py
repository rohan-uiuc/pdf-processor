import logging
import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from trustcall import create_extractor
from sqlalchemy.orm import Session

from database import Chunk, Document, DocumentMetadata as DBDocumentMetadata
from utils.logging_config import setup_detailed_logging
from sqlalchemy.orm.attributes import flag_modified

logger = setup_detailed_logging()


class DocumentMetadata(BaseModel):
    """
    A generic nested structure for hierarchical metadata where:
    - First level represents the primary/parent entities
    - Second level represents logical groupings/categories
    - Third+ levels represent nested attributes and their values
    The structure is flexible and can accommodate any number of nesting levels.
    """

    data: Dict[str, Dict[str, Any]] = Field(
        description="Nested structure: parent_entity -> category -> nested_attributes"
    )

    class Config:
        extra = "allow"
        json_schema_extra = {
            "example": {
                "data": {
                    "parent_entity_1": {  # Top-level entity
                        "category_1": {  # Logical grouping
                            "attribute_1": "value_1",
                            "nested_group": {  # Can have deeper nesting
                                "sub_attribute": "value_2"
                            },
                        },
                        "category_2": {"attribute_2": "value_3"},
                    }
                }
            }
        }


class DocumentMetadataProcessor:
    def __init__(self, engine=None):
        self.logger = logger
        self.engine = engine
        self.client = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o", temperature=0
        )
        self.extractor = create_extractor(
            self.client, tools=[DocumentMetadata], tool_choice="DocumentMetadata"
        )

    async def process_documents(self) -> List[Dict[str, Any]]:
        """Process documents to extract structured metadata."""
        self.logger.info("Starting document metadata extraction")

        if not self.engine:
            raise ValueError("Database engine not initialized")

        session = Session(self.engine)
        try:
            # Get all documents with completed chunking but pending metadata
            docs = (
                session.query(Document)
                .filter(
                    Document.chunk_status == "completed",
                    Document.metadata_status.in_(["pending", "failed"]),
                )
                .all()
            )

            results = []
            for doc in docs:
                try:
                    # Get all chunks for this document
                    chunks = (
                        session.query(Chunk).filter(Chunk.document_id == doc.id).all()
                    )

                    if not chunks:
                        self.logger.info(
                            f"No chunks found for document {doc.readable_filename}"
                        )
                        continue

                    self.logger.info(
                        f"Processing {len(chunks)} chunks for document {doc.readable_filename}"
                    )

                    # Separate table chunks and content chunks
                    table_chunks = [
                        chunk
                        for chunk in chunks
                        if chunk.chunk_type in ["Table", "TableChunk"]
                    ]
                    content_chunks = [
                        chunk
                        for chunk in chunks
                        if chunk.chunk_type not in ["Table", "TableChunk"]
                    ]

                    self.logger.info(
                        f"Processing {len(table_chunks)} table chunks and {len(content_chunks)} content chunks"
                    )
                    # Process content chunks in batches
                    content_texts = [chunk.content for chunk in content_chunks]
                    batch_size = 10

                    existing_metadata = None

                    # Create combined batch processing
                    chunk_batches = (
                        # Process each table chunk individually
                        [
                            [
                                (
                                    json.dumps(chunk.table_data)
                                    if chunk.table_data
                                    else chunk.table_html
                                )
                            ]
                            for chunk in table_chunks
                        ]
                        +
                        # Process content chunks in batches
                        [
                            content_texts[i : i + batch_size]
                            for i in range(0, len(content_texts), batch_size)
                        ]
                    )

                    for batch_idx, batch in enumerate(chunk_batches):
                        if not any(batch):  # Skip empty batches
                            continue

                        prompt = """You are an expert at analyzing documents and extracting structured metadata into clean, hierarchical formats.

                        Your task is to extract and organize all information into a nested structure that preserves relationships and values.

                        Output Format:
                        {
                            "data": {
                                "<parent_entity>": {            # Primary/top-level entity identifier
                                    "<category>": {             # Logical grouping of related information
                                        "<attribute>": <value>   # Attribute-value pairs, can be nested further
                                    }
                                }
                            }
                        }

                        Important Instructions:
                        1. PARENT ENTITY IDENTIFICATION
                          - Identify the main top-level entities that serve as primary identifiers
                          - These group related information together at the highest level

                        2. CATEGORY IDENTIFICATION
                          - Identify distinct logical groups of information
                          - Each category should group related attributes together

                        3. VALUE MAPPING
                          - For each parent entity and category:
                            * Map every attribute to its exact value
                            * Keep text exactly as shown
                            * Preserve all formatting and units
                            * Maintain relationships between values

                        4. DATA QUALITY
                          - Preserve exact text and values
                          - Don't modify or summarize values
                          - Maintain all relationships and hierarchies
                          - Don't skip any information

                        Remember:
                        - Focus on preserving relationships and hierarchies
                        - Be thorough in extracting all relevant data
                        - Keep the structure consistent across all entries
                        - Allow for flexible nesting where needed

                        Critical Requirements:
                        - Preserve exact text and values
                        - Include all fields and values
                        - Maintain hierarchical relationships
                        - Keep formatting and units intact
                        - Don't skip any information
                        - Don't summarize or modify values
                        """

                        try:

                            result = await self.extractor.ainvoke(
                                {
                                    "messages": [
                                        {"role": "system", "content": prompt},
                                        {
                                            "role": "user",
                                            "content": "Extract structured metadata from this content, preserving all technical specifications and relationships:\n\n"
                                            + "\n\n".join(batch),
                                        },
                                    ],
                                    "existing": (
                                        {"DocumentMetadata": existing_metadata}
                                        if existing_metadata
                                        else None
                                    ),
                                }
                            )

                            if result and "responses" in result and result["responses"]:
                                metadata = DocumentMetadata(
                                    **result["responses"][0].model_dump()
                                )

                                #     # Determine source chunk
                                #     source_chunk = None
                                #     if batch_idx < len(table_chunks):
                                #         source_chunk = table_chunks[batch_idx]
                                #     else:
                                #         content_batch_idx = batch_idx - len(table_chunks)
                                #         start_idx = content_batch_idx * batch_size
                                #         if start_idx < len(content_chunks):
                                #             source_chunk = content_chunks[start_idx]

                                # if source_chunk and metadata.data:
                                # Store parent entities with their full nested structure

                                # Keep track of the full metadata for context in next batch
                                existing_metadata = metadata.model_dump()

                        except Exception as e:
                            self.logger.error(f"Error processing batch: {str(e)}")
                            continue

                    for parent_entity, entity_data in metadata.data.items():
                        metadata_entry = DBDocumentMetadata(
                            document_id=doc.id,
                            # chunk_id=source_chunk.id,
                            field_name=parent_entity,  # Store the parent entity as field_name
                            field_value=entity_data,  # Store entire nested structure as JSON
                            confidence_score=90,
                            extraction_method="gpt-4o",
                            created_at=datetime.now(timezone.utc),
                        )
                        session.add(metadata_entry)

                    if existing_metadata:
                        # Store the structured format in processing_artifacts
                        doc.processing_artifacts["metadata"] = {
                            "processed_at": datetime.now(timezone.utc).isoformat(),
                            "results": existing_metadata,
                        }
                        doc.metadata_status = "completed"
                        flag_modified(doc, "processing_artifacts")  # Mark the field as modified
                        session.commit()

                        results.append(
                            {
                                "document_id": doc.id,
                                "processed_chunks": len(chunks),
                                "success": True,
                            }
                        )
                    else:
                        doc.metadata_status = "failed"
                        doc.last_error = "No metadata extracted"
                        session.commit()
                        results.append(
                            {"document_id": doc.id, "error": "No metadata extracted"}
                        )
                    return results

                except Exception as e:
                    self.logger.error(
                        f"Error processing document {doc.readable_filename}: {str(e)}"
                    )
                    doc.metadata_status = "failed"
                    doc.last_error = str(e)
                    session.commit()
                    results.append({"document_id": doc.id, "error": str(e)})

        except Exception as e:
            self.logger.error(f"Error in process_documents: {str(e)}")
            raise
        finally:
            session.close()
