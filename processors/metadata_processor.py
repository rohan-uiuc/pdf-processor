import json
from typing import List, Dict, Any, Optional
import logging
import os
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from utils.logging_config import setup_detailed_logging

logger = setup_detailed_logging()

class MetadataValue(BaseModel):
    field_name: str
    value: Any
    confidence_score: int
    source_chunk_id: Optional[int] = None
    extraction_method: str = "gpt-4o"

class MetadataList(BaseModel):
    metadata_values: List[MetadataValue]

class DocumentMetadataExtraction(BaseModel):
    document_id: int
    metadata_list: MetadataList

class MetadataProcessor:
    def __init__(self, engine=None):
        self.logger = logger
        self.engine = engine
        self.client = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

    async def extract_metadata(
        self, doc_id: int, schema: Dict[str, Any], chunks: List[Dict[str, Any]]
    ) -> List[DocumentMetadataExtraction]:
        """Extract metadata from chunks using trustcall based on schema."""
        try:
            # Create metadata extractor if not exists
            if not hasattr(self, "metadata_extractor"):
                from trustcall import create_extractor

                self.metadata_extractor = create_extractor(
                    self.client,
                    tools=[MetadataList],
                    tool_choice="MetadataList",
                    enable_inserts=True,
                )

            # Process chunks in batches to extract metadata
            batch_size = 5
            all_metadata = []
            metadata_list_obj = MetadataList(metadata_values=[])
            doc_object = DocumentMetadataExtraction(document_id=doc_id, metadata_list=metadata_list_obj)
            logger.info(f"Extracting metadata for document object: {doc_object}\n")

            for i in range(0, len(chunks), batch_size):
                logger.info(f"Metadata list object:  {metadata_list_obj}\n\n")
                batch = chunks[i : i + batch_size]

                chunk_texts = [
                    f"[Chunk {chunk.get('element_id', i)}]\n{chunk['text']}"
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
                        "existing": {"MetadataList": metadata_list_obj},
                    }
                )

                logger.info(f"Metadata extraction result: {result}")

                if result and result.get("responses"):
                    #metadata_extraction = result["responses"][0]
                    #all_metadata.extend(metadata_extraction.metadata_values)
                    metadata_list_obj = MetadataList(**result["responses"][0].model_dump())

            doc_object.metadata_list = metadata_list_obj
            logger.info(f"FINAL Extracted metadata: {doc_object}")

            return [doc_object.model_dump()]

        except Exception as e:
            self.logger.error(f"Error extracting metadata: {str(e)}")
            raise
