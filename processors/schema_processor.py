from typing import Dict, Any
import logging
import os
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import List

from utils.logging_config import setup_detailed_logging

logger = setup_detailed_logging()

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

class SchemaProcessor:
    def __init__(self, engine=None):
        self.logger = logger
        self.engine = engine
        self.client = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4")

    async def define_schema(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Define document schema using trustcall."""
        try:
            # Create schema extractor if not exists
            if not hasattr(self, "schema_extractor"):
                from trustcall import create_extractor

                self.schema_extractor = create_extractor(
                    self.client,
                    tools=[DocumentSchemaDefinition],
                    tool_choice="DocumentSchemaDefinition",
                    enable_inserts=True,
                )

            # Analyze document content to determine schema
            chunks = metadata.get("elements", [])
            chunk_texts = [
                chunk["text"] for chunk in chunks if chunk["type"] != "Title"
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

            logger.info(f"Schema definition prompt: {prompt}")

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
