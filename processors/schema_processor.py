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
        self.client = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

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

            prompt = f"""You are an expert in tractor specifications and documentation. Your task is to analyze high quality documents from tractor manuals and brochures to create a comprehensive schema for tractor metadata.
            Examine the provided text carefully and identify ALL possible metadata fields relevant to tractors, their components, their configurations, performance metrics, compliance information, and more. Your schema should include, but is not limited to, the following categories:

            1. Basic Information (e.g., Make, Model, Series)
            2. Technical Specifications (e.g., Engine, Transmission, Hydraulics)
            3. Dimensions and Capacities
            4. Optional Features and Configurations
            5. Performance Metrics
            6. Compliance and Certifications
            7. Components and Configurations
            8. Additional Fields

            For each identified field, provide:
            1. A clear, descriptive field name
            2. A concise description of what the field represents
            3. The expected data type (e.g., string, integer, list, nested object, all pydantic types)

            Your output should be an object containing JSON array of objects, each representing a field in the schema with name, description and type.

            Be thorough and creative in identifying relevant fields. Consider all aspects of tractor specifications and features that might be important for users of this schema.
            Everything that needs to be added/removed/changed is inside the field 'document_metadata_schema'.
            Do not try to use 'instance' field for jsonpatch as it is an internal field used to track instances of the schema. The main schema is stored in the 'document_metadata_schema' field.
            Analyze the following text and create a comprehensive schema.
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
