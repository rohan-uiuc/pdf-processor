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

    async def define_schema(self, chunks: Dict[str, Any]) -> List[DocumentSchemaDefinition]:
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
            # sample_content = "\n\n".join(
            #     chunk_texts[:6]
            # )  # Use first 5 chunks as sample

            # prompt = f"""You are an expert in tractor specifications and documentation. Your task is to analyze high quality documents from tractor manuals and brochures to create a comprehensive schema for tractor metadata.
            # Examine the provided text carefully and identify ALL possible metadata fields relevant to tractors, their components, their configurations, performance metrics, compliance information, and more. Your schema should include, but is not limited to, the following categories:

            # 1. Basic Information (e.g., Make, Model, Series)
            # 2. Technical Specifications (e.g., Engine, Transmission, Hydraulics)
            # 3. Dimensions and Capacities
            # 4. Optional Features and Configurations
            # 5. Performance Metrics
            # 6. Compliance and Certifications
            # 7. Components and Configurations
            # 8. Additional Fields

            # For each identified field, provide:
            # 1. A clear, descriptive field name
            # 2. A concise description of what the field represents
            # 3. The expected data type (e.g., string, integer, list, nested object, all pydantic types)

            # Your output should be an object containing JSON array of objects, each representing a field in the schema with name, description and type.

            # Be thorough and creative in identifying relevant fields. Consider all aspects of tractor specifications and features that might be important for users of this schema.
            # Everything that needs to be added/removed/changed is inside the field 'document_metadata_schema'.
            # Do not try to use 'instance' field for jsonpatch as it is an internal field used to track instances of the schema. The main schema is stored in the 'document_metadata_schema' field.
            # Analyze the following text and create a comprehensive schema.
            # Return as a DocumentSchemaDefinition with appropriate field definitions.
            # """

            # Separate table chunks and content chunks
            table_chunks = [chunk for chunk in chunks if chunk.chunk_type in ['Table', 'TableChunk']]
            content_chunks = [chunk for chunk in chunks if chunk.chunk_type not in ['Table', 'TableChunk']]

            # Process content chunks in batches
            content_texts = [chunk.content for chunk in content_chunks]
            batch_size = 5

            # Initialize schema object
            schema_object = DocumentSchemaDefinition(schema_type="", schema_version="", fields=[], description="")

            # Process tables individually and content in batches
            chunk_batches = (
                # Process each table chunk individually
                [[chunk.table_data] for chunk in table_chunks] +
                # Process content chunks in batches
                [content_texts[i:i + batch_size] for i in range(0, len(content_texts), batch_size)]
            )

            for batch in chunk_batches:
                sample_content = "\n\n".join(batch)

                prompt = f"""
                Analyze the content of the given PDF document and generate a structured JSON schema that accurately represents key metadata, concepts, and attributes found within the document. The schema should reflect the actual content rather than just the structure.

                ### **Instructions:**
                1. **Identify the Core Topics & Themes**
                - Determine the main subject of the document (e.g., technical specifications, research findings, product information).
                - Identify the most important data points that should be structured into the schema.

                2. **Extract Key Concepts as Schema Fields**
                - Define key metadata fields based on the document's core subject matter.
                - If the document describes a **product**, focus on specifications, performance, and features.
                - If the document is a **research paper**, focus on authors, methodology, findings, and references.

                3. **Define Data Types & Relationships**
                - Use appropriate data types such as `string`, `integer`, `boolean`, `array`, `object`.
                - Structure technical information with **nested objects** where applicable.

                4. **Ensure Scalability**
                - The schema should be adaptable to similar documents in the same category.
                - Optional fields should be included to account for missing information.

                5. **Table Data**
                - If the document contains tables, extract the table data and use it to define the schema.
                - All the fields, rows, columns, headers, merged cells, and hierarchies should be captured.

                Ensure that the schema accurately represents the content of the document. 
                Return the schema as a DocumentSchemaDefinition with appropriate field definitions.

                PDF Content: {sample_content}
                """
                result = await self.schema_extractor.ainvoke(
                    {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a schema definition expert.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "existing": {"DocumentSchemaDefinition": schema_object},
                    }
                )

                logger.info(f"Schema Generation Result: {result}")

                if result and result.get("responses"):
                    schema_object = DocumentSchemaDefinition(**result["responses"][0].model_dump())
                    

                # if not result or not result.get("responses"):
                #     raise ValueError("No schema definition generated")

            #schema_def = result["responses"][0]
            return [schema_object.model_dump()]

        except Exception as e:
            self.logger.error(f"Error defining schema: {str(e)}")
            raise
