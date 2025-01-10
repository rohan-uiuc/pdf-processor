import json
from typing import Dict, Any, Optional, Union, List
import logging
import os
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from utils.logging_config import setup_detailed_logging

logger = setup_detailed_logging()

class FieldType(BaseModel):
    type: str = Field(description="The type of the field")
    items: Optional[Union[str, 'NestedFieldDefinition']] = Field(description="The items of the field")
    properties: Optional[Dict[str, 'NestedFieldDefinition']] = Field(description="The properties of the field")

class NestedFieldDefinition(BaseModel):
    field_name: str = Field(description="The name of the field")
    description: str = Field(description="The description of the field")
    type: FieldType = Field(description="The type of the field")
    required: bool = Field(description="Whether the field is required")
    # example: Union[str, dict, list] = ""
    nested_fields: Optional[List['NestedFieldDefinition']] = Field(description="The nested fields of the field")

class DocumentSchemaDefinition(BaseModel):
    schema_type: str = Field(description="The type of the schema")
    schema_version: str = Field(description="The version of the schema")
    fields: List[NestedFieldDefinition] = Field(description="The fields of the schema")
    description: str = Field(description="The description of the schema")

# Required for forward references
NestedFieldDefinition.model_rebuild()

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
                [[json.dumps(chunk.table_data) if chunk.table_data else ""] for chunk in table_chunks] +
                # Process content chunks in batches
                [content_texts[i:i + batch_size] for i in range(0, len(content_texts), batch_size)]
            )

            for batch in chunk_batches:
                sample_content = "\n\n".join(batch)

                example_structure = '''
                  {
                    "field_name": "tractor",
                    "type": {
                      "type": "object",
                      "properties": {
                        "engine": {
                          "type": "object",
                          "properties": {...}
                        },
                        "features": {
                          "type": "array",
                          "items": {
                            "type": "object",
                            "properties": {...}
                          }
                        }
                      }
                    }
                  }
                '''

                prompt = f"""
                Analyze the content(chunks) of the given PDF document and generate a structured, hierarchical JSON schema that accurately represents key metadata, concepts, attributes, and their relationships found within the document. The schema should capture both the structure and nested relationships.

                ### **Instructions:**
                1. **Identify Core Entities & Their Relationships**
                - Determine the main entities in the document (e.g., Tractor, Engine, Transmission)
                - Establish relationships between entities (e.g., Tractor has-one Engine, has-many Features)
                - Create nested structures to represent these relationships

                2. **Define Complex Types & Nested Structures**
                - Use nested objects to represent complex entities
                - Define arrays of objects for collections
                - Capture hierarchical relationships in the type definitions
                - Example structure:
                  ```
                {example_structure}
                  ```

                3. **Extract Detailed Attributes**
                - For each entity, identify all relevant attributes
                - Group related attributes into nested objects
                - Use appropriate data types (string, number, boolean, array, object)
                - Include examples where possible

                4. **Define Relationships**
                - Explicitly define relationships between entities in the relationships field
                - Use descriptive relationship names (has-one, has-many, belongs-to)
                - Ensure bidirectional relationships are captured

                5. **Table Data Processing**
                - Convert table structures into nested object representations
                - Preserve table hierarchies in the schema
                - Capture relationships between table entities

                Remember:
                - Focus on creating deep, nested structures rather than flat fields
                - Use the nested_fields property to create hierarchical relationships
                - Define clear type definitions using the FieldType model
                - Include relationship definitions in the schema

                This is an iterative process - maintain existing fields and relationships while enhancing the schema with new information from each chunk.

                PDF Content: {sample_content}
                """
                result = await self.schema_extractor.ainvoke(
                    {
                        "messages": [
                            {
                                "role": "system",
                                "content": prompt,
                            },
                            {"role": "user", "content": "Analyze the content and generate a schema for this tractor manual. "},
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
