from typing import List, Dict, Any
import logging
from datetime import datetime

from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import elements_from_dicts
from database import get_session

from utils.logging_config import setup_detailed_logging

logger = setup_detailed_logging()

class ChunkProcessor:
    def __init__(self, engine=None):
        self.logger = logger
        self.engine = engine

    def chunk_elements(
        self, elements_str: List[Any], doc_id: int = None
    ) -> List[Dict[str, Any]]:
        """Chunk elements by title and return structured chunks with metadata."""
        self.logger.info("Starting element chunking with %d elements", len(elements_str))

        try:
            elements = elements_from_dicts(elements_str)

            # Log element types before chunking
            element_types = {}
            for element in elements:
                element_type = type(element).__name__
                element_types[element_type] = element_types.get(element_type, 0) + 1

            self.logger.info(f"Element type distribution before chunking: {element_types}")

            # Create chunks while preserving table elements
            chunks = chunk_by_title(
                elements,
                multipage_sections=True,
                combine_text_under_n_chars=500,
                new_after_n_chars=1500,
                max_characters=2000,
            )

            # Log chunk types after chunking
            chunk_types = {}
            for chunk in chunks:
                chunk_type = type(chunk).__name__
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

            self.logger.info(f"Chunk type distribution after chunking: {chunk_types}")

            # Structure chunks with metadata
            structured_chunks = []
            session = get_session(self.engine)
            try:
                for idx, chunk in enumerate(chunks, 1):
                    # Convert chunk to JSON to preserve all metadata
                    from unstructured.staging.base import elements_to_json
                    chunk_json = elements_to_json([chunk], indent=2)
                    self.logger.info(f"Chunk JSON: {chunk_json}")
                    # Create structured chunk data
                    chunk_data = {
                        'chunk_number': idx,
                        'content': chunk_json,
                        'type': type(chunk).__name__,
                    }
                    structured_chunks.append(chunk_data)

                    if idx % 100 == 0:
                        self.logger.info(f"Processed {idx} chunks...")

                self.logger.info(f"Successfully chunked {len(structured_chunks)} chunks")
                self.logger.info(f"Structured chunks: {structured_chunks}")
                return structured_chunks

            except Exception as e:
                session.rollback()
                self.logger.error(f"Failed to process chunks: {str(e)}")
                raise
            finally:
                session.close()

        except Exception as e:
            self.logger.error(f"Error chunking elements: {str(e)}")
            raise
