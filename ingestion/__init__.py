"""
FinSight AI Ingestion Package
PDF/Word extraction, text cleaning, metadata parsing, structure-aware chunking,
and the end-to-end ingestion pipeline.
"""

from ingestion.chunker import DocumentChunk, StructureAwareChunker
from ingestion.cleaner import TextCleaner
from ingestion.loader import DocumentLoader
from ingestion.metadata import DocumentMetadata, compute_file_hash, infer_metadata_from_filename_or_text
from ingestion.pipeline import IngestionPipeline, IngestionResult, pipeline

__all__ = [
    "DocumentLoader",
    "TextCleaner",
    "DocumentMetadata",
    "compute_file_hash",
    "infer_metadata_from_filename_or_text",
    "DocumentChunk",
    "StructureAwareChunker",
    "IngestionPipeline",
    "IngestionResult",
    "pipeline",
]
