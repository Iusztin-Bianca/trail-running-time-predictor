"""
Data management module for ML pipeline.

Handles Azure Blob Storage operations and training data initialization.
"""
from .blob_storage import BlobStorageManager
from .data_initialization import DataInitializer

__all__ = ["BlobStorageManager", "DataInitializer"]
