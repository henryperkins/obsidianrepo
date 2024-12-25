```python
from typing import Dict, Any, List
from datetime import datetime
import logging
from core.logger import LoggerSetup

# Custom exceptions for vector store operations
class VectorStoreError(Exception):
    """Base exception for vector store operations."""
    pass

class ValidationError(VectorStoreError):
    """Exception raised when data validation fails."""
    def __init__(self, message: str, missing_fields: List[str]):
        self.missing_fields = missing_fields
        super().__init__(f"{message}: {', '.join(missing_fields)}")

class StorageError(VectorStoreError):
    """Exception raised when vector storage operations fail."""
    pass

class CodeVectorStore:
    """Enhanced vector store with validation."""
    
    def __init__(self):
        self.logger = LoggerSetup.get_logger("vector_store")

    def _validate_node_data(self, node_data: Dict[str, Any]) -> None:
        """
        Validate required fields in node data.
        
        Args:
            node_data: Dictionary containing node data to validate
            
        Raises:
            ValidationError: If required fields are missing
        """
        required_fields = ['name', 'line_number', 'complexity']
        missing_fields = [field for field in required_fields if field not in node_data]
        
        if missing_fields:
            self.logger.error(f"Missing required fields in node data: {missing_fields}")
            raise ValidationError("Missing required fields", missing_fields)

    def _validate_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Validate metadata completeness.
        
        Args:
            metadata: Dictionary containing metadata to validate
            
        Raises:
            ValidationError: If required metadata is missing
        """
        required_metadata = ['type', 'name', 'file', 'line_number']
        missing_metadata = [field for field in required_metadata if field not in metadata]
        
        if missing_metadata:
            self.logger.error(f"Missing required metadata: {missing_metadata}")
            raise ValidationError("Missing required metadata", missing_metadata)

    async def store_ast_node(
        self,
        node_type: str,
        node_data: Dict[str, Any],
        file_path: str,
        namespace: str = "ast_nodes"
    ) -> None:
        """
        Store AST node with enhanced validation and error handling.
        
        Args:
            node_type: Type of AST node
            node_data: Node data to store
            file_path: Source file path
            namespace: Pinecone namespace
            
        Raises:
            ValidationError: If data validation fails
            StorageError: If storage operation fails
        """
        try:
            # Validate node data
            self._validate_node_data(node_data)
            
            # Generate vector ID with validated fields
            vector_id = f"{file_path}_{node_type}_{node_data['name']}_{node_data['line_number']}"
            
            # Prepare metadata
            metadata = {
                "type": node_type,
                "name": node_data["name"],
                "file": file_path,
                "line_number": node_data["line_number"],
                "complexity": node_data.get("complexity", 0),
                "timestamp": datetime.now().isoformat()
            }
            
            # Validate metadata
            self._validate_metadata(metadata)
            
            # Store in vector database
            try:
                await self._store_vector(vector_id, metadata, node_data, namespace)
                self.logger.info(f"Successfully stored node: {vector_id}")
                
            except Exception as e:
                raise StorageError(f"Failed to store vector: {str(e)}") from e
                
        except ValidationError as e:
            self.logger.error(f"Validation failed: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in store_ast_node: {str(e)}")
            raise VectorStoreError(f"Unexpected error: {str(e)}") from e

    async def _store_vector(
        self,
        vector_id: str,
        metadata: Dict[str, Any],
        node_data: Dict[str, Any],
        namespace: str
    ) -> None:
        """
        Internal method to store vector with metadata.
        
        Args:
            vector_id: Unique identifier for the vector
            metadata: Metadata for the vector
            node_data: Complete node data
            namespace: Storage namespace
            
        Raises:
            StorageError: If storage operation fails
        """
        try:
            # Generate embedding for the node
            vector = await self._generate_embedding(node_data)
            
            # Store in database
            await self.index.upsert(
                vectors=[(vector_id, vector, metadata)],
                namespace=namespace
            )
            
        except Exception as e:
            self.logger.error(f"Vector storage failed: {str(e)}")
            raise StorageError(f"Failed to store vector {vector_id}: {str(e)}") from e

    async def _generate_embedding(self, node_data: Dict[str, Any]) -> List[float]:
        """
        Generate embedding for node data.
        
        Args:
            node_data: Node data to generate embedding for
            
        Returns:
            List[float]: Generated embedding vector
            
        Raises:
            StorageError: If embedding generation fails
        """
        try:
            # Convert node data to string representation
            node_text = json.dumps(node_data, indent=2)
            
            # Generate embedding
            vector = await self.embeddings.embed_query(node_text)
            return vector
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {str(e)}")
            raise StorageError(f"Failed to generate embedding: {str(e)}") from e
```