```python
# rag/vectorstore.py

from typing import Dict, Any, List, Optional, Union
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.logger import LoggerSetup
import json
import hashlib
from datetime import datetime

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
    """Enhanced vector store with validation and error handling."""
    
    def __init__(self, index_name: str, api_key: str, environment: str):
        self.logger = LoggerSetup.get_logger("rag.vectorstore")
        
        try:
            # Initialize Pinecone with namespace support
            pinecone.init(api_key=api_key, environment=environment)
            
            # Create index if it doesn't exist
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    dimension=1536,  # OpenAI embeddings dimension
                    metric='cosine',
                    metadata_config={
                        "indexed": ["type", "name", "file", "line_number", "complexity"]
                    }
                )
            
            self.index = pinecone.Index(index_name)
            self.embeddings = OpenAIEmbeddings()
            self.logger.info(f"Successfully initialized vector store with index: {index_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize vector store: {str(e)}")
            raise VectorStoreError(f"Initialization failed: {str(e)}")

    def _validate_node_data(self, node_data: Dict[str, Any]) -> None:
        """Validate required fields in node data."""
        required_fields = ['name', 'line_number', 'complexity']
        missing_fields = [field for field in required_fields if field not in node_data]
        
        if missing_fields:
            self.logger.error(f"Missing required fields in node data: {missing_fields}")
            raise ValidationError("Missing required fields", missing_fields)

    def _validate_metadata(self, metadata: Dict[str, Any]) -> None:
        """Validate metadata completeness."""
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
        """Store AST node with validation and error handling."""
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
            
            # Store node
            node_text = json.dumps(node_data, indent=2)
            vector = await self.embeddings.aembed_query(node_text)
            
            self.index.upsert(
                vectors=[(vector_id, vector, metadata)],
                namespace=namespace
            )
            
            self.logger.info(f"Successfully stored AST node: {vector_id}")
            
        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Failed to store AST node: {str(e)}")
            raise StorageError(f"Storage operation failed: {str(e)}")

    async def query_ast_nodes(
        self,
        query: str,
        node_type: Optional[str] = None,
        top_k: int = 5,
        namespace: str = "ast_nodes"
    ) -> List[Dict[str, Any]]:
        """Query AST nodes with enhanced error handling."""
        try:
            # Generate query embedding
            query_vector = await self.embeddings.aembed_query(query)
            
            # Prepare filter if node_type is specified
            filter_dict = {"type": node_type} if node_type else None
            
            # Query Pinecone
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                filter=filter_dict,
                include_metadata=True
            )
            
            # Format results
            formatted_results = []
            for match in results.matches:
                formatted_results.append({
                    "score": match.score,
                    "metadata": match.metadata,
                    "id": match.id
                })
            
            self.logger.info(f"Successfully queried AST nodes: {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Failed to query AST nodes: {str(e)}")
            raise StorageError(f"Query operation failed: {str(e)}")

    async def store_ast_analysis(
        self,
        analysis_results: Dict[str, Any],
        file_path: str
    ) -> None:
        """Store complete AST analysis results."""
        try:
            # Store functions
            for func in analysis_results.get("functions", []):
                await self.store_ast_node("function", func, file_path)
            
            # Store classes
            for class_info in analysis_results.get("classes", []):
                await self.store_ast_node("class", class_info, file_path)
                
                # Store methods within classes
                for method in class_info.get("methods", []):
                    method_data = {
                        **method,
                        "class_name": class_info["name"]
                    }
                    await self.store_ast_node("method", method_data, file_path)
            
            self.logger.info(f"Successfully stored complete AST analysis for {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to store AST analysis: {str(e)}")
            raise StorageError(f"Analysis storage failed: {str(e)}")
```