```python
# api_client.py

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import asyncio
from datetime import datetime
from core.logger import LoggerSetup

@dataclass
class APIResponse:
    """Base class for all API responses."""
    status: str
    timestamp: datetime
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        response = {
            "status": self.status,
            "timestamp": self.timestamp.isoformat()
        }
        if self.error:
            response["error"] = self.error
        return response

@dataclass
class CodeAnalysisResult(APIResponse):
    """Standardized code analysis result."""
    details: Dict[str, Any]
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            **super().to_dict(),
            "details": self.details,
            "metrics": self.metrics
        }

@dataclass
class QueryResponse(APIResponse):
    """Standardized query response."""
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            **super().to_dict(),
            "answer": self.answer,
            "sources": self.sources,
            "metadata": self.metadata
        }

class ValidationError(Exception):
    """Exception raised when parameter validation fails."""
    pass

class APIError(Exception):
    """Base exception for API operations."""
    def __init__(
        self,
        message: str,
        operation: str,
        context: Dict[str, Any],
        original_error: Optional[Exception] = None
    ):
        self.operation = operation
        self.context = context
        self.original_error = original_error
        self.timestamp = datetime.now()
        super().__init__(message)

class ParameterValidator:
    """Validator for API parameters."""
    
    def __init__(self):
        self.logger = LoggerSetup.get_logger("parameter_validator")
        
        # Define standard constraints
        self.constraints = {
            "code_content": {
                "min_length": 1,
                "max_length": 100000,  # 100KB
                "required": True
            },
            "question": {
                "min_length": 1,
                "max_length": 1000,
                "pattern": r'^[^<>]*$',  # No HTML tags
                "required": True
            },
            "service": {
                "allowed_values": ["azure", "openai", "anthropic"],
                "required": True
            }
        }

    def validate(
        self,
        parameter_name: str,
        value: Any,
        custom_constraints: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Validate parameter against constraints.
        
        Args:
            parameter_name: Name of parameter
            value: Parameter value
            custom_constraints: Optional custom constraints
            
        Raises:
            ValidationError: If validation fails
        """
        constraints = custom_constraints or self.constraints.get(parameter_name, {})
        
        try:
            # Check required fields
            if constraints.get("required", False) and value is None:
                raise ValidationError(f"Parameter {parameter_name} is required")

            # Validate string parameters
            if isinstance(value, str):
                self._validate_string(parameter_name, value, constraints)
            
            # Validate allowed values
            allowed_values = constraints.get("allowed_values")
            if allowed_values and value not in allowed_values:
                raise ValidationError(
                    f"Parameter {parameter_name} must be one of: {allowed_values}"
                )
                
        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Validation error for {parameter_name}: {str(e)}")
            raise ValidationError(f"Validation failed: {str(e)}")

    def _validate_string(
        self,
        parameter_name: str,
        value: str,
        constraints: Dict[str, Any]
    ) -> None:
        """Validate string parameter constraints."""
        # Check length constraints
        min_length = constraints.get("min_length")
        if min_length and len(value) < min_length:
            raise ValidationError(
                f"Parameter {parameter_name} must be at least {min_length} characters"
            )
            
        max_length = constraints.get("max_length")
        if max_length and len(value) > max_length:
            raise ValidationError(
                f"Parameter {parameter_name} must not exceed {max_length} characters"
            )
            
        # Check pattern constraint
        pattern = constraints.get("pattern")
        if pattern and not re.match(pattern, value):
            raise ValidationError(
                f"Parameter {parameter_name} does not match required pattern"
            )

class StandardizedAPIClient:
    """API client with standardized validation and responses."""
    
    def __init__(self):
        self.validator = ParameterValidator()
        self.logger = LoggerSetup.get_logger("api_client")

    async def analyze_code(
        self,
        code_content: str,
        analysis_params: Optional[Dict[str, Any]] = None
    ) -> CodeAnalysisResult:
        """
        Analyze code with standardized return type.
        
        Args:
            code_content: Code to analyze
            analysis_params: Optional analysis parameters
            
        Returns:
            CodeAnalysisResult: Standardized analysis result
            
        Raises:
            ValidationError: If parameters are invalid
            APIError: If analysis fails
        """
        # Validate parameters
        self.validator.validate("code_content", code_content)
        
        operation_context = {
            'operation': 'code_analysis',
            'code_size': len(code_content),
            'timestamp': datetime.now().isoformat(),
            'params': analysis_params or {}
        }
        
        try:
            # Record operation start
            self.logger.info(
                "Starting code analysis",
                extra={'context': operation_context}
            )
            
            # Execute analysis
            analysis = await self.azure.analyze_code(
                code_content,
                code_type="module",
                **operation_context['params']
            )
            
            # Return standardized result
            return CodeAnalysisResult(
                status="success",
                timestamp=datetime.now(),
                details=analysis["details"],
                metrics=analysis["metrics"]
            )
            
        except Exception as e:
            self.logger.error(
                "Code analysis failed",
                extra={'context': operation_context},
                exc_info=True
            )
            
            raise APIError(
                message="Analysis failed",
                operation='code_analysis',
                context=operation_context,
                original_error=e
            )

    async def query(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> QueryResponse:
        """
        Execute query with standardized return type.
        
        Args:
            question: Query question
            context: Optional query context
            
        Returns:
            QueryResponse: Standardized query response
            
        Raises:
            ValidationError: If parameters are invalid
            APIError: If query fails
        """
        # Validate parameters
        self.validator.validate("question", question)
        
        try:
            # Execute query
            result = await self._execute_query(question, context)
            
            # Return standardized response
            return QueryResponse(
                status="success",
                timestamp=datetime.now(),
                answer=result["answer"],
                sources=result["sources"],
                metadata=result["metadata"]
            )
            
        except Exception as e:
            raise APIError(
                message="Query failed",
                operation='query',
                context={'question': question},
                original_error=e
            )

    async def _execute_query(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute actual query operation."""
        try:
            # Get relevant documents
            documents = await self.retriever.get_relevant_documents(
                question,
                context
            )
            
            # Generate response
            response = await self.llm.generate_response(
                question,
                documents,
                context
            )
            
            return {
                "answer": response.content,
                "sources": [doc.metadata for doc in documents],
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "question": question,
                    "document_count": len(documents)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Query execution failed: {str(e)}")
            raise
```