```python
from typing import Dict, Any, Optional
from datetime import datetime
import traceback
from core.logger import LoggerSetup

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
        self.timestamp = datetime.now().isoformat()
        
        # Include context in message
        full_message = f"{message} [Operation: {operation}]"
        if context:
            full_message += f" [Context: {context}]"
        if original_error:
            full_message += f" [Original error: {str(original_error)}]"
            
        super().__init__(full_message)

class CodeAnalysisError(APIError):
    """Exception for code analysis failures."""
    pass

class APIClient:
    """Enhanced API client with comprehensive error handling."""
    
    def __init__(self):
        self.logger = LoggerSetup.get_logger("api_client")

    async def analyze_code(
        self,
        code_content: str,
        analysis_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze code with enhanced error handling and context preservation.
        
        Args:
            code_content: Source code to analyze
            analysis_params: Optional analysis parameters
            
        Returns:
            Dict[str, Any]: Analysis results
            
        Raises:
            CodeAnalysisError: If analysis fails
        """
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
            
            # Perform analysis
            result = await self.azure.analyze_code(
                code_content,
                code_type="module",
                **operation_context['params']
            )
            
            # Log success
            self.logger.info(
                "Code analysis completed successfully",
                extra={'context': operation_context}
            )
            
            return result
            
        except Exception as e:
            # Capture full error context
            error_context = {
                **operation_context,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'stack_trace': traceback.format_exc()
            }
            
            # Log error with full context
            self.logger.error(
                "Code analysis failed",
                extra={'error_context': error_context},
                exc_info=True
            )
            
            # Record error metrics
            await self.metrics.record_error(
                service="api",
                operation="code_analysis",
                error_type=type(e).__name__,
                error_context=error_context
            )
            
            # Raise with preserved context
            raise CodeAnalysisError(
                message="Analysis failed",
                operation='code_analysis',
                context=error_context,
                original_error=e
            )

    async def make_api_request(
        self,
        endpoint: str,
        method: str = 'GET',
        data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make API request with enhanced error handling.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            data: Request data
            
        Returns:
            Dict[str, Any]: API response
            
        Raises:
            APIError: If request fails
        """
        operation_context = {
            'operation': 'api_request',
            'endpoint': endpoint,
            'method': method,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Log request
            self.logger.info(
                f"Making API request to {endpoint}",
                extra={'context': operation_context}
            )
            
            # Make request
            response = await self._execute_request(endpoint, method, data)
            
            # Log success
            self.logger.info(
                f"API request to {endpoint} completed successfully",
                extra={'context': operation_context}
            )
            
            return response
            
        except Exception as e:
            # Capture error context
            error_context = {
                **operation_context,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'stack_trace': traceback.format_exc()
            }
            
            # Log error
            self.logger.error(
                f"API request to {endpoint} failed",
                extra={'error_context': error_context},
                exc_info=True
            )
            
            # Record metrics
            await self.metrics.record_error(
                service="api",
                operation="api_request",
                error_type=type(e).__name__,
                error_context=error_context
            )
            
            # Raise with context
            raise APIError(
                message="API request failed",
                operation='api_request',
                context=error_context,
                original_error=e
            )

    async def _execute_request(
        self,
        endpoint: str,
        method: str,
        data: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        Execute API request with rate limiting and retries.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            data: Request data
            
        Returns:
            Dict[str, Any]: API response
            
        Raises:
            APIError: If request fails
        """
        # Apply rate limiting
        await self.rate_limiter.acquire()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method,
                    f"{self.base_url}/{endpoint}",
                    json=data,
                    headers=self._get_headers()
                ) as response:
                    response.raise_for_status()
                    return await response.json()
                    
        except aiohttp.ClientError as e:
            raise APIError(
                message="Request failed",
                operation='execute_request',
                context={'endpoint': endpoint, 'method': method},
                original_error=e
            )
```