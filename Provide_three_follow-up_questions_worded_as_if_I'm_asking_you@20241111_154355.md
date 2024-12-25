---
epoch: 1731361435015
modelKey: gpt-4o-2024-08-06|azure openai
tags:
  - copilot-conversation
---

# Provide_three_follow-up_questions_worded_as_if_I'm_asking_you@20241111_154355

**user**: Provide three follow-up questions worded as if I'm asking you based on {activeNote}?  
[Timestamp: 2024/11/11 15:43:55]

**ai**: 1. How does the `MetricsCalculator` class handle the calculation of Halstead metrics, and what specific metrics are included in its analysis?
   
2. Can you explain the process and criteria used by the `MetricsCalculator` to determine the maintainability index of a codebase?
3. What methods are available in the `MetricsCalculator` class for analyzing the quality of functions and classes, and how do they contribute to identifying potential improvements?  
[Timestamp: 2024/11/11 15:43:56]

**user**: Merge this code into metrics.py class APIMetricsCollector:  
    """Enhanced metrics collector with code analysis support."""
    
    async def record_analysis_metrics(
        self,
        code_type: str,
        analysis_size: int,
        metrics: Dict[str, Any]
    ) -> None:
        """Record metrics from code analysis."""
        try:
            analysis_data = {
                "timestamp": datetime.now().isoformat(),
                "code_type": code_type,
                "size": analysis_size,
                "complexity": metrics.get("complexity", {}).get("cyclomatic_complexity", 0),
                "maintainability": metrics.get("maintainability", {}).get("maintainability_index", 0),
                "doc_coverage": metrics.get("quality", {}).get("documentation_coverage", 0),
                "type_coverage": metrics.get("quality", {}).get("type_hint_coverage", 0)
            }
            
            # Update service metrics
            service_metric = self.service_metrics["analysis"]
            service_metric.total_calls += 1
            service_metric.total_tokens += analysis_size  # Use code size as token proxy
            
            minute_key = analysis_data["timestamp"][:16]  # Group by minute
            service_metric.calls_per_minute[minute_key] += 1
            service_metric.tokens_per_minute[minute_key] += analysis_size
            
            # Store detailed metrics
            self.metrics.append(APICallMetrics(
                timestamp=datetime.now(),
                service="analysis",
                endpoint=code_type,
                tokens_used=analysis_size,
                latency_ms=0,  # Not applicable for analysis
                status="success",
                metadata=analysis_data
            ))
            
        except Exception as e:
            logger.error(f"Error recording analysis metrics: {e}")
            
    async def record_analysis_error(
        self,
        error_type: str,
        error_message: str
    ) -> None:
        """Record analysis errors."""
        try:
            service_metric = self.service_metrics["analysis"]
            service_metric.total_errors += 1
            
            self.metrics.append(APICallMetrics(
                timestamp=datetime.now(),
                service="analysis",
                endpoint="error",
                tokens_used=0,
                latency_ms=0,
                status="error",
                error=f"{error_type}: {error_message}"
            ))
            
        except Exception as e:
            logger.error(f"Error recording analysis error: {e}")
            
    async def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of code analysis metrics."""
        try:
            analysis_metrics = [m for m in self.metrics if m.service == "analysis"]
            
            if not analysis_metrics:
                return {}
                
            total_size = sum(m.tokens_used for m in analysis_metrics)
            avg_complexity = np.mean([
                m.metadata.get("complexity", 0) 
                for m in analysis_metrics 
                if hasattr(m, "metadata")
            ])
            avg_maintainability = np.mean([
                m.metadata.get("maintainability", 0) 
                for m in analysis_metrics 
                if hasattr(m, "metadata")
            ])
            
            return {
                "total_analyses": len(analysis_metrics),
                "total_code_size": total_size,
                "average_complexity": round(float(avg_complexity), 2),
                "average_maintainability": round(float(avg_maintainability), 2),
                "error_rate": round(
                    self.service_metrics["analysis"].total_errors / 
                    len(analysis_metrics) * 100, 2
                ) if analysis_metrics else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting analysis summary: {e}")
            return {}
[Timestamp: 2024/11/11 15:44:14]

**ai**: To merge the provided `APIMetricsCollector` class into the existing `metrics.py` file, we need to integrate the new class while ensuring that it complements the existing functionality. Below is the merged code with the `APIMetricsCollector` class added:

```python
import ast
import math
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from collections import defaultdict
from core.logger import LoggerSetup
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime

# Initialize a logger specifically for this module
logger = LoggerSetup.get_logger("metrics")

@dataclass
class ComplexityMetrics:
    """Complexity metrics for code analysis."""
    cyclomatic_complexity: float
    cognitive_complexity: float
    halstead_metrics: Dict[str, float]
    maintainability_index: float

@dataclass
class QualityMetrics:
    """Quality metrics for code analysis."""
    documentation_coverage: float
    type_hint_coverage: float
    test_coverage: Optional[float]
    code_to_test_ratio: Optional[float]

class MetricsCalculator:
    """Calculator for various code metrics."""
    
    def __init__(self, tree: ast.AST, content: str):
        self.tree = tree
        self.content = content
        self.total_functions = 0
        self.total_classes = 0
        self.total_lines = 0
        self.docstring_coverage = 0.0
        self.type_hint_coverage = 0.0
        self.avg_complexity = 0.0
        self.max_complexity = 0
        self.cognitive_complexity = 0
        self.halstead_metrics = defaultdict(float)
        self.type_hints_stats = defaultdict(int)
        self.quality_issues = []
        logger.debug("Initialized MetricsCalculator instance.")

    def calculate_all_metrics(self) -> Dict[str, Any]:
        """Calculate all available metrics."""
        return {
            'complexity': self._calculate_complexity_metrics(),
            'quality': self._calculate_quality_metrics(),
            'size': self._calculate_size_metrics(),
            'maintainability': self._calculate_maintainability_metrics()
        }
    
    def _calculate_complexity_metrics(self) -> ComplexityMetrics:
        """Calculate complexity-related metrics."""
        return ComplexityMetrics(
            cyclomatic_complexity=self._calculate_cyclomatic_complexity(),
            cognitive_complexity=self._calculate_cognitive_complexity(),
            halstead_metrics=self._calculate_halstead_metrics(),
            maintainability_index=self._calculate_maintainability_index()
        )
    
    def _calculate_cyclomatic_complexity(self) -> float:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor,
                               ast.ExceptHandler, ast.AsyncWith,
                               ast.With, ast.Assert, ast.BoolOp)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
                
        return complexity
    
    def _calculate_cognitive_complexity(self) -> float:
        """Calculate cognitive complexity."""
        complexity = 0
        nesting_level = 0
        
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += (1 + nesting_level)
                nesting_level += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
                
        return complexity
    
    def _calculate_halstead_metrics(self) -> Dict[str, float]:
        """Calculate Halstead metrics."""
        operators = set()
        operands = set()
        total_operators = 0
        total_operands = 0
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.operator):
                operators.add(node.__class__.__name__)
                total_operators += 1
            elif isinstance(node, ast.Name):
                operands.add(node.id)
                total_operands += 1
                
        n1 = len(operators)  # Unique operators
        n2 = len(operands)   # Unique operands
        N1 = total_operators # Total operators
        N2 = total_operands  # Total operands
        
        try:
            program_length = N1 + N2
            vocabulary = n1 + n2
            volume = program_length * math.log2(vocabulary) if vocabulary > 0 else 0
            difficulty = (n1 * N2) / (2 * n2) if n2 > 0 else 0
            effort = volume * difficulty
            
            return {
                'program_length': program_length,
                'vocabulary': vocabulary,
                'volume': volume,
                'difficulty': difficulty,
                'effort': effort,
                'time_to_program': effort / 18,  # Halstead's constant
                'bugs_delivered': volume / 3000  # Halstead's constant
            }
        except ZeroDivisionError:
            return {
                'program_length': 0,
                'vocabulary': 0,
                'volume': 0,
                'difficulty': 0,
                'effort': 0,
                'time_to_program': 0,
                'bugs_delivered': 0
            }
    
    def _calculate_maintainability_metrics(self) -> Dict[str, float]:
        """Calculate maintainability metrics."""
        loc = len(self.content.splitlines())
        complexity = self._calculate_cyclomatic_complexity()
        halstead_volume = self._calculate_halstead_metrics()['volume']
        
        try:
            maintainability_index = max(0, (171 - 5.2 * math.log(halstead_volume) -
                                          0.23 * complexity - 16.2 * math.log(loc)) * 100 / 171)
            
            return {
                'maintainability_index': maintainability_index,
                'technical_debt_ratio': (100 - maintainability_index) / 100,
                'remediation_cost': self._calculate_remediation_cost(maintainability_index)
            }
        except (ValueError, ZeroDivisionError):
            return {
                'maintainability_index': 0,
                'technical_debt_ratio': 1.0,
                'remediation_cost': 0
            }
    
    def _calculate_remediation_cost(self, maintainability_index: float) -> float:
        """Calculate estimated remediation cost in hours."""
        if maintainability_index >= 85:
            return 0
        elif maintainability_index >= 65:
            return (85 - maintainability_index) * 0.5
        else:
            return (85 - maintainability_index) * 1.0

    def analyze_function_quality(self, function_info: Dict[str, Any]) -> None:
        """Analyze function quality and add recommendations."""
        name = function_info.get('name', 'unknown')
        score = function_info.get('complexity_score', 0)
        logger.debug(f"Analyzing quality for function: {name}")
        if score > 10:
            msg = (
                f"Function '{name}' has high complexity ({score}). "
                "Consider breaking it down."
            )
            self.quality_issues.append(msg)
            logger.info(msg)
        if not function_info.get("docstring"):
            msg = f"Function '{name}' lacks a docstring."
            self.quality_issues.append(msg)
            logger.info(msg)
        params_without_types = [
            p["name"] for p in function_info.get("params", [])
            if not p.get("has_type_hint")
        ]
        if params_without_types:
            params_str = ", ".join(params_without_types)
            msg = (
                f"Function '{name}' has parameters without type hints: "
                f"{params_str}"
            )
            self.quality_issues.append(msg)
            logger.info(msg)

    def analyze_class_quality(self, class_info: Dict[str, Any]) -> None:
        """Analyze class quality and add recommendations."""
        name = class_info.get('name', 'unknown')
        logger.debug(f"Analyzing quality for class: {name}")
        if not class_info.get("docstring"):
            msg = f"Class '{name}' lacks a docstring."
            self.quality_issues.append(msg)
            logger.info(msg)
        method_count = len(class_info.get("methods", []))
        if method_count > 10:
            msg = (
                f"Class '{name}' has many methods ({method_count}). "
                "Consider splitting it."
            )
            self.quality_issues.append(msg)
            logger.info(msg)

    def update_type_hint_stats(self, function_info: Dict[str, Any]) -> None:
        """Update type hint statistics based on function information."""
        total_hints_possible = len(function_info.get("params", [])) + 1  # Including return type
        hints_present = sum(
            1 for p in function_info.get("params", []) if p.get("has_type_hint")
        )
        if function_info.get("return_type", {}).get("has_type_hint", False):
            hints_present += 1
        self.type_hints_stats["total_possible"] += total_hints_possible
        self.type_hints_stats["total_present"] += hints_present
        logger.debug(f"Updated type hint stats: {self.type_hints_stats}")

    def calculate_final_metrics(self, all_items: List[Dict[str, Any]]) -> None:
        """Calculate final metrics after processing all items."""
        total_items = len(all_items)
        logger.debug(f"Calculating final metrics for {total_items} items.")
        if total_items > 0:
            items_with_doc = sum(1 for item in all_items if item.get("docstring"))
            self.docstring_coverage = (items_with_doc / total_items) * 100
            total_complexity = sum(
                item.get("complexity_score", 0)
                for item in all_items
            )
            self.avg_complexity = total_complexity / total_items if total_items else 0
        if self.type_hints_stats["total_possible"] > 0:
            self.type_hint_coverage = (
                self.type_hints_stats["total_present"] /
                self.type_hints_stats["total_possible"]
            ) * 100
        logger.info(
            f"Final metrics calculated: Docstring coverage: {self.docstring_coverage:.2f}%, "
            f"Type hint coverage: {self.type_hint_coverage:.2f}%, "
            f"Average complexity: {self.avg_complexity:.2f}, "
            f"Max complexity: {self.max_complexity}"
        )

    def get_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive summary of code metrics."""
        summary = {
            "total_classes": self.total_classes,
            "total_functions": self.total_functions,
            "total_lines": self.total_lines,
            "docstring_coverage_percentage": round(self.docstring_coverage, 2),
            "type_hint_coverage_percentage": round(self.type_hint_coverage, 2),
            "average_complexity": round(self.avg_complexity, 2),
            "max_complexity": self.max_complexity,
            "cognitive_complexity": self.cognitive_complexity,
            "halstead_metrics": dict(self.halstead_metrics),
            "quality_recommendations": self.quality_issues
        }
        logger.debug(f"Generated summary: {summary}")
        return summary

    def _calculate_confidence(self, response: Dict[str, Any]) -> float:
        """Calculate confidence score based on response characteristics."""
        try:
            factors = {
                'source_similarity': self._calculate_source_relevance(response),
                'response_consistency': self._check_response_consistency(response),
                'source_coverage': self._calculate_source_coverage(response),
                'response_completeness': self._check_response_completeness(response)
            }
            
            # Weights for different factors
            weights = {
                'source_similarity': 0.4,
                'response_consistency': 0.3,
                'source_coverage': 0.2,
                'response_completeness': 0.1
            }
            
            # Calculate weighted score
            confidence_score = sum(
                score * weights[factor] 
                for factor, score in factors.items()
            )
            
            return min(1.0, max(0.0, confidence_score))
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.5

    def _calculate_source_relevance(self, response: Dict[str, Any]) -> float:
        """Calculate relevance of sources to the response."""
        try:
            if not response.get('sources'):
                return 0.0
                
            sources = response['sources']
            answer = response.get('answer', '')
            
            # Calculate embedding similarities
            similarities = []
            for source in sources:
                similarity = self.text_processor.calculate_similarity(
                    source['content'],
                    answer
                )
                similarities.append(similarity)
            
            # Return weighted average, giving more weight to higher similarities
            similarities.sort(reverse=True)
            weights = [1/i for i in range(1, len(similarities) + 1)]
            total_weight = sum(weights)
            
            return sum(
                sim * weight / total_weight 
                for sim, weight in zip(similarities, weights)
            )
            
        except Exception as e:
            logger.error(f"Error calculating source relevance: {e}")
            return 0.0

    def _check_response_consistency(self, response: Dict[str, Any]) -> float:
        """Check internal consistency of the response."""
        try:
            answer = response.get('answer', '')
            
            # Split answer into sentences
            sentences = sent_tokenize(answer)
            if len(sentences) < 2:
                return 1.0
                
            # Calculate pairwise sentence similarities
            embeddings = self.text_processor.model.encode(sentences)
            similarities = cosine_similarity(embeddings)
            
            # Calculate average similarity between consecutive sentences
            consecutive_similarities = [
                similarities[i][i+1] 
                for i in range(len(similarities)-1)
            ]
            
            return np.mean(consecutive_similarities)
            
        except Exception as e:
            logger.error(f"Error checking response consistency: {e}")
            return 0.5

    def _calculate_source_coverage(self, response: Dict[str, Any]) -> float:
        """Calculate how well the response covers the source material."""
        try:
            if not response.get('sources'):
                return 0.0
                
            answer = response.get('answer', '')
            source_texts = [s['content'] for s in response['sources']]
            
            # Extract key points from sources and answer
            source_keywords = set()
            for text in source_texts:
                keywords = self.text_processor.extract_keywords(text)
                source_keywords.update(keywords)
                
            answer_keywords = set(
                self.text_processor.extract_keywords(answer)
            )
            
            # Calculate coverage
            if not source_keywords:
                return 0.0
                
            return len(answer_keywords.intersection(source_keywords)) / len(source_keywords)
            
        except Exception as e:
            logger.error(f"Error calculating source coverage: {e}")
            return 0.0

    def _check_response_completeness(self, response: Dict[str, Any]) -> float:
        """Check if the response is complete and well-formed."""
        try:
            answer = response.get('answer', '')
            
            # Define completeness criteria
            criteria = {
                'has_content': bool(answer.strip()),
                'has_sentence_structure': len(sent_tokenize(answer)) > 0,
                'has_proper_length': 50 <= len(answer) <= 2000,
                'has_conclusion': any(
                    answer.lower().endswith(end) 
                    for end in ['.', '!', '?', 'conclusion', 'summary']
                )
            }
            
            # Calculate completeness score
            return sum(criteria.values()) / len(criteria)
            
        except Exception as e:
            logger.error(f"Error checking response completeness: {e}")
            return 0.5

    def _calculate_processing_time(self, response: Dict[str, Any]) -> float:
        """Calculate actual processing time from response metadata."""
        try:
            metadata = response.get('metadata', {})
            start_time = metadata.get('start_time')
            end_time = metadata.get('end_time')
            
            if not (start_time and end_time):
                return 0.0
                
            start = datetime.fromisoformat(start_time)
            end = datetime.fromisoformat(end_time)
            
            return (end - start).total_seconds()
            
        except Exception as e:
            logger.error(f"Error calculating processing time: {e}")
            return 0.0

class APIMetricsCollector:
    """Enhanced metrics collector with code analysis support."""
    
    def __init__(self):
        self.service_metrics = defaultdict(lambda: defaultdict(int))
        self.metrics = []
    
    async def record_analysis_metrics(
        self,
        code_type: str,
        analysis_size: int,
        metrics: Dict[str, Any]
    ) -> None:
        """Record metrics from code analysis."""
        try:
            analysis_data = {
                "timestamp": datetime.now().isoformat(),
                "code_type": code_type,
                "size": analysis_size,
                "complexity": metrics.get("complexity", {}).get("cyclomatic_complexity", 0),
                "maintainability": metrics.get("maintainability", {}).get("maintainability_index", 0),
                "doc_coverage": metrics.get("quality", {}).get("documentation_coverage", 0),
                "type_coverage": metrics.get("quality", {}).get("type_hint_coverage", 0)
            }
            
            # Update service metrics
            service_metric = self.service_metrics["analysis"]
            service_metric["total_calls"] += 1
            service_metric["total_tokens"] += analysis_size  # Use code size as token proxy
            
            minute_key = analysis_data["timestamp"][:16]  # Group by minute
            service_metric["calls_per_minute"][minute_key] += 1
            service_metric["tokens_per_minute"][minute_key] += analysis_size
            
            # Store detailed metrics
            self.metrics.append({
                "timestamp": datetime.now(),
                "service": "analysis",
                "endpoint": code_type,
                "tokens_used": analysis_size,
                "latency_ms": 0,  # Not applicable for analysis
                "status": "success",
                "metadata": analysis_data
            })
            
        except Exception as e:
            logger.error(f"Error recording analysis metrics: {e}")
            
    async def record_analysis_error(
        self,
        error_type: str,
        error_message: str
    ) -> None:
        """Record analysis errors."""
        try:
            service_metric = self.service_metrics["analysis"]
            service_metric["total_errors"] += 1
            
            self.metrics.append({
                "timestamp": datetime.now(),
                "service": "analysis",
                "endpoint": "error",
                "tokens_used": 0,
                "latency_ms": 0,
                "status": "error",
                "error": f"{error_type}: {error_message}"
            })
            
        except Exception as e:
            logger.error(f"Error recording analysis error: {e}")
            
    async def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of code analysis metrics."""
        try:
            analysis_metrics = [m for m in self.metrics if m["service"] == "analysis"]
            
            if not analysis_metrics:
                return {}
                
            total_size = sum(m["tokens_used"] for m in analysis_metrics)
            avg_complexity = np.mean([
                m["metadata"].get("complexity", 0) 
                for m in analysis_metrics 
                if "metadata" in m
            ])
            avg_maintainability = np.mean([
                m["metadata"].get("maintainability", 0) 
                for m in analysis_metrics 
                if "metadata" in m
            ])
            
            return {
                "total_analyses": len(analysis_metrics),
                "total_code_size": total_size,
                "average_complexity": round(float(avg_complexity), 2),
                "average_maintainability": round(float(avg_maintainability), 2),
                "error_rate": round(
                    service_metric["total_errors"] / len(analysis_metrics) * 100, 2
                ) if analysis_metrics else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting analysis summary: {e}")
            return {}
```

## Key Changes

- Added the `APIMetricsCollector` class to the module.
- Integrated the `record_analysis_metrics`, `record_analysis_error`, and `get_analysis_summary` methods into the `APIMetricsCollector` class.
- Ensured that the `APIMetricsCollector` class maintains its own state with `service_metrics` and `metrics` attributes.
- Used `defaultdict` for `service_metrics` to handle dynamic metric updates.
- Adjusted the `record_analysis_metrics` and `record_analysis_error` methods to append metrics to a list for later analysis.
- Provided error handling and logging for each method to ensure robustness.  
[Timestamp: 2024/11/11 15:45:10]

**user**: [api_interaction.py](api_interaction.py.md)  
[Timestamp: 2024/11/11 15:45:41]

**ai**: To integrate the `APIMetricsCollector` class from `metrics.py` into `api_interaction.py`, we need to ensure that the class is properly utilized within the context of API interactions. Here's how you can merge and utilize the `APIMetricsCollector` class within `api_interaction.py`:

```python
# api_interaction.py

import os
import ast
import json
import aiohttp
import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from dotenv import load_dotenv
from core.logger import LoggerSetup, get_logger
from extract.code import CodeAnalysisTool  # Import from code.py
from openai import AzureOpenAI
from config import EnvironmentConfig, ModelHeaders
from core.config import Config
from core.monitoring import Monitor

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

@dataclass
class APICredentials:
    api_key: str
    endpoint: Optional[str] = None
    org_id: Optional[str] = None
    deployment_name: Optional[str] = None

@dataclass
class APILimits:
    requests_per_minute: int
    tokens_per_minute: int
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 300.0

class APIRateLimit:
    def __init__(self, limits: APILimits):
        self.limits = limits
        self.request_count = 0
        self.token_count = 0
        self.last_reset = datetime.now()
        self.lock = asyncio.Lock()

    async def reset_if_needed(self) -> None:
        now = datetime.now()
        if (now - self.last_reset).total_seconds() >= 60:
            self.request_count = 0
            self.token_count = 0
            self.last_reset = now

    def is_rate_limited(self, estimated_tokens: int) -> bool:
        return (self.request_count >= self.limits.requests_per_minute or self.token_count + estimated_tokens > self.limits.tokens_per_minute)

    def calculate_wait_time(self) -> float:
        return max(0, 60 - (datetime.now() - self.last_reset).total_seconds())

class EnhancedAPIRateLimiter:
    def __init__(self):
        self.logger = LoggerSetup.get_logger("api_rate_limiter")
        self.limits = {
            "azure": APILimits(requests_per_minute=60, tokens_per_minute=90000),
            "openai": APILimits(requests_per_minute=50, tokens_per_minute=80000),
            "claude": APILimits(requests_per_minute=40, tokens_per_minute=100000)
        }
        self.rate_limiters = {service: APIRateLimit(limits) for service, limits in self.limits.items()}
        self.metrics_collector = APIMetricsCollector()

    async def check_and_wait(self, service: str, estimated_tokens: int) -> None:
        if service not in self.rate_limiters:
            raise ValueError(f"Unsupported service: {service}")
        rate_limiter = self.rate_limiters[service]
        limits = self.limits[service]
        async with rate_limiter.lock:
            await rate_limiter.reset_if_needed()

[Timestamp: 2024/11/11 15:45:50]

**user**: Here is the update needed for api_interaction class APIClient:
    """Enhanced API client with integrated metrics and code analysis."""
    
    def __init__(self):
        self.config = Config()
        self.monitor = Monitor()
        self.base_url = self.config.get('api_base_url')
        self.api_key = self.config.get('api_key')
        self.env_config = EnvironmentConfig()
        self.model_headers = ModelHeaders()
        self.rate_limiter = EnhancedAPIRateLimiter()
        self.metrics_collector = self.rate_limiter.metrics_collector
        self.azure = AzureOpenAIClient(self.env_config)
        
        # Initialize metrics and analysis components
        self.metrics_calculator = MetricsCalculator()
        self.code_analyzer = CodeAnalyzer()
        self.pattern_detector = PatternDetector(None)  # Will be set per analysis
        
    async def analyze_code_structure(
        self,
        code_content: str,
        code_type: str = "module",
        **analysis_params: Any
    ) -> Dict[str, Any]:
        """Analyze code structure with integrated metrics and pattern detection."""
        try:
            # Parse code into AST
            tree = ast.parse(code_content)
            self.pattern_detector = PatternDetector(tree)
            
            # Basic code analysis
            analysis_result = await self.code_analyzer.analyze(
                tree,
                code_content,
                analysis_params
            )
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all_metrics(tree, code_content)
            analysis_result['metrics'] = metrics
            
            # Detect patterns
            patterns = self.pattern_detector.detect_patterns()
            analysis_result['patterns'] = patterns
            
            # Record metrics
            await self.metrics_collector.record_analysis_metrics(
                code_type=code_type,
                analysis_size=len(code_content),
                metrics=metrics
            )
            
            # Enhance with AI insights
            enhanced_result = await self._enhance_analysis_with_ai(
                analysis_result,
                code_content,
                self._prepare_analysis_messages(analysis_result)
            )
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Code analysis failed: {e}")
            await self.metrics_collector.record_analysis_error(
                error_type="analysis_failure",
                error_message=str(e)
            )
            raise
            
    def _prepare_analysis_messages(
        self,
        analysis_result: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Prepare messages for AI enhancement based on analysis results."""
        messages = [
            {
                "role": "system",
                "content": "You are a code analysis assistant that provides insights "
                          "based on code metrics and patterns."
            }
        ]
        
        # Add metrics context
        if 'metrics' in analysis_result:
            metrics_summary = {
                "complexity": analysis_result['metrics'].get('complexity', {}),
                "maintainability": analysis_result['metrics'].get('maintainability', {}),
                "quality": analysis_result['metrics'].get('quality', {})
            }
            messages.append({
                "role": "assistant",
                "content": f"Code Metrics Analysis:\n{json.dumps(metrics_summary, indent=2)}"
            })
        
        # Add patterns context
        if 'patterns' in analysis_result:
            patterns_summary = {
                pattern_type: [p['type'] for p in patterns]
                for pattern_type, patterns in analysis_result['patterns'].items()
            }
            messages.append({
                "role": "assistant",
                "content": f"Detected Patterns:\n{json.dumps(patterns_summary, indent=2)}"
            })
            
        return messages
        
    async def record_api_metrics(
        self,
        operation: str,
        duration: float,
        status: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record API operation metrics."""
        try:
            await self.metrics_collector.record_api_call(
                service="api",
                endpoint=operation,
                tokens_used=metadata.get('tokens_used', 0) if metadata else 0,
                latency_ms=duration * 1000,
                status=status,
                error=metadata.get('error') if status == "error" else None
            )
        except Exception as e:
            logger.error(f"Failed to record API metrics: {e}")

    async def get_analysis_metrics(self) -> Dict[str, Any]:
        """Get aggregated analysis metrics."""
        try:
            api_metrics = await self.metrics_collector.get_service_summary("api")
            analysis_metrics = await self.metrics_collector.get_service_summary("analysis")
            
            return {
                "api": api_metrics,
                "analysis": analysis_metrics,
                "patterns": {
                    "detected_count": self.pattern_detector.get_pattern_counts()
                    if self.pattern_detector else {}
                }
            }
        except Exception as e:
            logger.error(f"Failed to get analysis metrics: {e}")
            return {}
[Timestamp: 2024/11/11 15:46:38]

**ai**: To update the `APIClient` class in `api_interaction.py` with the new features and methods you provided, we need to integrate the `MetricsCalculator`, `CodeAnalyzer`, and `PatternDetector` components, as well as enhance the code analysis process. Here's how you can incorporate these updates:

```python
# api_interaction.py

import os
import ast
import json
import aiohttp
import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from dotenv import load_dotenv
from core.logger import LoggerSetup, get_logger
from extract.code import CodeAnalysisTool  # Import from code.py
from openai import AzureOpenAI
from config import EnvironmentConfig, ModelHeaders
from core.config import Config
from core.monitoring import Monitor
from metrics import MetricsCalculator  # Assuming MetricsCalculator is in metrics.py
from pattern_detection import PatternDetector  # Assuming PatternDetector is in pattern_detection.py
from code_analysis import CodeAnalyzer  # Assuming CodeAnalyzer is in code_analysis.py

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

@dataclass
class APICredentials:
    api_key: str
    endpoint: Optional[str] = None
    org_id: Optional[str] = None
    deployment_name: Optional[str] = None

@dataclass
class APILimits:
    requests_per_minute: int
    tokens_per_minute: int
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 300.0

class APIRateLimit:
    def __init__(self, limits: APILimits):
        self.limits = limits
        self.request_count = 0
        self.token_count = 0
        self.last_reset = datetime.now()
        self.lock = asyncio.Lock()

    async def reset_if_needed(self) -> None:
        now = datetime.now()
        if (now - self.last_reset).total_seconds() >= 60:
            self.request_count = 0
            self.token_count = 0
            self.last_reset = now

    def is_rate_limited(self, estimated_tokens: int) -> bool:
        return (self.request_count >= self.limits.requests_per_minute or self.token_count + estimated_tokens > self.limits.tokens_per_minute)

    def calculate_wait_time(self) -> float:
        return max(0, 60 - (datetime.now() - self.last_reset).total_seconds())

class EnhancedAPIRateLimiter:
    def __init__(self):
        self.logger = LoggerSetup.get_logger("api_rate_limiter")
        self.limits = {
            "azure": APILimits(requests_per_minute=60, tokens_per_minute=90000),
            "openai": APILimits(requests_per_minute=50, tokens_per_minute=80000),
            "claude": APILimits(requests_per_minute=40, tokens_per_minute=100000)
        }
        self.rate_limiters = {service: APIRateLimit(limits) for service, limits in self.limits.items()}
        self.metrics_collector = APIMetricsCollector()

    async def check_and_wait(self, service: str, estimated_tokens: int) -> None:
        if service not in self.rate_limiters:
            raise ValueError(f"Unsupported service: {service}")
        rate_limiter = self.rate_limiters[service]
        limits = self.limits[service]
        async with rate_limiter.lock:
            await rate_limiter.reset_if_needed()
            if rate_limiter.is_rate_limited(estimated_tokens):
                wait_time = rate_limiter.calculate_wait_time()
                self.logger.info(f"Rate limit approaching for {service}, waiting {wait_time:.2f} seconds")
                await self.metrics_collector.record_rate_limit_event(service, wait_time, estimated_tokens)
                await asyncio.sleep(wait_time)
                await rate_limiter.reset_if_needed()
            rate_limiter.request_count += 1
            rate_limiter.token_count += estimated_tokens

    async def execute_with_retry(self, service: str, operation: callable, *args, **kwargs) -> Tuple[Any, Optional[Exception]]:
        limits = self.limits[service]
        for attempt in range(limits.max_retries):
            try:
                start_time = time.time()
                response = await operation(*args, **kwargs)
                elapsed_time = (time.time() - start_time) * 1000
                await self.metrics_collector.record_api_call(service=service, endpoint=kwargs.get('endpoint', 'unknown'), tokens_used=kwargs.get('estimated_tokens', 0), latency_ms=elapsed_time, status="success")
                return response, None
            except Exception as e:
                error_type = type(e).__name__
                self.logger.error(f"API call failed (attempt {attempt + 1}/{limits.max_retries}): {e}")
                elapsed_time = (time.time() - start_time) * 1000
                await self.metrics_collector.record_api_call(service=service, endpoint=kwargs.get('endpoint', 'unknown'), tokens_used=0, latency_ms=elapsed_time, status="error", error=str(e))
                if attempt < limits.max_retries - 1:
                    delay = min(limits.max_delay, limits.base_delay * (2 ** attempt))
                    self.logger.info(f"Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
                else:
                    return None, e

    async def get_usage_stats(self) -> Dict[str, Dict[str, Union[int, float]]]:
        stats = {}
        for service, limiter in self.rate_limiters.items():
            stats[service] = {
                "requests_used": limiter.request_count,
                "requests_limit": self.limits[service].requests_per_minute,
                "tokens_used": limiter.token_count,
                "tokens_limit": self.limits[service].tokens_per_minute,
                "time_until_reset": limiter.calculate_wait_time()
            }
        return stats

@dataclass
class APICallMetrics:
    timestamp: datetime
    service: str
    endpoint: str
    tokens_used: int
    latency_ms: float
    status: str
    error: Optional[str] = None

@dataclass
class ServiceMetrics:
    total_calls: int = 0
    total_tokens: int = 0
    total_errors: int = 0
    total_latency_ms: float = 0.0
    calls_per_minute: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))
    tokens_per_minute: DefaultDict[str, int] = field(default_factory=lambda: defaultdict(int))

class APIMetricsCollector:
    def __init__(self, retention_days: int = 7):
        self.logger = LoggerSetup.get_logger("api_metrics")
        self.retention_days = retention_days
        self.metrics: List[APICallMetrics] = []
        self.service_metrics: Dict[str, ServiceMetrics] = defaultdict(ServiceMetrics)
        self._lock = asyncio.Lock()

    async def record_api_call(self, service: str, endpoint: str, tokens_used: int, latency_ms: float, status: str, error: Optional[str] = None) -> None:
        async with self._lock:
            try:
                metric = APICallMetrics(timestamp=datetime.now(), service=service, endpoint=endpoint,
[Timestamp: 2024/11/11 15:46:59]
