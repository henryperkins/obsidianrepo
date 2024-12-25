### Gaps and Integration Opportunities

1. **Metrics Integration with AI Interaction**
```python
# ai_interaction.py
class AIInteractionHandler:
    def __init__(self, config: AzureOpenAIConfig, ...):
        # Add metrics analyzer
        self.metrics_analyzer = Metrics()  # Currently missing

    async def process_code(self, source_code: str) -> Tuple[str, str]:
        try:
            # Add code quality metrics to metadata
            tree = ast.parse(source_code)
            code_metrics = {
                'maintainability_index': self.metrics_analyzer.calculate_maintainability_index(tree),
                'dependencies': self.metrics_analyzer.analyze_dependencies(tree),
                'overall_complexity': self.metrics_analyzer.calculate_complexity(tree)
            }
            
            # Include metrics in the extraction process
            extractor = ExtractionManager()
            metadata = extractor.extract_metadata(source_code, metrics=code_metrics)
```

2. **Documentation Enhancement with Metrics**
```python
# markdown_generator.py
class MarkdownDocumentationGenerator:
    def __init__(self, source_code: str, module_path: Optional[str] = None):
        self.metrics = Metrics()
        self.code_metrics = self._calculate_metrics(source_code)

    def _calculate_metrics(self, source_code: str) -> Dict[str, Any]:
        tree = ast.parse(source_code)
        return {
            'maintainability': self.metrics.calculate_maintainability_index(tree),
            'complexity': self.metrics.calculate_complexity(tree),
            'dependencies': self.metrics.analyze_dependencies(tree),
            'halstead': self.metrics.calculate_halstead_metrics(tree)
        }

    def generate_markdown(self) -> str:
        sections = [
            self._generate_header(),
            self._generate_overview(),
            self._generate_metrics_section(),  # New section
            self._generate_classes_section(),
            self._generate_functions_section(),
        ]
        return "\n\n".join(filter(None, sections))

    def _generate_metrics_section(self) -> str:
        """Generate metrics section with code quality insights."""
        return f"""## Code Quality Metrics

### Maintainability
- Index: {self.code_metrics['maintainability']}/100
- Status: {self._get_maintainability_status(self.code_metrics['maintainability'])}

### Complexity
- Overall Complexity: {self.code_metrics['complexity']}
- Halstead Volume: {self.code_metrics['halstead']['program_volume']:.2f}

### Dependencies
- Standard Library: {', '.join(self.code_metrics['dependencies']['stdlib'])}
- Third Party: {', '.join(self.code_metrics['dependencies']['third_party'])}
- Local: {', '.join(self.code_metrics['dependencies']['local'])}
"""
```

3. **Extraction Manager Enhancement**
```python
# extraction_manager.py
class ExtractionManager:
    def __init__(self):
        self.metrics = Metrics()

    def extract_metadata(self, source_code: str, metrics: Optional[Dict] = None) -> Dict[str, Any]:
        tree = ast.parse(source_code)
        
        # Calculate metrics if not provided
        if not metrics:
            metrics = {
                'maintainability': self.metrics.calculate_maintainability_index(tree),
                'complexity': self.metrics.calculate_complexity(tree),
                'dependencies': self.metrics.analyze_dependencies(tree)
            }

        # Enhanced metadata extraction with metrics
        metadata = {
            'classes': self._extract_classes_with_metrics(tree),
            'functions': self._extract_functions_with_metrics(tree),
            'metrics': metrics
        }
        return metadata

    def _extract_functions_with_metrics(self, tree: ast.AST) -> List[Dict[str, Any]]:
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_metrics = {
                    'cyclomatic_complexity': self.metrics.calculate_cyclomatic_complexity(node),
                    'cognitive_complexity': self.metrics.calculate_cognitive_complexity(node),
                    'maintainability': self.metrics.calculate_maintainability_index(node)
                }
                functions.append({
                    'name': node.name,
                    'metrics': function_metrics,
                    # ... other existing metadata
                })
        return functions
```

4. **Monitoring Integration**
```python
# monitoring.py
class MetricsCollector:
    def __init__(self):
        self.code_metrics = Metrics()
        self.metrics_history = defaultdict(list)

    def track_code_quality(self, source_code: str, context: str) -> None:
        """Track code quality metrics over time."""
        try:
            tree = ast.parse(source_code)
            metrics = {
                'timestamp': datetime.now(),
                'context': context,
                'maintainability': self.code_metrics.calculate_maintainability_index(tree),
                'complexity': self.code_metrics.calculate_complexity(tree),
                'dependencies': self.code_metrics.analyze_dependencies(tree)
            }
            self.metrics_history[context].append(metrics)
        except Exception as e:
            logger.error(f"Error tracking code quality: {e}")

    def get_code_quality_trends(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get trends in code quality metrics over time."""
        return dict(self.metrics_history)
```

5. **Configuration Enhancement**
```python
# config.py
@dataclass
class MetricsConfig:
    """Configuration for code metrics analysis."""
    complexity_threshold: int = 10
    maintainability_threshold: float = 65.0
    enable_dependency_analysis: bool = True
    track_history: bool = True
    history_retention_days: int = 30

@dataclass
class AzureOpenAIConfig(AIModelConfig):
    metrics_config: MetricsConfig = field(default_factory=MetricsConfig)
```

6. **CLI Integration**
```python
# main.py
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate documentation using AI.")
    # ... existing arguments ...
    parser.add_argument("--analyze-metrics", action="store_true", 
                       help="Include code quality metrics in documentation")
    parser.add_argument("--metrics-threshold", type=float, default=65.0,
                       help="Maintainability index threshold")
    return parser.parse_args()

async def run_workflow(args: argparse.Namespace) -> None:
    try:
        config = await load_and_validate_config(args)
        if args.analyze_metrics:
            config.metrics_config.maintainability_threshold = args.metrics_threshold

        async with initialize_components(config) as components:
            metrics_collector = components.get("metrics_collector")
            orchestrator = WorkflowOrchestrator(config, metrics_collector)
            results = await orchestrator.run(args.source_path, args.output_dir)
            
            if args.analyze_metrics:
                await generate_metrics_report(results, args.output_dir)
```

7. **Report Generation**
```python
# Add new file: metrics_reporter.py
class MetricsReporter:
    """Generates comprehensive code quality reports."""
    
    def __init__(self, metrics: Metrics):
        self.metrics = metrics

    async def generate_report(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Generate a detailed metrics report."""
        report = {
            'summary': self._generate_summary(results),
            'trends': self._analyze_trends(results),
            'recommendations': self._generate_recommendations(results)
        }
        
        # Generate HTML report
        await self._save_html_report(report, output_dir / 'metrics_report.html')
        
        # Generate JSON data
        await self._save_json_data(report, output_dir / 'metrics_data.json')
```

### Integration Opportunities

1. **Quality Gates**:
```python
class QualityGate:
    def __init__(self, metrics_config: MetricsConfig):
        self.config = metrics_config
        self.metrics = Metrics()

    def check_quality(self, source_code: str) -> Tuple[bool, List[str]]:
        """Check if code meets quality standards."""
        tree = ast.parse(source_code)
        issues = []
        
        # Check maintainability
        maintainability = self.metrics.calculate_maintainability_index(tree)
        if maintainability < self.config.maintainability_threshold:
            issues.append(f"Maintainability index {maintainability} below threshold")

        # Check complexity
        complexity = self.metrics.calculate_complexity(tree)
        if complexity > self.config.complexity_threshold:
            issues.append(f"Complexity {complexity} exceeds threshold")

        return len(issues) == 0, issues
```

2. **Metrics History**:
```python
class MetricsHistory:
    """Tracks and analyzes code metrics over time."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.metrics = Metrics()

    async def store_metrics(self, file_path: str, metrics_data: Dict[str, Any]):
        """Store metrics data with timestamp."""
        # Implementation for storing metrics history

    async def get_trends(self, file_path: str, days: int = 30) -> Dict[str, List]:
        """Get metrics trends for a file over time."""
        # Implementation for retrieving and analyzing trends
```

3. **Automated Recommendations**:
```python
class MetricsOptimizer:
    """Provides recommendations for improving code quality."""
    
    def analyze_and_recommend(self, source_code: str) -> List[str]:
        """Analyze code and provide improvement recommendations."""
        tree = ast.parse(source_code)
        recommendations = []
        
        # Add recommendations based on metrics
        maintainability = self.metrics.calculate_maintainability_index(tree)
        if maintainability < 65:
            recommendations.append("Consider breaking down complex functions")
            
        return recommendations
```