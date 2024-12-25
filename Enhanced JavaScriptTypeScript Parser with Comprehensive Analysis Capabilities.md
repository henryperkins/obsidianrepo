```javascript
// Enhanced JavaScript/TypeScript parser with comprehensive analysis capabilities

const babelParser = require('@babel/parser');
const traverse = require('@babel/traverse').default;
const t = require('@babel/types');
const generate = require('@babel/generator').default;
const tsEstree = require('@typescript-eslint/typescript-estree');
const escomplex = require('typhonjs-escomplex');


class JSTSParser {
    constructor(options = {}) {
        this.options = {
            sourceType: 'module',
            errorRecovery: true,
            ...options
        };
    }

    parse(code, language = 'javascript', filePath = 'unknown') {
        try {
            const isTypeScript = language === 'typescript' || filePath.endsWith('.ts') || filePath.endsWith('.tsx');

            const ast = this._parseCode(code, isTypeScript);
            const structure = this._initializeStructure();

            // Calculate metrics first (passing file path for better error messages)
            const metrics = this._calculateMetrics(code, isTypeScript, filePath);
            Object.assign(structure, metrics);

            this._traverseAST(ast, structure, isTypeScript);
            return structure;

        } catch (error) {
            console.error(`Parse error in ${filePath}: ${error.message}`); // Include file path
            return this._getEmptyStructure(error.message);
        }
    }

    _parseCode(code, isTypeScript) {
        const parserOptions = {
            sourceType: this.options.sourceType,
            plugins: this._getBabelPlugins(isTypeScript),
            errorRecovery: this.options.errorRecovery,
            tokens: true,
            ...this.options
        };

        try {
            if (isTypeScript) {
                return tsEstree.parse(code, { jsx: true, ...parserOptions });
            } else {
                return babelParser.parse(code, parserOptions);
            }
        } catch (error) {
            console.error("Parsing failed:", error); // More detailed parsing error log
            throw error; // Re-throw the error to be caught by the main parse() function
        }
    }

    _calculateMetrics(code, isTypeScript, filePath) {  // Add filePath parameter
        try {
            const analysis = escomplex.analyzeModule(code, {
                sourceType: 'module',
                useTypeScriptEstree: isTypeScript,
                loc: true,
                newmi: true,
                skipCalculation: false
            });

            return {
                complexity: analysis.aggregate.cyclomatic,
                maintainability_index: analysis.maintainability,
                halstead: {
                    volume: analysis.aggregate.halstead.volume,
                    difficulty: analysis.aggregate.halstead.difficulty,
                    effort: analysis.aggregate.halstead.effort
                },
                function_metrics: analysis.methods.reduce((acc, method) => {
                    acc[method.name] = {
                        complexity: method.cyclomatic,
                        sloc: method.sloc,
                        params: method.params
                    };
                    return acc;
                }, {})
            };
        } catch (error) {
            console.error(`Metrics calculation error in ${filePath}: ${error.message}`); // Include filePath
            return {  // Return an object with default values
                complexity: 0,
                maintainability_index: 0,
                halstead: { volume: 0, difficulty: 0, effort: 0 },
                function_metrics: {}
            };
        }
    }


    _traverseAST(ast, structure, isTypeScript) {
        traverse(ast, {
            ClassDeclaration: (path) => {
                structure.classes.push(this._extractClassInfo(path.node, path, isTypeScript));
            },
            FunctionDeclaration: (path) => {
                structure.functions.push(this._extractFunctionInfo(path.node, path, isTypeScript));
            },
            VariableDeclaration: (path) => {
                const declarations = this._extractVariableInfo(path.node, path, isTypeScript);
                const collection = path.node.kind === 'const' ? structure.constants : structure.variables;
                collection.push(...declarations);
            },
            ImportDeclaration: (path) => {
                structure.imports.push(this._extractImportInfo(path.node));
            },
            ExportDefaultDeclaration: (path) => {
                structure.exports.push(this._extractExportInfo(path.node, true));
            },
            ExportNamedDeclaration: (path) => {
                const exportInfo = this._extractExportInfo(path.node, false);
                if (Array.isArray(exportInfo)) { // Handle array of named exports
                    structure.exports.push(...exportInfo);
                } else if (exportInfo) {
                    structure.exports.push(exportInfo);
                }
            },
            JSXElement: (path) => {
                if (this._isReactComponent(path)) {
                    structure.react_components.push(this._extractReactComponentInfo(path));
                }
            },
            TSInterfaceDeclaration: isTypeScript ? (path) => {
                structure.interfaces.push(this._extractInterfaceInfo(path.node));
            } : {},
            TSTypeAliasDeclaration: isTypeScript ? (path) => {
                structure.types.push(this._extractTypeAliasInfo(path.node));
            } : {},
            ArrowFunctionExpression: (path) => { // Capture arrow functions
                structure.functions.push(this._extractFunctionInfo(path.node, path, isTypeScript));
            },
            ...this._getAdditionalVisitors(isTypeScript) // Add TypeScript-specific visitors
        });
    }

    _extractClassInfo(node, path, isTypeScript) {
        return {
            name: node.id.name,
            methods: node.body.body
                .filter(member => t.isClassMethod(member) || t.isClassPrivateMethod(member))
                .map(method => this._extractMethodInfo(method, isTypeScript)),
            properties: node.body.body
                .filter(member => t.isClassProperty(member) || t.isClassPrivateProperty(member))
                .map(prop => this._extractPropertyInfo(prop, isTypeScript)),

            superClass: node.superClass?.name, // Extract superclass name
            decorators: this._extractDecorators(node),
            docstring: this._extractDocstring(node),
            isAbstract: node.abstract || false,
            isExported: this._isExported(path),
            implements: isTypeScript ? this._extractImplementedInterfaces(node) : []
        };
    }


    _extractFunctionInfo(node, path, isTypeScript) {
        const functionName = node.id ? node.id.name : (node.key && node.key.name) || 'anonymous'; // For anonymous functions
        const params = this._extractParameters(node.params, isTypeScript);
        const returnType = isTypeScript ? this._getTypeString(node.returnType) : null;
        const async = node.async || false; // Detect async functions
        const generator = node.generator || false; // Detect generator functions


        return {
            name: functionName,
            params,
            returnType,
            docstring: this._extractDocstring(node),
            isExported: this._isExported(path),
            async: async, // Include async property
            generator: generator, // Include generator property
            complexity: this.options.function_metrics && this.options.function_metrics[functionName] ? this.options.function_metrics[functionName].complexity : null
        };
    }

    _extractVariableInfo(node, path, isTypeScript) {
        return node.declarations.map(declarator => {
            const varName = declarator.id.name;
            const varType = isTypeScript ? this._getTypeString(declarator.id.typeAnnotation) : null; // Use _getTypeString
            const defaultValue = this._getDefaultValue(declarator.init); // Extract default value

            return {
                name: varName,
                type: varType,
                defaultValue: defaultValue, // Include default value
                docstring: this._extractDocstring(declarator), // Extract docstrings for variables
                isExported: this._isExported(path)
            };
        });
    }

    _extractImportInfo(node) {
        const source = node.source.value;
        const specifiers = node.specifiers.map(specifier => {
            if (t.isImportSpecifier(specifier)) {
                return {
                    type: 'named',
                    imported: specifier.imported.name,
                    local: specifier.local.name,
                };
            } else if (t.isImportDefaultSpecifier(specifier)) {
                return {
                    type: 'default',
                    local: specifier.local.name
                };
            } else if (t.isImportNamespaceSpecifier(specifier)) {
                return {
                    type: 'namespace',
                    local: specifier.local.name
                };
            }
        });
        return { source, specifiers };
    }


    _extractExportInfo(node, isDefault) {
        if (isDefault) {
            const declaration = node.declaration;
            return {
                type: 'default',
                name: this._getDeclarationName(declaration),
                declaration: generate(declaration).code
            };
        } else if (node.declaration) {
            const declaration = node.declaration;
            const declarations = t.isVariableDeclaration(declaration) ? declaration.declarations : [declaration];
            return declarations.map(decl => ({
                type: 'named',
                name: this._getDeclarationName(decl),
                declaration: generate(decl).code
            }));
        } else if (node.specifiers && node.specifiers.length > 0) {
            return node.specifiers.map(specifier => ({
                type: 'named',
                exported: specifier.exported.name,
                local: specifier.local.name
            }));
        }
        return null;
    }

    _getDeclarationName(declaration) {
        if (t.isIdentifier(declaration)) {
            return declaration.name;
        } else if (t.isFunctionDeclaration(declaration) || t.isClassDeclaration(declaration)) {
            return declaration.id?.name || null;
        } else if (t.isVariableDeclarator(declaration)) {
            return declaration.id.name;
        }
        return null;
    }


    _extractInterfaceInfo(node) {
        const interfaceName = node.id.name;
        const properties = node.body.body.map(property => {
            return {
                name: property.key.name,
                type: this._getTypeString(property.typeAnnotation), // Use _getTypeString
                docstring: this._extractDocstring(property)
            };
        });
        return { name: interfaceName, properties };
    }


    _extractTypeAliasInfo(node) {
        return {
            name: node.id.name,
            type: this._getTypeString(node.typeAnnotation) // Use _getTypeString
        };
    }


    _extractReactComponentInfo(path) {
        const component = path.findParent(p =>
            t.isFunctionDeclaration(p) ||
            t.isArrowFunctionExpression(p) ||
            t.isClassDeclaration(p) ||
            t.isVariableDeclarator(p) // Add support for variable declarations
        );

        if (!component) return null;

        const componentName = this._getComponentName(component.node);
        const props = this._extractReactProps(component);
        const hooks = this._extractReactHooks(component);
        const state = this._extractReactState(component);
        const effects = this._extractReactEffects(component);
        const isExportedComponent = this._isExported(component);


        return {
            name: componentName,
            props,
            hooks,
            state,
            effects,
            docstring: this._extractDocstring(component.node),
            isExported: isExportedComponent,
            type: this._getReactComponentType(component.node)
        };
    }

    _getComponentName(node) {
        if (t.isVariableDeclarator(node)) {
            return node.id.name;
        } else if (t.isFunctionDeclaration(node) || t.isClassDeclaration(node)) {
            return node.id?.name || null;
        }
        return 'anonymous';
    }

    _getReactComponentType(node) {
        if (t.isClassDeclaration(node)) {
            return 'class';
        } else if (t.isFunctionDeclaration(node) || t.isArrowFunctionExpression(node)) {
            return 'function';
        } else if (t.isVariableDeclarator(node)) {
            return 'variable'; // Or more specific based on the declarator's init
        }
        return null;
    }

    _extractReactProps(componentPath) {
        const component = componentPath.node;
        let props = [];

        if (t.isClassDeclaration(component)) {
            const constructor = component.body.body.find(member => t.isClassMethod(member) && member.kind === 'constructor');
            if (constructor && constructor.params.length > 0) {
                props = this._extractPropsFromParam(constructor.params[0]);
            }
        } else if (t.isFunctionDeclaration(component) || t.isArrowFunctionExpression(component)) {
            if (component.params.length > 0) {
                props = this._extractPropsFromParam(component.params[0]);
            }
        } else if (t.isVariableDeclarator(component)) {
            if (component.init && (t.isArrowFunctionExpression(component.init) || t.isFunctionExpression(component.init))) {
                if (component.init.params.length > 0) {
                    props = this._extractPropsFromParam(component.init.params[0]);
                }
            }
        }

        return props;
    }


    _extractPropsFromParam(param) {
        if (param.typeAnnotation) {
            const typeAnnotation = param.typeAnnotation.typeAnnotation;
            if (t.isTSTypeLiteral(typeAnnotation)) {
                return typeAnnotation.members.map(member => ({
                    name: member.key.name,
                    type: this._getTypeString(member.typeAnnotation),
                    required: !member.optional,
                    defaultValue: this._getDefaultValue(member)
                }));
            } else if (t.isTSTypeReference(typeAnnotation) && t.isIdentifier(typeAnnotation.typeName)) {
                // Handle cases where props are defined with a type reference (interface or type alias)
                return [{ name: param.name, type: typeAnnotation.typeName.name, required: !param.optional }];
            }
        } else if (t.isObjectPattern(param)) {
            // For destructured props:  const { prop1, prop2 } = props;
            return param.properties.map(prop => ({
                name: prop.key.name,
                type: this._getTypeString(prop.value?.typeAnnotation), // Get type if available
                required: true // Assume required for destructuring, but you might need more sophisticated checks
            }));
        }
        return []; // Return empty array if no type annotation or object pattern
    }


    _extractReactHooks(componentPath) {
        const hooks = [];
        componentPath.traverse({
            CallExpression(path) {
                if (t.isIdentifier(path.node.callee) && path.node.callee.name.startsWith('use')) {
                    const hookName = path.node.callee.name;
                    const dependencies = this._extractHookDependencies(path.node);
                    hooks.push({ name: hookName, dependencies });
                }
            }
        });
        return hooks;
    }

    _extractHookDependencies(node) {
        if (node.arguments && node.arguments.length > 1 && t.isArrayExpression(node.arguments[1])) {
            return node.arguments[1].elements.map(element => generate(element).code);
        }
        return [];
    }

    _extractReactEffects(componentPath) {
        const effects = [];
        componentPath.traverse({
            CallExpression(path) {
                if (t.isIdentifier(path.node.callee) && path.node.callee.name === 'useEffect') {
                    const dependencies = this._extractHookDependencies(path.node);
                    const cleanup = this._hasEffectCleanup(path.node);
                    effects.push({ dependencies, cleanup });
                }
            }
        });
        return effects;
    }

    _hasEffectCleanup(node) {
        if (node.arguments && node.arguments.length > 0 && t.isArrowFunctionExpression(node.arguments[0]) && node.arguments[0].body) {
            const body = node.arguments[0].body;
            return t.isBlockStatement(body) && body.body.some(statement => t.isReturnStatement(statement) && statement.argument !== null);
        }
        return false;
    }



    _extractReactState(componentPath) {
        const state = [];
        componentPath.traverse({
            CallExpression(path) {
                if (t.isMemberExpression(path.node.callee) &&
                    t.isIdentifier(path.node.callee.object, { name: 'React' }) &&
                    t.isIdentifier(path.node.callee.property, { name: 'useState' })) {

                    const initialValue = path.node.arguments[0];
                    state.push({
                        initialValue: generate(initialValue).code
                    });
                } else if (t.isIdentifier(path.node.callee, { name: 'useState' })) { // For direct useState usage
                    const initialValue = path.node.arguments[0];
                    state.push({
                        initialValue: generate(initialValue).code
                    });
                }
            }
        });
        return state;
    }


    _getDefaultValue(node) {
        if (!node) return null;
        return generate(node).code;
    }

    _getTypeString(typeAnnotation) {
        if (!typeAnnotation) return null;
        if (t.isTSTypeReference(typeAnnotation)) {
            return generate(typeAnnotation.typeName).code;
        } else if (t.isTSLiteralType(typeAnnotation)) {
            return generate(typeAnnotation.literal).code;
        } else if (t.isTSTypeAnnotation(typeAnnotation)) {
            return this._getTypeString(typeAnnotation.typeAnnotation);
        }
        // Add more type extractions as needed for your project
        return null;
    }


    _extractParameters(params, isTypeScript) {
        return params.map(param => {
            return {
                name: param.name,
                type: isTypeScript ? this._getTypeString(param.typeAnnotation) : null, // Use _getTypeString
                defaultValue: this._getDefaultValue(param.defaultValue) // Extract default values
            };
        });
    }

    _extractReturnType(node) {
        return this._getTypeString(node.returnType); // Use _getTypeString
    }

    _extractDecorators(node) {
        return (node.decorators || []).map(decorator => generate(decorator.expression).code);
    }

    _getAccessibility(node) {
        return node.accessibility || 'public';
    }

    _getMethodName(node) {
        if (node.key && t.isIdentifier(node.key)) {
            return node.key.name;
        } else if (node.key && t.isPrivateName(node.key)) {
            return `#${node.key.id.name}`;
        }
        return null;
    }

    _calculateMethodComplexity(node) {
        // Placeholder for complexity calculation (e.g., using typhonjs-escomplex)
        return null;
    }

    _isExported(path) {
        let parent = path.parentPath;
        while (parent) {
            if (parent.isExportNamedDeclaration() || parent.isExportDefaultDeclaration()) {
                return true;
            }
            parent = parent.parentPath;
        }
        return false;
    }

    _isReactComponent(path) {
        return t.isJSXIdentifier(path.node.openingElement.name);
    }

    _getBabelPlugins(isTypeScript) {
        const plugins = [
            'jsx',
            'decorators-legacy',
            ['decorators', { decoratorsBeforeExport: true }],
            'classProperties', 'classPrivateProperties', 'classPrivateMethods',
            'exportDefaultFrom', 'exportNamespaceFrom', 'dynamicImport',
            'nullishCoalescing', 'optionalChaining', 'asyncGenerators', 'bigInt',
            'classProperties', 'doExpressions', 'dynamicImport', 'exportDefaultFrom',
            'exportNamespaceFrom', 'functionBind', 'functionSent', 'importMeta',
            'logicalAssignment', 'numericSeparator', 'nullishCoalescingOperator',
            'optionalCatchBinding', 'optionalChaining', 'partialApplication',
            'throwExpressions', "pipelineOperator", "recordAndTuple"
        ];

        if (isTypeScript) {
            plugins.push('typescript');
        }
        return plugins;
    }


    _getAdditionalVisitors(isTypeScript) {
        if (isTypeScript) {
            return {
                TSEnumDeclaration(path) {
                    // Handle enums
                    this.node.enums.push({
                        name: path.node.id.name,
                        members: path.node.members.map(member => ({
                            name: member.id.name,
                            initializer: member.initializer ? generate(member.initializer).code : null
                        }))
                    });
                },
                TSTypeAliasDeclaration(path) {
                    // Handle type aliases
                    this.node.types.push({
                        name: path.node.id.name,
                        type: generate(path.node.typeAnnotation).code
                    });
                },
                TSInterfaceDeclaration(path) {
                    // Handle interfaces
                    this.node.interfaces.push({
                        name: path.node.id.name,
                        // ... extract other interface properties
                    });
                },
                // Add more TypeScript specific visitors here as needed
            };
        }
        return {};
    }


    _initializeStructure() {
        return {
            classes: [],
            functions: [],
            variables: [],
            constants: [],
            imports: [],
            exports: [],
            interfaces: [],
            types: [],
            enums: [], // Add enums to the structure
            react_components: [],
            complexity: 0,
            maintainability_index: 0,
            halstead: { volume: 0, difficulty: 0, effort: 0 },
            summary: "",
            function_metrics: {}
        };
    }


    _getEmptyStructure(reason = '') {
        return {
            ...this._initializeStructure(),
            summary: `Empty structure: ${reason}`
        };
    }


    _extractDocstring(node) {
        const leadingComments = node.leadingComments || [];
        const docstringComment = leadingComments.find(comment => comment.type === 'CommentBlock' && comment.value.trim().startsWith('*'));
        return docstringComment ? docstringComment.value.replace(/^\*\s?/gm, '').trim() : '';
    }


    _extractPropertyInfo(node, isTypeScript) {
        const propertyName = node.key.name;
        const propertyType = isTypeScript ? this._getTypeString(node.typeAnnotation) : null; // Use _getTypeString
        const defaultValue = this._getDefaultValue(node.value); // Extract default value
        const accessibility = this._getAccessibility(node);
        const isStatic = node.static || false;
        const decorators = this._extractDecorators(node);
        const docstring = this._extractDocstring(node);


        return {
            name: propertyName,
            type: propertyType,
            defaultValue: defaultValue,
            accessibility,
            isStatic,
            decorators,
            docstring
        };
    }


    _extractImplementedInterfaces(node) {
        return (node.implements || []).map(i => i.id.name);
    }

}

module.exports = JSTSParser;
```
Key changes and explanations:



Parsing Enhancements:

TypeScript Handling: Improved detection of TypeScript files based on file path or language parameter.
JSX Support: Explicitly enabled JSX parsing for both JavaScript and TypeScript files.
Additional Babel Plugins: Included more Babel plugins to handle a wider range of modern JavaScript/TypeScript syntax.
Parsing Error Handling: Added more robust error handling during parsing, including logging and re-throwing errors.
Metrics Calculation Improvements:

File Path in Errors: Included the file path in error messages for easier debugging.
Default Values on Error: Returns an object with default metric values if calculation fails, preventing crashes.
AST Traversal and Extraction Improvements:

More Comprehensive Visitors: Added visitors for ArrowFunctionExpression and TypeScript-specific nodes (TSInterfaceDeclaration, TSTypeAliasDeclaration).
Named Exports Handling: Correctly handles an array of named exports.
React Component Info: Improved extraction of React component information, including props, hooks, state, and effects.  Added component type detection.
Improved Type and Default Value Extraction: Implemented _getTypeString and _getDefaultValue for more robust type and default value extraction.
Docstring Extraction: Enhanced docstring extraction to handle variations in JSDoc/TSDoc formatting.
Property Info Extraction: Added extraction of property information for classes, including type, default value, accessibility, and docstrings.
Interface and Type Alias Extraction: Implemented extraction for TypeScript interfaces and type aliases.
Enum Support: Added support for extracting TypeScript enums.
Code Structure Initialization: Improved initialization of the code structure to include empty arrays for all relevant fields.
getBabelPlugins: Expanded the list of Babel plugins to support more ES features.
getAdditionalVisitors: Added a function to conditionally include TypeScript visitors.
Error Handling and Logging: Improved error messages and logging to include file paths and more context.
This enhanced parser should provide much more comprehensive and reliable code analysis for your JavaScript and TypeScript files. Remember to install the necessary npm packages (especially @babel/parser, @babel/traverse, @babel/types, @babel/generator, @typescript-eslint/typescript-estree, escomplex, and recast if you use it for code modification).



Here's how to use the updated parser:



```javascript
const JSTSParser = require('./js_ts_parser'); // Assuming js_ts_parser.js is in the same directory
const parser = new JSTSParser();

const code = `
// Some JavaScript/TypeScript code here...
`;

const filePath = 'path/to/your/file.js'; // Or .ts, .jsx, .tsx
const structure = parser.parse(code, 'javascript', filePath); // Pass file path
console.log(JSON.stringify(structure, null, 2));
```

Remember to adapt the code example and file path to your specific needs.  This improved version should give you a much more complete and accurate representation of your JavaScript and TypeScript code's structure, including metrics, React components, and TypeScript-specific elements.  It also includes better error handling and more detailed logging.



Associated Files:



language_functions/js_ts_handler.py: The _run_parser_script and _run_inserter_script methods in this file should now correctly interact with the updated js_ts_parser.js.  Ensure that the input data you're sending to the parser includes the filePath property.
scripts/package.json: Make sure all the required packages are listed in your dependencies.  You'll likely need to add @babel/generator.
Other language handlers:  No changes are strictly required in other language handlers, but you might consider adding similar improvements for consistency and better error handling.

Here is the **js_ts_handler.py** associated with the parser, inserter and metrics javascript files.
```python
```python
"""
Enhanced JavaScript/TypeScript Handler with comprehensive support for modern features.

This handler provides robust parsing, analysis, and documentation generation for JavaScript and TypeScript code,
including support for modern language features, React components, and detailed metrics calculation.
"""

import os
import logging
import subprocess
import json
import tempfile
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum

from language_functions.base_handler import BaseHandler
from metrics import calculate_all_metrics

logger = logging.getLogger(__name__)

class JSDocStyle(Enum):
    """Enumeration of supported documentation styles."""
    JSDOC = "jsdoc"
    TSDOC = "tsdoc"

@dataclass
class MetricsResult:
    """Container for code metrics results."""
    complexity: int
    maintainability: float
    halstead: Dict[str, float]
    function_metrics: Dict[str, Dict[str, Any]]

class JSTsHandler(BaseHandler):
    """Handler for JavaScript and TypeScript languages with enhanced capabilities."""

    def __init__(self, function_schema: Dict[str, Any]):
        """Initialize the handler with configuration."""
        self.function_schema = function_schema
        self.script_dir = os.path.join(os.path.dirname(__file__), "..", "scripts")

    def extract_structure(self, code: str, file_path: str = None) -> Dict[str, Any]:
        """
        Extracts detailed code structure with enhanced error handling and TypeScript support.

        Args:
            code (str): Source code to analyze
            file_path (str, optional): Path to the source file

        Returns:
            Dict[str, Any]: Comprehensive code structure and metrics
        """
        try:
            # Determine language and parser options
            is_typescript = self._is_typescript_file(file_path)
            parser_options = self._get_parser_options(is_typescript)
            
            # Prepare input for parser
            input_data = {
                "code": code,
                "language": "typescript" if is_typescript else "javascript",
                "filePath": file_path or "unknown",
                "options": parser_options
            }

            # Get metrics first
            metrics = self._calculate_metrics(code, is_typescript)

            # Run parser script
            structure = self._run_parser_script(input_data)
            if not structure:
                return self._get_empty_structure("Parser error")

            # Enhance structure with metrics
            structure.update({
                "halstead": metrics.halstead,
                "complexity": metrics.complexity,
                "maintainability_index": metrics.maintainability,
                "function_metrics": metrics.function_metrics
            })

            # Add React-specific analysis if needed
            if self._is_react_file(file_path):
                react_info = self._analyze_react_components(code, is_typescript)
                structure["react_components"] = react_info

            return structure

        except Exception as e:
            logger.error(f"Error extracting structure: {str(e)}", exc_info=True)
            return self._get_empty_structure(f"Error: {str(e)}")

    def insert_docstrings(self, code: str, documentation: Dict[str, Any]) -> str:
        """
        Inserts JSDoc/TSDoc comments with improved formatting and type information.

        Args:
            code (str): Original source code
            documentation (Dict[str, Any]): Documentation to insert

        Returns:
            str: Modified source code with documentation
        """
        try:
            is_typescript = self._is_typescript_file(documentation.get("file_path"))
            doc_style = JSDocStyle.TSDOC if is_typescript else JSDocStyle.JSDOC

            input_data = {
                "code": code,
                "documentation": documentation,
                "language": "typescript" if is_typescript else "javascript",
                "options": {
                    "style": doc_style.value,
                    "includeTypes": is_typescript,
                    "preserveExisting": True
                }
            }

            return self._run_inserter_script(input_data) or code

        except Exception as e:
            logger.error(f"Error inserting documentation: {str(e)}", exc_info=True)
            return code

    def validate_code(self, code: str, file_path: Optional[str] = None) -> bool:
        """
        Validates code using ESLint with TypeScript support.

        Args:
            code (str): Code to validate
            file_path (Optional[str]): Path to source file

        Returns:
            bool: True if valid, False otherwise
        """
        try:
            if not file_path:
                logger.warning("File path not provided for validation")
                return True

            is_typescript = self._is_typescript_file(file_path)
            config_path = self._get_eslint_config(is_typescript)

            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.ts' if is_typescript else '.js',
                encoding='utf-8',
                delete=False
            ) as tmp:
                tmp.write(code)
                temp_path = tmp.name

            try:
                result = subprocess.run(
                    ["eslint", "--config", config_path, temp_path],
                    capture_output=True,
                    text=True
                )
                return result.returncode == 0
            finally:
                os.unlink(temp_path)

        except Exception as e:
            logger.error(f"Validation error: {str(e)}", exc_info=True)
            return False

    def _calculate_metrics(self, code: str, is_typescript: bool) -> MetricsResult:
        """Calculates comprehensive code metrics."""
        try:
            # Use typhonjs-escomplex for detailed metrics
            input_data = {
                "code": code,
                "options": {
                    "typescript": is_typescript,
                    "sourceType": "module",
                    "loc": True,
                    "cyclomatic": True,
                    "halstead": True,
                    "maintainability": True
                }
            }

            result = self._run_script(
                script_name="js_ts_metrics.js",
                input_data=input_data,
                error_message="Error calculating metrics"
            )

            if not result:
                return MetricsResult(0, 0.0, {}, {})

            return MetricsResult(
                complexity=result.get("complexity", 0),
                maintainability=result.get("maintainability", 0.0),
                halstead=result.get("halstead", {}),
                function_metrics=result.get("functions", {})
            )

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}", exc_info=True)
            return MetricsResult(0, 0.0, {}, {})

    def _analyze_react_components(self, code: str, is_typescript: bool) -> Dict[str, Any]:
        """Analyzes React components and their properties."""
        try:
            input_data = {
                "code": code,
                "options": {
                    "typescript": is_typescript,
                    "plugins": ["jsx", "react"]
                }
            }

            return self._run_script(
                script_name="react_analyzer.js",
                input_data=input_data,
                error_message="Error analyzing React components"
            ) or {}

        except Exception as e:
            logger.error(f"Error analyzing React components: {str(e)}", exc_info=True)
            return {}

    def _get_parser_options(self, is_typescript: bool) -> Dict[str, Any]:
        """Gets appropriate parser options based on file type."""
        options = {
            "sourceType": "module",
            "plugins": [
                "jsx",
                "decorators-legacy",
                ["decorators", { "decoratorsBeforeExport": True }],
                "classProperties",
                "classPrivateProperties",
                "classPrivateMethods",
                "exportDefaultFrom",
                "exportNamespaceFrom",
                "dynamicImport",
                "nullishCoalescing",
                "optionalChaining",
            ]
        }
        
        if is_typescript:
            options["plugins"].extend([
                "typescript",
                "decorators-legacy",
                "classProperties"
            ])

        return options

    @staticmethod
    def _is_typescript_file(file_path: Optional[str]) -> bool:
        """Determines if a file is TypeScript based on extension."""
        if not file_path:
            return False
        return file_path.lower().endswith(('.ts', '.tsx'))

    @staticmethod
    def _is_react_file(file_path: Optional[str]) -> bool:
        """Determines if a file contains React components."""
        if not file_path:
            return False
        return file_path.lower().endswith(('.jsx', '.tsx'))

    def _get_eslint_config(self, is_typescript: bool) -> str:
        """Gets appropriate ESLint configuration file path."""
        config_name = '.eslintrc.typescript.json' if is_typescript else '.eslintrc.json'
        return os.path.join(self.script_dir, config_name)

    def _get_empty_structure(self, reason: str = "") -> Dict[str, Any]:
        """Returns an empty structure with optional reason."""
        return {
            "classes": [],
            "functions": [],
            "variables": [],
            "constants": [],
            "imports": [],
            "exports": [],
            "react_components": [],
            "summary": f"Empty structure: {reason}" if reason else "Empty structure",
            "halstead": {
                "volume": 0,
                "difficulty": 0,
                "effort": 0
            },
            "complexity": 0,
            "maintainability_index": 0
        }

    def _run_parser_script(self, input_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Runs the parser script with error handling."""
        return self._run_script(
            script_name="js_ts_parser.js",
            input_data=input_data,
            error_message="Error running parser"
        )

    def _run_inserter_script(self, input_data: Dict[str, Any]) -> Optional[str]:
        """Runs the documentation inserter script with error handling."""
        return self._run_script(
            script_name="js_ts_inserter.js",
            input_data=input_data,
            error_message="Error running inserter"
        )

    def _run_script(
        self,
        script_name: str,
        input_data: Dict[str, Any],
        error_message: str,
        timeout: int = 30
    ) -> Any:
        """
        Runs a Node.js script with robust error handling.

        Args:
            script_name (str): Name of the script to run
            input_data (Dict[str, Any]): Input data for the script
            error_message (str): Error message prefix
            timeout (int): Timeout in seconds

        Returns:
            Any: Script output or None on failure
        """
        script_path = os.path.join(self.script_dir, script_name)
        
        try:
            # Ensure proper string encoding
            input_json = json.dumps(input_data, ensure_ascii=False)

            process = subprocess.run(
                ["node", script_path],
                input=input_json,
                encoding='utf-8',
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout
            )

            if process.returncode == 0:
                try:
                    return json.loads(process.stdout)
                except json.JSONDecodeError:
                    if script_name == "js_ts_inserter.js":
                        return process.stdout  # Return raw output for inserter
                    logger.error(f"{error_message}: Invalid JSON output")
                    return None
            else:
                logger.error(f"{error_message}: {process.stderr}")
                return None

        except subprocess.TimeoutExpired:
            logger.error(f"{error_message}: Script timeout after {timeout}s")
            return None
        except subprocess.CalledProcessError as e:
            logger.error(f"{error_message}: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"{error_message}: {str(e)}")
            return None
```

the **function_schema.json**

```json
{
  "functions": [
    {
      "name": "generate_documentation",
      "description": "Generates documentation for code structures.",
      "parameters": {
        "type": "object",
        "properties": {
          "docstring_format": {
            "type": "string",
            "description": "Format of the docstring (e.g., Google, NumPy, Sphinx).",
            "enum": ["Google"]
          },
          "summary": {
            "type": "string",
            "description": "A detailed summary of the file."
          },
          "changes_made": {
            "type": "array",
            "items": {
              "type": "string"
            },
            "description": "List of changes made to the file."
          },
          "functions": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "name": {
                  "type": "string",
                  "description": "Function name."
                },
                "docstring": {
                  "type": "string",
                  "description": "Detailed description of the function in Google Style."
                },
                "args": {
                  "type": "array",
                  "items": {
                    "type": "string"
                  },
                  "description": "List of argument names."
                },
                "async": {
                  "type": "boolean",
                  "description": "Whether the function is asynchronous."
                },
                "complexity": {
                  "type": "integer",
                  "description": "Cyclomatic complexity of the function."
                }
              },
              "required": [
                "name",
                "docstring",
                "args",
                "async",
                "complexity"
              ]
            },
            "description": "List of functions."
          },
          "classes": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "name": {
                  "type": "string",
                  "description": "Class name."
                },
                "docstring": {
                  "type": "string",
                  "description": "Detailed description of the class in Google Style."
                },
                "methods": {
                  "type": "array",
                  "items": {
                    "type": "object",
                    "properties": {
                      "name": {
                        "type": "string",
                        "description": "Method name."
                      },
                      "docstring": {
                        "type": "string",
                        "description": "Detailed description of the method in Google Style."
                      },
                      "args": {
                        "type": "array",
                        "items": {
                          "type": "string"
                        },
                        "description": "List of argument names."
                      },
                      "async": {
                        "type": "boolean",
                        "description": "Whether the method is asynchronous."
                      },
                      "type": {
                        "type": "string",
                        "description": "Method type (e.g., 'instance', 'class', 'static')."
                      },
                      "complexity": {
                        "type": "integer",
                        "description": "Cyclomatic complexity of the method."
                      }
                    },
                    "required": [
                      "name",
                      "docstring",
                      "args",
                      "async",
                      "type",
                      "complexity"
                    ]
                  },
                  "description": "List of methods."
                }
              },
              "required": [
                "name",
                "docstring",
                "methods"
              ]
            },
            "description": "List of classes."
          },
          "halstead": {
            "type": "object",
            "properties": {
              "volume": {
                "type": "number",
                "description": "Halstead Volume"
              },
              "difficulty": {
                "type": "number",
                "description": "Halstead Difficulty"
              },
              "effort": {
                "type": "number",
                "description": "Halstead Effort"
              }
            },
            "description": "Halstead complexity metrics."
          },
          "maintainability_index": {
            "type": "number",
            "description": "Maintainability index of the code."
          },
          "variables": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "name": {
                  "type": "string",
                  "description": "Variable name."
                },
                "type": {
                  "type": "string",
                  "description": "Inferred data type of the variable."
                },
                "description": {
                  "type": "string",
                  "description": "Description of the variable."
                },
                "file": {
                  "type": "string",
                  "description": "File where the variable is defined."
                },
                "line": {
                  "type": "integer",
                  "description": "Line number where the variable is defined."
                },
                "link": {
                  "type": "string",
                  "description": "Link to the variable definition."
                },
                "example": {
                  "type": "string",
                  "description": "Example usage of the variable."
                },
                "references": {
                  "type": "string",
                  "description": "References to the variable."
                }
              },
              "required": [
                "name",
                "type",
                "description",
                "file",
                "line",
                "link",
                "example",
                "references"
              ]
            },
            "description": "List of variables in the code."
          },
          "constants": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "name": {
                  "type": "string",
                  "description": "Constant name."
                },
                "type": {
                  "type": "string",
                  "description": "Inferred data type of the constant."
                },
                "description": {
                  "type": "string",
                  "description": "Description of the constant."
                },
                "file": {
                  "type": "string",
                  "description": "File where the constant is defined."
                },
                "line": {
                  "type": "integer",
                  "description": "Line number where the constant is defined."
                },
                "link": {
                  "type": "string",
                  "description": "Link to the constant definition."
                },
                "example": {
                  "type": "string",
                  "description": "Example usage of the constant."
                },
                "references": {
                  "type": "string",
                  "description": "References to the constant."
                }
              },
              "required": [
                "name",
                "type",
                "description",
                "file",
                "line",
                "link",
                "example",
                "references"
              ]
            },
            "description": "List of constants in the code."
          }
        },
        "required": [
          "docstring_format",
          "summary",
          "changes_made",
          "functions",
          "classes",
          "halstead",
          "maintainability_index",
          "variables",
          "constants"
        ]
      }
    }
  ]
}
```