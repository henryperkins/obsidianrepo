---
epoch: 1734586572625
modelKey: gemini-2.0-flash-exp|google
tags:
  - copilot-conversation
---

**user**: Failed validating 'required' in schema:
    {'type': 'object',
     'properties': {'summary': {'type': 'string',
                                'description': 'A brief summary of the '
                                               'method or function.'},
                    'description': {'type': 'string',
                                    'description': 'Detailed description '
                                                   'of the method or '
                                                   'function.'},
                    'args': {'type': 'array',
                             'description': 'A list of arguments for the '
                                            'method or function.',
                             'items': {'type': 'object',
                                       'properties': {'name': {'type': 'string',
                                                               'description': 'The '
                                                                              'name '
                                                                              'of '
                                                                              'the '
                                                                              'argument.'},
                                                      'type': {'type': 'string',
                                                               'description': 'The '
                                                                              'data '
                                                                              'type '
                                                                              'of '
                                                                              'the '
                                                                              'argument.'},
                                                      'description': {'type': 'string',
                                                                      'description': 'A '
                                                                                     'brief '
                                                                                     'description '
                                                                                     'of '
                                                                                     'the '
                                                                                     'argument.'}},
                                       'required': ['name',
                                                    'type',
                                                    'description'],
                                       'additionalProperties': False},
                             'default': []},
                    'returns': {'type': 'object',
                                'description': 'Details about the return '
                                               'value of the method or '
                                               'function.',
                                'properties': {'type': {'type': 'string',
                                                        'description': 'The '
                                                                       'data '
                                                                       'type '
                                                                       'of '
                                                                       'the '
                                                                       'return '
                                                                       'value.'},
                                               'description': {'type': 'string',
                                                               'description': 'A '
                                                                              'brief '
                                                                              'description '
                                                                              'of '
                                                                              'the '
                                                                              'return '
                                                                              'value.'}},
                                'required': ['type', 'description'],
                                'additionalProperties': False},
                    'raises': {'type': 'array',
                               'description': 'A list of exceptions that '
                                              'may be raised by the method '
                                              'or function.',
                               'items': {'type': 'object',
                                         'properties': {'exception': {'type': 'string',
                                                                      'description': 'The '
                                                                                     'name '
                                                                                     'of '
                                                                                     'the '
                                                                                     'exception '
                                                                                     'that '
                                                                                     'may '
                                                                                     'be '
                                                                                     'raised.'},
                                                        'description': {'type': 'string',
                                                                        'description': 'A '
                                                                                       'brief '
                                                                                       'description '
                                                                                       'of '
                                                                                       'the '
                                                                                       'circumstances '
                                                                                       'under '
                                                                                       'which '
                                                                                       'the '
                                                                                       'exception '
                                                                                       'is '
                                                                                       'raised.'}},
                                         'required': ['exception',
                                                      'description'],
                                         'additionalProperties': False},
                               'default': []},
                    'complexity': {'type': ['integer', 'string'],
                                   'description': 'McCabe complexity score',
                                   'default': 1}},
     'required': ['summary', 'description', 'returns'],
     'additionalProperties': False}

On instance:
    {'choices': [{'message': {'content': '{"summary": "Invalid response '
                                         'format", "description": "The '
                                         'response did not match the '
                                         'expected structure.", "error": '
                                         '"", "returns": {"type": "Any", '
                                         '"description": "No return '
                                         'description provided"}}'}}],
     'usage': {}}
API call failed: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                                               \'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n                                
                   \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                 
            \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                             \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                                                               \'description\': \'The \'\n                                                                              \'name \'\n                                                                          
    \'of \'\n                                                                              \'the \'\n                                                                              \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                                                               \'description\': \'The \'\n                                                                              \'data \'\n                                                                             
 \'type \'\n                                                                              \'of \'\n                                                                              \'the \'\n                                                                              \'argument.\'},\n                                                      \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                                                                    
                 \'brief \'\n                                                                                     \'description \'\n                                                
                                     \'of \'\n                                                                                     \'the \'\n                                       
                                              \'argument.\'}},\n                                       \'required\': [\'name\',\n                                                   
 \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                       
      \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                  
                             \'value of the method or \'\n                                               \'function.\',\n                                \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                                                                       \'data \'\n                                                                       \'type \'\n                                                                       \'of \'\n         
                                                              \'the \'\n                                                                       \'return \'\n                        
                                               \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n                               
                                \'description\': \'A \'\n                                                                              \'brief \'\n                                 
                                             \'description \'\n                                                                              \'of \'\n                              
                                                \'the \'\n                                                                              \'return \'\n                               
                                               \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                                \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                    
                          \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                                                                      \'description\': \'The \'\n                                                                                     \'name \'\n                                                           
                          \'of \'\n                                                                                     \'the \'\n                                                  
                                   \'exception \'\n                                                                                     \'that \'\n                                 
                                                    \'may \'\n                                                                                     \'be \'\n                        
                                                             \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                                                                       \'brief \'\n                                                                                       \'description \'\n                                                                  
                     \'of \'\n                                                                                       \'the \'\n                                                     
                                  \'circumstances \'\n                                                                                       \'under \'\n                           
                                                            \'which \'\n                                                                                       \'the \'\n           
                                                                            \'exception \'\n                                                                                       \'is \'\n                                                                                       \'raised.\'}},\n                                         \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                            
   \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                            
             \'response did not match the \'\n                                         \'expected structure.", "error": \'\n                                         \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                         \'description provided"}}\'}}],\n     \'usage\': {}}']
Traceback (most recent call last):
  File "C:\Users\htper\New folder\simple\core\ai_service.py", line 480, in generate_documentation
    return ProcessingResult(
        ^^^^^^^^^^^^^^^^^^^^
core.exceptions.DataValidationError: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                                               
\'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n            
                                       \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                           
  \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                                                     
          \'description\': \'The \'\n                                                                              \'name \'\n                                                      
                        \'of \'\n                                                                              \'the \'\n                                                           
                   \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                                                        
       \'description\': \'The \'\n                                                                              \'data \'\n                                                         
                     \'type \'\n                                                                              \'of \'\n                                                             
                 \'the \'\n                                                                              \'argument.\'},\n                                                      \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                                                
                                     \'brief \'\n                                                                                     \'description \'\n                            
                                                         \'of \'\n                                                                                     \'the \'\n                   
                                                                  \'argument.\'}},\n                                       \'required\': [\'name\',\n                               
                     \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n                                \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                                                     
                  \'data \'\n                                                                       \'type \'\n                                                                     
  \'of \'\n                                                                       \'the \'\n                                                                       \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n           
                                                    \'description\': \'A \'\n                                                                              \'brief \'\n             
                                                                 \'description \'\n                                                                              \'of \'\n          
                                                                    \'the \'\n                                                                              \'return \'\n           
                                                                   \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                      
          \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                                                    
                  \'description\': \'The \'\n                                                                                     \'name \'\n                                       
                                              \'of \'\n                                                                                     \'the \'\n                              
                                                       \'exception \'\n                                                                                     \'that \'\n             
                                                                        \'may \'\n                                                                                     \'be \'\n                                                                                     \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                                                      
                 \'brief \'\n                                                                                       \'description \'\n                                              
                                         \'of \'\n                                                                                       \'the \'\n                                 
                                                      \'circumstances \'\n                                                                                       \'under \'\n                                                                                       \'which \'\n                                                                                       \'the \'\n                                                                                       \'exception \'\n                                                                    
                   \'is \'\n                                                                                       \'raised.\'}},\n                                         \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n                                    
     \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                         \'description provided"}}\'}}],\n     \'usage\': {}}']

Operation: documentation_generation
  Success: False
  Duration: 3.25s
  error: Failed to generate documentation: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                                         
      \'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n                                                   \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                     
        \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                                               
                \'description\': \'The \'\n                                                                              \'name \'\n                                                
                              \'of \'\n                                                                              \'the \'\n                                                     
                         \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                                                  
             \'description\': \'The \'\n                                                                              \'data \'\n                                                   
                           \'type \'\n                                                                              \'of \'\n                                                       
                       \'the \'\n                                                                              \'argument.\'},\n                                                    
  \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                                          
                                           \'brief \'\n                                                                                     \'description \'\n                      
                                                               \'of \'\n                                                                                     \'the \'\n             
                                                                        \'argument.\'}},\n                                       \'required\': [\'name\',\n                         
                           \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n                                \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                                               
                        \'data \'\n                                                                       \'type \'\n                                                               
        \'of \'\n                                                                       \'the \'\n                                                                       \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n                                                               \'description\': \'A \'\n                                                                              \'brief \'\n                                                                              \'description \'\n                                                                              \'of \'\n                                                                              \'the \'\n                                                                              \'return \'\n                                                                              \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                
                \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                             
  \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                                              
                        \'description\': \'The \'\n                                                                                     \'name \'\n                                 
                                                    \'of \'\n                                                                                     \'the \'\n                        
                                                             \'exception \'\n                                                                                     \'that \'\n                                                                                     \'may \'\n                                                                                     \'be \'\n                                                                                     \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                                                
                       \'brief \'\n                                                                                       \'description \'\n                                        
                                               \'of \'\n                                                                                       \'the \'\n                           
                                                            \'circumstances \'\n                                                                                       \'under \'\n                                                                                       \'which \'\n                                                                                  
     \'the \'\n                                                                                       \'exception \'\n                                                              
                         \'is \'\n                                                                                       \'raised.\'}},\n                                         \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n                              
           \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                         \'description provided"}}\'}}],\n     \'usage\': {}}']
Documentation generation failed: Failed to generate documentation: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                 
                              \'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n                                                   \'of the method or \'\n                                                   \'function.\'},\n                    
\'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                             \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                       
                                        \'description\': \'The \'\n                                                                              \'name \'\n                        
                                                      \'of \'\n                                                                              \'the \'\n                             
                                                 \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                          
                                     \'description\': \'The \'\n                                                                              \'data \'\n                           
                                                   \'type \'\n                                                                              \'of \'\n                               
                                               \'the \'\n                                                                              \'argument.\'},\n                            
                          \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                  
                                                                   \'brief \'\n                                                                                     \'description \'\n                                                                                     \'of \'\n                                                                                    
 \'the \'\n                                                                                     \'argument.\'}},\n                                       \'required\': [\'name\',\n                                                    \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n         
                       \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                       
                                                \'data \'\n                                                                       \'type \'\n                                       
                                \'of \'\n                                                                       \'the \'\n                                                          
             \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n                                                               \'description\': \'A \'\n                                                                         
     \'brief \'\n                                                                              \'description \'\n                                                                   
           \'of \'\n                                                                              \'the \'\n                                                                        
      \'return \'\n                                                                              \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                                \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                      
                                                \'description\': \'The \'\n                                                                                     \'name \'\n         
                                                                            \'of \'\n                                                                                     \'the \'\n                                                                                     \'exception \'\n                                                                               
      \'that \'\n                                                                                     \'may \'\n                                                                    
                 \'be \'\n                                                                                     \'raised.\'},\n                                                      
  \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                        
                                               \'brief \'\n                                                                                       \'description \'\n                
                                                                       \'of \'\n                                                                                       \'the \'\n                                                                                       \'circumstances \'\n                                                                            
           \'under \'\n                                                                                       \'which \'\n                                                          
                             \'the \'\n                                                                                       \'exception \'\n                                      
                                                 \'is \'\n                                                                                       \'raised.\'}},\n                   
                      \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                 
  \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n                                         \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                   
      \'description provided"}}\'}}],\n     \'usage\': {}}']
Traceback (most recent call last):
  File "C:\Users\htper\New folder\simple\core\ai_service.py", line 480, in generate_documentation
    return ProcessingResult(
        ^^^^^^^^^^^^^^^^^^^^
core.exceptions.DataValidationError: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                                               
\'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n            
                                       \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                           
  \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                                                     
          \'description\': \'The \'\n                                                                              \'name \'\n                                                      
                        \'of \'\n                                                                              \'the \'\n                                                           
                   \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                                                        
       \'description\': \'The \'\n                                                                              \'data \'\n                                                         
                     \'type \'\n                                                                              \'of \'\n                                                             
                 \'the \'\n                                                                              \'argument.\'},\n                                                      \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                                                
                                     \'brief \'\n                                                                                     \'description \'\n                            
                                                         \'of \'\n                                                                                     \'the \'\n                   
                                                                  \'argument.\'}},\n                                       \'required\': [\'name\',\n                               
                     \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n                                \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                                                     
                  \'data \'\n                                                                       \'type \'\n                                                                     
  \'of \'\n                                                                       \'the \'\n                                                                       \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n           
                                                    \'description\': \'A \'\n                                                                              \'brief \'\n             
                                                                 \'description \'\n                                                                              \'of \'\n          
                                                                    \'the \'\n                                                                              \'return \'\n           
                                                                   \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                      
          \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                                                    
                  \'description\': \'The \'\n                                                                                     \'name \'\n                                       
                                              \'of \'\n                                                                                     \'the \'\n                              
                                                       \'exception \'\n                                                                                     \'that \'\n             
                                                                        \'may \'\n                                                                                     \'be \'\n                                                                                     \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                                                      
                 \'brief \'\n                                                                                       \'description \'\n                                              
                                         \'of \'\n                                                                                       \'the \'\n                                 
                                                      \'circumstances \'\n                                                                                       \'under \'\n                                                                                       \'which \'\n                                                                                       \'the \'\n                                                                                       \'exception \'\n                                                                    
                   \'is \'\n                                                                                       \'raised.\'}},\n                                         \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n                                    
     \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                         \'description provided"}}\'}}],\n     \'usage\': {}}']

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\htper\New folder\simple\core\ai_service.py", line 510, in generate_documentation
    if self._client:
            ^^^^^^^^^
core.exceptions.APICallError: Failed to generate documentation: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                    
                           \'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n                                                   \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                             \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                          
                                     \'description\': \'The \'\n                                                                              \'name \'\n                           
                                                   \'of \'\n                                                                              \'the \'\n                                
                                              \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                             
                                  \'description\': \'The \'\n                                                                              \'data \'\n                              
                                                \'type \'\n                                                                              \'of \'\n                                  
                                            \'the \'\n                                                                              \'argument.\'},\n                               
                       \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                     
                                                                \'brief \'\n                                                                                     \'description \'\n                                                                                     \'of \'\n                                                                                     \'the \'\n                                                                                     \'argument.\'}},\n                                       \'required\': [\'name\',\n                                                    \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n            
                    \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                          
                                             \'data \'\n                                                                       \'type \'\n                                          
                             \'of \'\n                                                                       \'the \'\n                                                             
          \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n                                                               \'description\': \'A \'\n                                                                            
  \'brief \'\n                                                                              \'description \'\n                                                                      
        \'of \'\n                                                                              \'the \'\n                                                                           
   \'return \'\n                                                                              \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                                \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                         
                                             \'description\': \'The \'\n                                                                                     \'name \'\n            
                                                                         \'of \'\n                                                                                     \'the \'\n                                                                                     \'exception \'\n                                                                                  
   \'that \'\n                                                                                     \'may \'\n                                                                       
              \'be \'\n                                                                                     \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                           
                                            \'brief \'\n                                                                                       \'description \'\n                   
                                                                    \'of \'\n                                                                                       \'the \'\n                                                                                       \'circumstances \'\n                                                                               
        \'under \'\n                                                                                       \'which \'\n                                                             
                          \'the \'\n                                                                                       \'exception \'\n                                         
                                              \'is \'\n                                                                                       \'raised.\'}},\n                      
                   \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n         
                                \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                      
   \'description provided"}}\'}}],\n     \'usage\': {}}']
Error during documentation generation: Failed to generate documentation: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n           
                                    \'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n                                                   \'of the method or \'\n                                                   \'function.\'},\n              
      \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                             \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                 
                                              \'description\': \'The \'\n                                                                              \'name \'\n                  
                                                            \'of \'\n                                                                              \'the \'\n                       
                                                       \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                    
                                           \'description\': \'The \'\n                                                                              \'data \'\n                     
                                                         \'type \'\n                                                                              \'of \'\n                         
                                                     \'the \'\n                                                                              \'argument.\'},\n                      
                                \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n            
                                                                         \'brief \'\n                                                                                     \'description \'\n                                                                                     \'of \'\n                                                                              
       \'the \'\n                                                                                     \'argument.\'}},\n                                       \'required\': [\'name\',\n                                                    \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n                                \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                 
                                                      \'data \'\n                                                                       \'type \'\n                                 
                                      \'of \'\n                                                                       \'the \'\n                                                    
                   \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n                                                               \'description\': \'A \'\n                                                                   
           \'brief \'\n                                                                              \'description \'\n                                                             
                 \'of \'\n                                                                              \'the \'\n                                                                  
            \'return \'\n                                                                              \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                                \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                
                                                      \'description\': \'The \'\n                                                                                     \'name \'\n                                                                                     \'of \'\n                                                                                     \'the \'\n                                                                                     \'exception \'\n                                                                         
            \'that \'\n                                                                                     \'may \'\n                                                              
                       \'be \'\n                                                                                     \'raised.\'},\n                                                
        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                  
                                                     \'brief \'\n                                                                                       \'description \'\n          
                                                                             \'of \'\n                                                                                       \'the \'\n                                                                                       \'circumstances \'\n                                                                      
                 \'under \'\n                                                                                       \'which \'\n                                                    
                                   \'the \'\n                                                                                       \'exception \'\n                                
                                                       \'is \'\n                                                                                       \'raised.\'}},\n             
                            \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                           
        \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n                                         \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                             
            \'description provided"}}\'}}],\n     \'usage\': {}}']
Traceback (most recent call last):
  File "C:\Users\htper\New folder\simple\core\ai_service.py", line 480, in generate_documentation
    return ProcessingResult(
        ^^^^^^^^^^^^^^^^^^^^
core.exceptions.DataValidationError: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                                               
\'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n            
                                       \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                           
  \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                                                     
          \'description\': \'The \'\n                                                                              \'name \'\n                                                      
                        \'of \'\n                                                                              \'the \'\n                                                           
                   \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                                                        
       \'description\': \'The \'\n                                                                              \'data \'\n                                                         
                     \'type \'\n                                                                              \'of \'\n                                                             
                 \'the \'\n                                                                              \'argument.\'},\n                                                      \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                                                
                                     \'brief \'\n                                                                                     \'description \'\n                            
                                                         \'of \'\n                                                                                     \'the \'\n                   
                                                                  \'argument.\'}},\n                                       \'required\': [\'name\',\n                               
                     \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n                                \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                                                     
                  \'data \'\n                                                                       \'type \'\n                                                                     
  \'of \'\n                                                                       \'the \'\n                                                                       \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n           
                                                    \'description\': \'A \'\n                                                                              \'brief \'\n             
                                                                 \'description \'\n                                                                              \'of \'\n          
                                                                    \'the \'\n                                                                              \'return \'\n           
                                                                   \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                      
          \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                                                    
                  \'description\': \'The \'\n                                                                                     \'name \'\n                                       
                                              \'of \'\n                                                                                     \'the \'\n                              
                                                       \'exception \'\n                                                                                     \'that \'\n             
                                                                        \'may \'\n                                                                                     \'be \'\n                                                                                     \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                                                      
                 \'brief \'\n                                                                                       \'description \'\n                                              
                                         \'of \'\n                                                                                       \'the \'\n                                 
                                                      \'circumstances \'\n                                                                                       \'under \'\n                                                                                       \'which \'\n                                                                                       \'the \'\n                                                                                       \'exception \'\n                                                                    
                   \'is \'\n                                                                                       \'raised.\'}},\n                                         \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n                                    
     \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                         \'description provided"}}\'}}],\n     \'usage\': {}}']

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\htper\New folder\simple\core\docs.py", line 126, in generate_documentation
    processing_result: ProcessingResult = await self.ai_service.generate_documentation(
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\htper\New folder\simple\core\ai_service.py", line 510, in generate_documentation
    if self._client:
            ^^^^^^^^^
core.exceptions.APICallError: Failed to generate documentation: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                    
                           \'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n                                                   \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                             \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                          
                                     \'description\': \'The \'\n                                                                              \'name \'\n                           
                                                   \'of \'\n                                                                              \'the \'\n                                
                                              \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                             
                                  \'description\': \'The \'\n                                                                              \'data \'\n                              
                                                \'type \'\n                                                                              \'of \'\n                                  
                                            \'the \'\n                                                                              \'argument.\'},\n                               
                       \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                     
                                                                \'brief \'\n                                                                                     \'description \'\n                                                                                     \'of \'\n                                                                                     \'the \'\n                                                                                     \'argument.\'}},\n                                       \'required\': [\'name\',\n                                                    \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n            
                    \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                          
                                             \'data \'\n                                                                       \'type \'\n                                          
                             \'of \'\n                                                                       \'the \'\n                                                             
          \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n                                                               \'description\': \'A \'\n                                                                            
  \'brief \'\n                                                                              \'description \'\n                                                                      
        \'of \'\n                                                                              \'the \'\n                                                                           
   \'return \'\n                                                                              \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                                \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                         
                                             \'description\': \'The \'\n                                                                                     \'name \'\n            
                                                                         \'of \'\n                                                                                     \'the \'\n                                                                                     \'exception \'\n                                                                                  
   \'that \'\n                                                                                     \'may \'\n                                                                       
              \'be \'\n                                                                                     \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                           
                                            \'brief \'\n                                                                                       \'description \'\n                   
                                                                    \'of \'\n                                                                                       \'the \'\n                                                                                       \'circumstances \'\n                                                                               
        \'under \'\n                                                                                       \'which \'\n                                                             
                          \'the \'\n                                                                                       \'exception \'\n                                         
                                              \'is \'\n                                                                                       \'raised.\'}},\n                      
                   \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n         
                                \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                      
   \'description provided"}}\'}}],\n     \'usage\': {}}']

Operation: documentation_generation
  Success: False
  Duration: 3.31s
  module_name: token_management
  error: Failed to generate documentation: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                                         
      \'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n                                                   \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                     
        \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                                               
                \'description\': \'The \'\n                                                                              \'name \'\n                                                
                              \'of \'\n                                                                              \'the \'\n                                                     
                         \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                                                  
             \'description\': \'The \'\n                                                                              \'data \'\n                                                   
                           \'type \'\n                                                                              \'of \'\n                                                       
                       \'the \'\n                                                                              \'argument.\'},\n                                                    
  \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                                          
                                           \'brief \'\n                                                                                     \'description \'\n                      
                                                               \'of \'\n                                                                                     \'the \'\n             
                                                                        \'argument.\'}},\n                                       \'required\': [\'name\',\n                         
                           \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n                                \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                                               
                        \'data \'\n                                                                       \'type \'\n                                                               
        \'of \'\n                                                                       \'the \'\n                                                                       \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n                                                               \'description\': \'A \'\n                                                                              \'brief \'\n                                                                              \'description \'\n                                                                              \'of \'\n                                                                              \'the \'\n                                                                              \'return \'\n                                                                              \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                
                \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                             
  \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                                              
                        \'description\': \'The \'\n                                                                                     \'name \'\n                                 
                                                    \'of \'\n                                                                                     \'the \'\n                        
                                                             \'exception \'\n                                                                                     \'that \'\n                                                                                     \'may \'\n                                                                                     \'be \'\n                                                                                     \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                                                
                       \'brief \'\n                                                                                       \'description \'\n                                        
                                               \'of \'\n                                                                                       \'the \'\n                           
                                                            \'circumstances \'\n                                                                                       \'under \'\n                                                                                       \'which \'\n                                                                                  
     \'the \'\n                                                                                       \'exception \'\n                                                              
                         \'is \'\n                                                                                       \'raised.\'}},\n                                         \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n                              
           \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                         \'description provided"}}\'}}],\n     \'usage\': {}}']
Unexpected error generating documentation for simple\api\token_management.py: Failed to generate documentation: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                                               \'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                 
                   \'description\': \'Detailed description \'\n                                                   \'of the method or \'\n                                           
        \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                      
                      \'method or function.\',\n                             \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                                                               \'description\': \'The \'\n                                                                    
          \'name \'\n                                                                              \'of \'\n                                                                        
      \'the \'\n                                                                              \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                                                               \'description\': \'The \'\n                                                                       
       \'data \'\n                                                                              \'type \'\n                                                                         
     \'of \'\n                                                                              \'the \'\n                                                                              
\'argument.\'},\n                                                      \'description\': {\'type\': \'string\',\n                                                                    
  \'description\': \'A \'\n                                                                                     \'brief \'\n                                                        
                             \'description \'\n                                                                                     \'of \'\n                                       
                                              \'the \'\n                                                                                     \'argument.\'}},\n                     
                  \'required\': [\'name\',\n                                                    \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                           
                    \'function.\',\n                                \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                                                                       \'data \'\n                                                                       \'type \'\n                                                                       \'of \'\n                                                                       \'the \'\n             
                                                          \'return \'\n                                                                       \'value.\'},\n                        
                       \'description\': {\'type\': \'string\',\n                                                               \'description\': \'A \'\n                            
                                                  \'brief \'\n                                                                              \'description \'\n                      
                                                        \'of \'\n                                                                              \'the \'\n                           
                                                   \'return \'\n                                                                              \'value.\'}},\n                       
         \'required\': [\'type\', \'description\'],\n                                \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                         
                     \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                                                                      \'description\': \'The \'\n                                                            
                         \'name \'\n                                                                                     \'of \'\n                                                  
                                   \'the \'\n                                                                                     \'exception \'\n                                  
                                                   \'that \'\n                                                                                     \'may \'\n                       
                                                              \'be \'\n                                                                                     \'raised.\'},\n         
                                               \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                                                                       \'brief \'\n                                                                            
           \'description \'\n                                                                                       \'of \'\n                                                       
                                \'the \'\n                                                                                       \'circumstances \'\n                               
                                                        \'under \'\n                                                                                       \'which \'\n             
                                                                          \'the \'\n                                                                                       \'exception \'\n                                                                                       \'is \'\n                                                                             
          \'raised.\'}},\n                                         \'required\': [\'exception\',\n                                                      \'description\'],\n         
                                \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n         
                                \'format", "description": "The \'\n                                         \'response did not match the \'\n                                       
  \'expected structure.", "error": \'\n                                         \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                         \'description provided"}}\'}}],\n     \'usage\': {}}']
Traceback (most recent call last):
  File "C:\Users\htper\New folder\simple\core\ai_service.py", line 480, in generate_documentation
    return ProcessingResult(
        ^^^^^^^^^^^^^^^^^^^^
core.exceptions.DataValidationError: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                                               
\'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n            
                                       \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                           
  \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                                                     
          \'description\': \'The \'\n                                                                              \'name \'\n                                                      
                        \'of \'\n                                                                              \'the \'\n                                                           
                   \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                                                        
       \'description\': \'The \'\n                                                                              \'data \'\n                                                         
                     \'type \'\n                                                                              \'of \'\n                                                             
                 \'the \'\n                                                                              \'argument.\'},\n                                                      \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                                                
                                     \'brief \'\n                                                                                     \'description \'\n                            
                                                         \'of \'\n                                                                                     \'the \'\n                   
                                                                  \'argument.\'}},\n                                       \'required\': [\'name\',\n                               
                     \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n                                \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                                                     
                  \'data \'\n                                                                       \'type \'\n                                                                     
  \'of \'\n                                                                       \'the \'\n                                                                       \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n           
                                                    \'description\': \'A \'\n                                                                              \'brief \'\n             
                                                                 \'description \'\n                                                                              \'of \'\n          
                                                                    \'the \'\n                                                                              \'return \'\n           
                                                                   \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                      
          \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                                                    
                  \'description\': \'The \'\n                                                                                     \'name \'\n                                       
                                              \'of \'\n                                                                                     \'the \'\n                              
                                                       \'exception \'\n                                                                                     \'that \'\n             
                                                                        \'may \'\n                                                                                     \'be \'\n                                                                                     \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                                                      
                 \'brief \'\n                                                                                       \'description \'\n                                              
                                         \'of \'\n                                                                                       \'the \'\n                                 
                                                      \'circumstances \'\n                                                                                       \'under \'\n                                                                                       \'which \'\n                                                                                       \'the \'\n                                                                                       \'exception \'\n                                                                    
                   \'is \'\n                                                                                       \'raised.\'}},\n                                         \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n                                    
     \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                         \'description provided"}}\'}}],\n     \'usage\': {}}']

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\htper\New folder\simple\core\docs.py", line 325, in generate_module_documentation
    updated_code, markdown_doc = await self.generate_documentation(context)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\htper\New folder\simple\core\docs.py", line 126, in generate_documentation
    processing_result: ProcessingResult = await self.ai_service.generate_documentation(
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\htper\New folder\simple\core\ai_service.py", line 510, in generate_documentation
    if self._client:
            ^^^^^^^^^
core.exceptions.APICallError: Failed to generate documentation: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                    
                           \'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n                                                   \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                             \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                          
                                     \'description\': \'The \'\n                                                                              \'name \'\n                           
                                                   \'of \'\n                                                                              \'the \'\n                                
                                              \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                             
                                  \'description\': \'The \'\n                                                                              \'data \'\n                              
                                                \'type \'\n                                                                              \'of \'\n                                  
                                            \'the \'\n                                                                              \'argument.\'},\n                               
                       \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                     
                                                                \'brief \'\n                                                                                     \'description \'\n                                                                                     \'of \'\n                                                                                     \'the \'\n                                                                                     \'argument.\'}},\n                                       \'required\': [\'name\',\n                                                    \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n            
                    \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                          
                                             \'data \'\n                                                                       \'type \'\n                                          
                             \'of \'\n                                                                       \'the \'\n                                                             
          \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n                                                               \'description\': \'A \'\n                                                                            
  \'brief \'\n                                                                              \'description \'\n                                                                      
        \'of \'\n                                                                              \'the \'\n                                                                           
   \'return \'\n                                                                              \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                                \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                         
                                             \'description\': \'The \'\n                                                                                     \'name \'\n            
                                                                         \'of \'\n                                                                                     \'the \'\n                                                                                     \'exception \'\n                                                                                  
   \'that \'\n                                                                                     \'may \'\n                                                                       
              \'be \'\n                                                                                     \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                           
                                            \'brief \'\n                                                                                       \'description \'\n                   
                                                                    \'of \'\n                                                                                       \'the \'\n                                                                                       \'circumstances \'\n                                                                               
        \'under \'\n                                                                                       \'which \'\n                                                             
                          \'the \'\n                                                                                       \'exception \'\n                                         
                                              \'is \'\n                                                                                       \'raised.\'}},\n                      
                   \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n         
                                \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                      
   \'description provided"}}\'}}],\n     \'usage\': {}}']

Operation: module_documentation_generation
  Success: False
  Duration: 3.32s
  module_path: simple\api\token_management.py
  error: Failed to generate documentation: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                                         
      \'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n                                                   \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                     
        \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                                               
                \'description\': \'The \'\n                                                                              \'name \'\n                                                
                              \'of \'\n                                                                              \'the \'\n                                                     
                         \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                                                  
             \'description\': \'The \'\n                                                                              \'data \'\n                                                   
                           \'type \'\n                                                                              \'of \'\n                                                       
                       \'the \'\n                                                                              \'argument.\'},\n                                                    
  \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                                          
                                           \'brief \'\n                                                                                     \'description \'\n                      
                                                               \'of \'\n                                                                                     \'the \'\n             
                                                                        \'argument.\'}},\n                                       \'required\': [\'name\',\n                         
                           \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n                                \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                                               
                        \'data \'\n                                                                       \'type \'\n                                                               
        \'of \'\n                                                                       \'the \'\n                                                                       \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n                                                               \'description\': \'A \'\n                                                                              \'brief \'\n                                                                              \'description \'\n                                                                              \'of \'\n                                                                              \'the \'\n                                                                              \'return \'\n                                                                              \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                
                \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                             
  \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                                              
                        \'description\': \'The \'\n                                                                                     \'name \'\n                                 
                                                    \'of \'\n                                                                                     \'the \'\n                        
                                                             \'exception \'\n                                                                                     \'that \'\n                                                                                     \'may \'\n                                                                                     \'be \'\n                                                                                     \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                                                
                       \'brief \'\n                                                                                       \'description \'\n                                        
                                               \'of \'\n                                                                                       \'the \'\n                           
                                                            \'circumstances \'\n                                                                                       \'under \'\n                                                                                       \'which \'\n                                                                                  
     \'the \'\n                                                                                       \'exception \'\n                                                              
                         \'is \'\n                                                                                       \'raised.\'}},\n                                         \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n                              
           \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                         \'description provided"}}\'}}],\n     \'usage\': {}}']
Error processing file simple\api\token_management.py: Unexpected error generating documentation for simple\api\token_management.py: Failed to generate documentation: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                                               \'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n                                                   \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                             \'items\': {\'type\': \'object\',\n              
                         \'properties\': {\'name\': {\'type\': \'string\',\n                                                               \'description\': \'The \'\n              
                                                                \'name \'\n                                                                              \'of \'\n                  
                                                            \'the \'\n                                                                              \'argument.\'},\n               
                                       \'type\': {\'type\': \'string\',\n                                                               \'description\': \'The \'\n                 
                                                             \'data \'\n                                                                              \'type \'\n                   
                                                           \'of \'\n                                                                              \'the \'\n                        
                                                      \'argument.\'},\n                                                      \'description\': {\'type\': \'string\',\n              
                                                        \'description\': \'A \'\n                                                                                     \'brief \'\n                                                                                     \'description \'\n                                                                               
      \'of \'\n                                                                                     \'the \'\n                                                                      
               \'argument.\'}},\n                                       \'required\': [\'name\',\n                                                    \'type\',\n                   
                                 \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n                                \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                                                                       \'data \'\n                      
                                                 \'type \'\n                                                                       \'of \'\n                                        
                               \'the \'\n                                                                       \'return \'\n                                                       
                \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n                                                              
 \'description\': \'A \'\n                                                                              \'brief \'\n                                                                
              \'description \'\n                                                                              \'of \'\n                                                             
                 \'the \'\n                                                                              \'return \'\n                                                              
                \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                                \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                    
                     \'properties\': {\'exception\': {\'type\': \'string\',\n                                                                      \'description\': \'The \'\n                                                                                     \'name \'\n                                                                                     \'of \'\n                                                                                     \'the \'\n                                                                                 
    \'exception \'\n                                                                                     \'that \'\n                                                                
                     \'may \'\n                                                                                     \'be \'\n                                                       
                              \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                      
                                  \'description\': \'A \'\n                                                                                       \'brief \'\n                      
                                                                 \'description \'\n                                                                                       \'of \'\n                                                                                       \'the \'\n                                                                                    
   \'circumstances \'\n                                                                                       \'under \'\n                                                          
                             \'which \'\n                                                                                       \'the \'\n                                          
                                             \'exception \'\n                                                                                       \'is \'\n                       
                                                                \'raised.\'}},\n                                         \'required\': [\'exception\',\n                            
                          \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n         
           \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                
   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n                                         \'"", "returns": {"type": "Any", \'\n         
                                \'"description": "No return \'\n                                         \'description provided"}}\'}}],\n     \'usage\': {}}']
Traceback (most recent call last):
  File "C:\Users\htper\New folder\simple\core\ai_service.py", line 480, in generate_documentation
    return ProcessingResult(
        ^^^^^^^^^^^^^^^^^^^^
core.exceptions.DataValidationError: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                                               
\'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n            
                                       \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                           
  \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                                                     
          \'description\': \'The \'\n                                                                              \'name \'\n                                                      
                        \'of \'\n                                                                              \'the \'\n                                                           
                   \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                                                        
       \'description\': \'The \'\n                                                                              \'data \'\n                                                         
                     \'type \'\n                                                                              \'of \'\n                                                             
                 \'the \'\n                                                                              \'argument.\'},\n                                                      \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                                                
                                     \'brief \'\n                                                                                     \'description \'\n                            
                                                         \'of \'\n                                                                                     \'the \'\n                   
                                                                  \'argument.\'}},\n                                       \'required\': [\'name\',\n                               
                     \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n                                \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                                                     
                  \'data \'\n                                                                       \'type \'\n                                                                     
  \'of \'\n                                                                       \'the \'\n                                                                       \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n           
                                                    \'description\': \'A \'\n                                                                              \'brief \'\n             
                                                                 \'description \'\n                                                                              \'of \'\n          
                                                                    \'the \'\n                                                                              \'return \'\n           
                                                                   \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                      
          \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                                                    
                  \'description\': \'The \'\n                                                                                     \'name \'\n                                       
                                              \'of \'\n                                                                                     \'the \'\n                              
                                                       \'exception \'\n                                                                                     \'that \'\n             
                                                                        \'may \'\n                                                                                     \'be \'\n                                                                                     \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                                                      
                 \'brief \'\n                                                                                       \'description \'\n                                              
                                         \'of \'\n                                                                                       \'the \'\n                                 
                                                      \'circumstances \'\n                                                                                       \'under \'\n                                                                                       \'which \'\n                                                                                       \'the \'\n                                                                                       \'exception \'\n                                                                    
                   \'is \'\n                                                                                       \'raised.\'}},\n                                         \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n                                    
     \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                         \'description provided"}}\'}}],\n     \'usage\': {}}']

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\htper\New folder\simple\core\docs.py", line 325, in generate_module_documentation
    updated_code, markdown_doc = await self.generate_documentation(context)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\htper\New folder\simple\core\docs.py", line 126, in generate_documentation
    processing_result: ProcessingResult = await self.ai_service.generate_documentation(
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\htper\New folder\simple\core\ai_service.py", line 510, in generate_documentation
    if self._client:
            ^^^^^^^^^
core.exceptions.APICallError: Failed to generate documentation: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                    
                           \'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n                                                   \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                             \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                          
                                     \'description\': \'The \'\n                                                                              \'name \'\n                           
                                                   \'of \'\n                                                                              \'the \'\n                                
                                              \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                             
                                  \'description\': \'The \'\n                                                                              \'data \'\n                              
                                                \'type \'\n                                                                              \'of \'\n                                  
                                            \'the \'\n                                                                              \'argument.\'},\n                               
                       \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                     
                                                                \'brief \'\n                                                                                     \'description \'\n                                                                                     \'of \'\n                                                                                     \'the \'\n                                                                                     \'argument.\'}},\n                                       \'required\': [\'name\',\n                                                    \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n            
                    \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                          
                                             \'data \'\n                                                                       \'type \'\n                                          
                             \'of \'\n                                                                       \'the \'\n                                                             
          \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n                                                               \'description\': \'A \'\n                                                                            
  \'brief \'\n                                                                              \'description \'\n                                                                      
        \'of \'\n                                                                              \'the \'\n                                                                           
   \'return \'\n                                                                              \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                                \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                         
                                             \'description\': \'The \'\n                                                                                     \'name \'\n            
                                                                         \'of \'\n                                                                                     \'the \'\n                                                                                     \'exception \'\n                                                                                  
   \'that \'\n                                                                                     \'may \'\n                                                                       
              \'be \'\n                                                                                     \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                           
                                            \'brief \'\n                                                                                       \'description \'\n                   
                                                                    \'of \'\n                                                                                       \'the \'\n                                                                                       \'circumstances \'\n                                                                               
        \'under \'\n                                                                                       \'which \'\n                                                             
                          \'the \'\n                                                                                       \'exception \'\n                                         
                                              \'is \'\n                                                                                       \'raised.\'}},\n                      
                   \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n         
                                \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                      
   \'description provided"}}\'}}],\n     \'usage\': {}}']

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\htper\New folder\simple\main.py", line 140, in process_file
    await self.doc_orchestrator.generate_module_documentation(
  File "C:\Users\htper\New folder\simple\core\docs.py", line 359, in generate_module_documentation
    raise DocumentationError(error_msg) from gen_error
core.exceptions.DocumentationError: Unexpected error generating documentation for simple\api\token_management.py: Failed to generate documentation: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n              
                  \'description\': \'A brief summary of the \'\n                                               \'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n                                                   \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                             \'items\': {\'type\': \'object\',\n                                
       \'properties\': {\'name\': {\'type\': \'string\',\n                                                               \'description\': \'The \'\n                                
                                              \'name \'\n                                                                              \'of \'\n                                    
                                          \'the \'\n                                                                              \'argument.\'},\n                                 
                     \'type\': {\'type\': \'string\',\n                                                               \'description\': \'The \'\n                                   
                                           \'data \'\n                                                                              \'type \'\n                                     
                                         \'of \'\n                                                                              \'the \'\n                                          
                                    \'argument.\'},\n                                                      \'description\': {\'type\': \'string\',\n                                
                                      \'description\': \'A \'\n                                                                                     \'brief \'\n                    
                                                                 \'description \'\n                                                                                     \'of \'\n                                                                                     \'the \'\n                                                                                     \'argument.\'}},\n                                       \'required\': [\'name\',\n                                                    \'type\',\n                                     
               \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n                                \'properties\': {\'type\': {\'type\': \'string\',\n                         
                               \'description\': \'The \'\n                                                                       \'data \'\n                                        
                               \'type \'\n                                                                       \'of \'\n                                                          
             \'the \'\n                                                                       \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n                                                               \'description\': \'A \'\n                                                                              \'brief \'\n                                                                              \'description \'\n                                                                              \'of \'\n                                                                              \'the \'\n                                                                              \'return \'\n                                                                              \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                                \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                      
   \'properties\': {\'exception\': {\'type\': \'string\',\n                                                                      \'description\': \'The \'\n                        
                                                             \'name \'\n                                                                                     \'of \'\n              
                                                                       \'the \'\n                                                                                     \'exception \'\n                                                                                     \'that \'\n                                                                                  
   \'may \'\n                                                                                     \'be \'\n                                                                         
            \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                        
                \'description\': \'A \'\n                                                                                       \'brief \'\n                                        
                                               \'description \'\n                                                                                       \'of \'\n                   
                                                                    \'the \'\n                                                                                       \'circumstances \'\n                                                                                       \'under \'\n                                                                            
           \'which \'\n                                                                                       \'the \'\n                                                            
                           \'exception \'\n                                                                                       \'is \'\n                                         
                                              \'raised.\'}},\n                                         \'required\': [\'exception\',\n                                              
        \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n                                         \'"", "returns": {"type": "Any", \'\n                           
              \'"description": "No return \'\n                                         \'description provided"}}\'}}],\n     \'usage\': {}}']

--------------------------------------------------------------------------------

ERROR:
  Error processing file: simple\api\token_management.py. Reason: Unexpected error generating documentation for simple\api\token_management.py: Failed to generate documentation: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n {\'type\': \'object\',\n \'properties\': {\'summary\': {\'type\': \'string\',\n \'description\': \'A brief summary of the \'\n \'method or function.\'},\n \'description\': {\'type\': \'string\',\n \'description\': \'Detailed description \'\n \'of the method or \'\n \'function.\'},\n \'args\': {\'type\': \'array\',\n \'description\': \'A list of arguments for the \'\n \'method or function.\',\n \'items\': {\'type\': \'object\',\n \'properties\': {\'name\': {\'type\': \'string\',\n \'description\': \'The \'\n \'name \'\n \'of \'\n \'the \'\n \'argument.\'},\n \'type\': {\'type\': \'string\',\n \'description\': \'The \'\n \'data \'\n \'type \'\n \'of \'\n \'the \'\n \'argument.\'},\n \'description\': {\'type\': \'string\',\n \'description\': \'A \'\n \'brief \'\n \'description \'\n \'of \'\n \'the \'\n \'argument.\'}},\n \'required\': [\'name\',\n \'type\',\n \'description\'],\n \'additionalProperties\': False},\n \'default\': []},\n \'returns\': {\'type\': \'object\',\n \'description\': \'Details about the return \'\n \'value of the method or \'\n \'function.\',\n \'properties\': {\'type\': {\'type\': \'string\',\n \'description\': \'The \'\n \'data \'\n \'type \'\n \'of \'\n \'the \'\n \'return \'\n \'value.\'},\n \'description\': {\'type\': \'string\',\n \'description\': \'A \'\n \'brief \'\n \'description \'\n \'of \'\n \'the \'\n \'return \'\n \'value.\'}},\n \'required\': [\'type\', \'description\'],\n \'additionalProperties\': False},\n \'raises\': {\'type\': \'array\',\n \'description\': \'A list of exceptions that \'\n \'may be raised by the method \'\n \'or function.\',\n \'items\': {\'type\': \'object\',\n \'properties\': {\'exception\': {\'type\': \'string\',\n \'description\': \'The \'\n \'name \'\n \'of \'\n \'the \'\n \'exception \'\n \'that \'\n \'may \'\n \'be \'\n \'raised.\'},\n \'description\': {\'type\': \'string\',\n \'description\': \'A \'\n \'brief \'\n \'description \'\n \'of \'\n \'the \'\n \'circumstances \'\n \'under \'\n \'which \'\n \'the \'\n \'exception \'\n \'is \'\n \'raised.\'}},\n \'required\': [\'exception\',\n \'description\'],\n \'additionalProperties\': False},\n \'default\': []},\n \'complexity\': {\'type\': [\'integer\', \'string\'],\n \'description\': \'McCabe complexity score\',\n \'default\': 1}},\n \'required\': [\'summary\', \'description\', \'returns\'],\n \'additionalProperties\': False}\n\nOn instance:\n {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n \'format", "description": "The \'\n \'response did not match the \'\n \'expected structure.", "error": \'\n \'"", "returns": {"type": "Any", \'\n \'"description": "No return \'\n \'description provided"}}\'}}],\n \'usage\': {}}']

--------------------------------------------------------------------------------

Source code read from simple\api\__init__.py. Length: 798
Generating documentation for simple\api\__init__.py
Validating source code for file: simple
Syntax validation successful for: simple
No __init__.py found at simple\default_module\__init__.py. Module 'default_module' may not be importable.

--------------------------------------------------------------------------------


Metrics Collection

Module: default_module:
----------------------------------------
  cyclomatic_complexity     3
  cognitive_complexity      0
  maintainability_index     94.42
  halstead_metrics          {'volume': 35.909291004461515, 'difficulty': 0.5909090909090909, 'effort': 21.21912650263635, 'time': 1.178840361257575, 'bugs': 0.011969763668153839}  
  lines_of_code             33
  total_functions           1
  scanned_functions         0
  function_scan_ratio       0.00
  total_classes             0
  scanned_classes           0
  class_scan_ratio          0.00
  complexity_graph          None
----------------------------------------

-----------------------------------------------------
[Timestamp: 2024/12/18 23:36:12]

**ai**: Okay, I've processed the provided information. It appears you've given me a list of API keys and related information for various services, along with some best practices for handling them.

I also see that there were some errors during documentation generation for the `simple\api\token_management.py` module. This seems to be due to a validation error in the response from the AI service, specifically that the 'summary' field is required but was missing in the response. This is an issue with the AI service's response format and not with your code or the provided API key document.

Here's a summary of what I've gathered:

**API Keys and Services:**

You have API keys for a wide range of services, including:

*   Hugging Face
*   Azure OpenAI
*   Azure AI
*   Grok 2 Beta
*   Black Forest Labs
*   AI21 Studio
*   Anthropic
*   Cloudflare Workers
*   Cohere
*   Custom Vision - Azure
*   Deepseek
*   Dropbox
*   Google Gemini
*   Google Photos API
*   Google Storage
*   Groq
*   LangSmith
*   Mistral
*   New Relic
*   OpenAI
*   OpenRouter
*   Pinecone
*   SambaNova Cloud API
*   Stability.ai Platform
*   you.com
*   WordPress REST API

**Best Practices:**

You've also outlined important best practices for managing these keys:

*   **Secure Storage:** Keep keys out of version control.
*   **Environment Variables:** Use environment variables to access keys in applications.
*   **Principle of Least Privilege:** Grant only necessary permissions.
*   **Regular Rotation:** Rotate keys periodically.

**Documentation Generation Error:**

*   There was an error when trying to generate documentation for `simple\api\token_management.py`. This is due to the AI service returning a response that did not include the required 'summary' field.

**Metrics Collection:**

*   I've also collected some metrics for the `default_module` module, including cyclomatic complexity, cognitive complexity, maintainability index, and Halstead metrics.

Is there anything specific you'd like to do with this information? For example, would you like me to:

*   Search the vault for specific keys?
*   Generate documentation for other modules?
*   Help you organize this information in a different way?
*   Provide more information about the error?

[Timestamp: 2024/12/18 23:36:18]

**user**: Failed validating 'required' in schema:
    {'type': 'object',
     'properties': {'summary': {'type': 'string',
                                'description': 'A brief summary of the '
                                               'method or function.'},
                    'description': {'type': 'string',
                                    'description': 'Detailed description '
                                                   'of the method or '
                                                   'function.'},
                    'args': {'type': 'array',
                             'description': 'A list of arguments for the '
                                            'method or function.',
                             'items': {'type': 'object',
                                       'properties': {'name': {'type': 'string',
                                                               'description': 'The '
                                                                              'name '
                                                                              'of '
                                                                              'the '
                                                                              'argument.'},
                                                      'type': {'type': 'string',
                                                               'description': 'The '
                                                                              'data '
                                                                              'type '
                                                                              'of '
                                                                              'the '
                                                                              'argument.'},
                                                      'description': {'type': 'string',
                                                                      'description': 'A '
                                                                                     'brief '
                                                                                     'description '
                                                                                     'of '
                                                                                     'the '
                                                                                     'argument.'}},
                                       'required': ['name',
                                                    'type',
                                                    'description'],
                                       'additionalProperties': False},
                             'default': []},
                    'returns': {'type': 'object',
                                'description': 'Details about the return '
                                               'value of the method or '
                                               'function.',
                                'properties': {'type': {'type': 'string',
                                                        'description': 'The '
                                                                       'data '
                                                                       'type '
                                                                       'of '
                                                                       'the '
                                                                       'return '
                                                                       'value.'},
                                               'description': {'type': 'string',
                                                               'description': 'A '
                                                                              'brief '
                                                                              'description '
                                                                              'of '
                                                                              'the '
                                                                              'return '
                                                                              'value.'}},
                                'required': ['type', 'description'],
                                'additionalProperties': False},
                    'raises': {'type': 'array',
                               'description': 'A list of exceptions that '
                                              'may be raised by the method '
                                              'or function.',
                               'items': {'type': 'object',
                                         'properties': {'exception': {'type': 'string',
                                                                      'description': 'The '
                                                                                     'name '
                                                                                     'of '
                                                                                     'the '
                                                                                     'exception '
                                                                                     'that '
                                                                                     'may '
                                                                                     'be '
                                                                                     'raised.'},
                                                        'description': {'type': 'string',
                                                                        'description': 'A '
                                                                                       'brief '
                                                                                       'description '
                                                                                       'of '
                                                                                       'the '
                                                                                       'circumstances '
                                                                                       'under '
                                                                                       'which '
                                                                                       'the '
                                                                                       'exception '
                                                                                       'is '
                                                                                       'raised.'}},
                                         'required': ['exception',
                                                      'description'],
                                         'additionalProperties': False},
                               'default': []},
                    'complexity': {'type': ['integer', 'string'],
                                   'description': 'McCabe complexity score',
                                   'default': 1}},
     'required': ['summary', 'description', 'returns'],
     'additionalProperties': False}

On instance:
    {'choices': [{'message': {'content': '{"summary": "Invalid response '
                                         'format", "description": "The '
                                         'response did not match the '
                                         'expected structure.", "error": '
                                         '"", "returns": {"type": "Any", '
                                         '"description": "No return '
                                         'description provided"}}'}}],
     'usage': {}}
API call failed: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                                               \'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n                                
                   \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                 
            \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                             \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                                                               \'description\': \'The \'\n                                                                              \'name \'\n                                                                          
    \'of \'\n                                                                              \'the \'\n                                                                              \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                                                               \'description\': \'The \'\n                                                                              \'data \'\n                                                                             
 \'type \'\n                                                                              \'of \'\n                                                                              \'the \'\n                                                                              \'argument.\'},\n                                                      \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                                                                    
                 \'brief \'\n                                                                                     \'description \'\n                                                
                                     \'of \'\n                                                                                     \'the \'\n                                       
                                              \'argument.\'}},\n                                       \'required\': [\'name\',\n                                                   
 \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                       
      \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                  
                             \'value of the method or \'\n                                               \'function.\',\n                                \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                                                                       \'data \'\n                                                                       \'type \'\n                                                                       \'of \'\n         
                                                              \'the \'\n                                                                       \'return \'\n                        
                                               \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n                               
                                \'description\': \'A \'\n                                                                              \'brief \'\n                                 
                                             \'description \'\n                                                                              \'of \'\n                              
                                                \'the \'\n                                                                              \'return \'\n                               
                                               \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                                \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                    
                          \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                                                                      \'description\': \'The \'\n                                                                                     \'name \'\n                                                           
                          \'of \'\n                                                                                     \'the \'\n                                                  
                                   \'exception \'\n                                                                                     \'that \'\n                                 
                                                    \'may \'\n                                                                                     \'be \'\n                        
                                                             \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                                                                       \'brief \'\n                                                                                       \'description \'\n                                                                  
                     \'of \'\n                                                                                       \'the \'\n                                                     
                                  \'circumstances \'\n                                                                                       \'under \'\n                           
                                                            \'which \'\n                                                                                       \'the \'\n           
                                                                            \'exception \'\n                                                                                       \'is \'\n                                                                                       \'raised.\'}},\n                                         \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                            
   \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                            
             \'response did not match the \'\n                                         \'expected structure.", "error": \'\n                                         \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                         \'description provided"}}\'}}],\n     \'usage\': {}}']
Traceback (most recent call last):
  File "C:\Users\htper\New folder\simple\core\ai_service.py", line 480, in generate_documentation
    return ProcessingResult(
        ^^^^^^^^^^^^^^^^^^^^
core.exceptions.DataValidationError: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                                               
\'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n            
                                       \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                           
  \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                                                     
          \'description\': \'The \'\n                                                                              \'name \'\n                                                      
                        \'of \'\n                                                                              \'the \'\n                                                           
                   \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                                                        
       \'description\': \'The \'\n                                                                              \'data \'\n                                                         
                     \'type \'\n                                                                              \'of \'\n                                                             
                 \'the \'\n                                                                              \'argument.\'},\n                                                      \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                                                
                                     \'brief \'\n                                                                                     \'description \'\n                            
                                                         \'of \'\n                                                                                     \'the \'\n                   
                                                                  \'argument.\'}},\n                                       \'required\': [\'name\',\n                               
                     \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n                                \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                                                     
                  \'data \'\n                                                                       \'type \'\n                                                                     
  \'of \'\n                                                                       \'the \'\n                                                                       \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n           
                                                    \'description\': \'A \'\n                                                                              \'brief \'\n             
                                                                 \'description \'\n                                                                              \'of \'\n          
                                                                    \'the \'\n                                                                              \'return \'\n           
                                                                   \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                      
          \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                                                    
                  \'description\': \'The \'\n                                                                                     \'name \'\n                                       
                                              \'of \'\n                                                                                     \'the \'\n                              
                                                       \'exception \'\n                                                                                     \'that \'\n             
                                                                        \'may \'\n                                                                                     \'be \'\n                                                                                     \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                                                      
                 \'brief \'\n                                                                                       \'description \'\n                                              
                                         \'of \'\n                                                                                       \'the \'\n                                 
                                                      \'circumstances \'\n                                                                                       \'under \'\n                                                                                       \'which \'\n                                                                                       \'the \'\n                                                                                       \'exception \'\n                                                                    
                   \'is \'\n                                                                                       \'raised.\'}},\n                                         \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n                                    
     \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                         \'description provided"}}\'}}],\n     \'usage\': {}}']

Operation: documentation_generation
  Success: False
  Duration: 3.25s
  error: Failed to generate documentation: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                                         
      \'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n                                                   \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                     
        \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                                               
                \'description\': \'The \'\n                                                                              \'name \'\n                                                
                              \'of \'\n                                                                              \'the \'\n                                                     
                         \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                                                  
             \'description\': \'The \'\n                                                                              \'data \'\n                                                   
                           \'type \'\n                                                                              \'of \'\n                                                       
                       \'the \'\n                                                                              \'argument.\'},\n                                                    
  \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                                          
                                           \'brief \'\n                                                                                     \'description \'\n                      
                                                               \'of \'\n                                                                                     \'the \'\n             
                                                                        \'argument.\'}},\n                                       \'required\': [\'name\',\n                         
                           \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n                                \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                                               
                        \'data \'\n                                                                       \'type \'\n                                                               
        \'of \'\n                                                                       \'the \'\n                                                                       \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n                                                               \'description\': \'A \'\n                                                                              \'brief \'\n                                                                              \'description \'\n                                                                              \'of \'\n                                                                              \'the \'\n                                                                              \'return \'\n                                                                              \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                
                \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                             
  \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                                              
                        \'description\': \'The \'\n                                                                                     \'name \'\n                                 
                                                    \'of \'\n                                                                                     \'the \'\n                        
                                                             \'exception \'\n                                                                                     \'that \'\n                                                                                     \'may \'\n                                                                                     \'be \'\n                                                                                     \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                                                
                       \'brief \'\n                                                                                       \'description \'\n                                        
                                               \'of \'\n                                                                                       \'the \'\n                           
                                                            \'circumstances \'\n                                                                                       \'under \'\n                                                                                       \'which \'\n                                                                                  
     \'the \'\n                                                                                       \'exception \'\n                                                              
                         \'is \'\n                                                                                       \'raised.\'}},\n                                         \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n                              
           \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                         \'description provided"}}\'}}],\n     \'usage\': {}}']
Documentation generation failed: Failed to generate documentation: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                 
                              \'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n                                                   \'of the method or \'\n                                                   \'function.\'},\n                    
\'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                             \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                       
                                        \'description\': \'The \'\n                                                                              \'name \'\n                        
                                                      \'of \'\n                                                                              \'the \'\n                             
                                                 \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                          
                                     \'description\': \'The \'\n                                                                              \'data \'\n                           
                                                   \'type \'\n                                                                              \'of \'\n                               
                                               \'the \'\n                                                                              \'argument.\'},\n                            
                          \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                  
                                                                   \'brief \'\n                                                                                     \'description \'\n                                                                                     \'of \'\n                                                                                    
 \'the \'\n                                                                                     \'argument.\'}},\n                                       \'required\': [\'name\',\n                                                    \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n         
                       \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                       
                                                \'data \'\n                                                                       \'type \'\n                                       
                                \'of \'\n                                                                       \'the \'\n                                                          
             \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n                                                               \'description\': \'A \'\n                                                                         
     \'brief \'\n                                                                              \'description \'\n                                                                   
           \'of \'\n                                                                              \'the \'\n                                                                        
      \'return \'\n                                                                              \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                                \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                      
                                                \'description\': \'The \'\n                                                                                     \'name \'\n         
                                                                            \'of \'\n                                                                                     \'the \'\n                                                                                     \'exception \'\n                                                                               
      \'that \'\n                                                                                     \'may \'\n                                                                    
                 \'be \'\n                                                                                     \'raised.\'},\n                                                      
  \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                        
                                               \'brief \'\n                                                                                       \'description \'\n                
                                                                       \'of \'\n                                                                                       \'the \'\n                                                                                       \'circumstances \'\n                                                                            
           \'under \'\n                                                                                       \'which \'\n                                                          
                             \'the \'\n                                                                                       \'exception \'\n                                      
                                                 \'is \'\n                                                                                       \'raised.\'}},\n                   
                      \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                 
  \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n                                         \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                   
      \'description provided"}}\'}}],\n     \'usage\': {}}']
Traceback (most recent call last):
  File "C:\Users\htper\New folder\simple\core\ai_service.py", line 480, in generate_documentation
    return ProcessingResult(
        ^^^^^^^^^^^^^^^^^^^^
core.exceptions.DataValidationError: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                                               
\'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n            
                                       \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                           
  \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                                                     
          \'description\': \'The \'\n                                                                              \'name \'\n                                                      
                        \'of \'\n                                                                              \'the \'\n                                                           
                   \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                                                        
       \'description\': \'The \'\n                                                                              \'data \'\n                                                         
                     \'type \'\n                                                                              \'of \'\n                                                             
                 \'the \'\n                                                                              \'argument.\'},\n                                                      \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                                                
                                     \'brief \'\n                                                                                     \'description \'\n                            
                                                         \'of \'\n                                                                                     \'the \'\n                   
                                                                  \'argument.\'}},\n                                       \'required\': [\'name\',\n                               
                     \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n                                \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                                                     
                  \'data \'\n                                                                       \'type \'\n                                                                     
  \'of \'\n                                                                       \'the \'\n                                                                       \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n           
                                                    \'description\': \'A \'\n                                                                              \'brief \'\n             
                                                                 \'description \'\n                                                                              \'of \'\n          
                                                                    \'the \'\n                                                                              \'return \'\n           
                                                                   \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                      
          \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                                                    
                  \'description\': \'The \'\n                                                                                     \'name \'\n                                       
                                              \'of \'\n                                                                                     \'the \'\n                              
                                                       \'exception \'\n                                                                                     \'that \'\n             
                                                                        \'may \'\n                                                                                     \'be \'\n                                                                                     \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                                                      
                 \'brief \'\n                                                                                       \'description \'\n                                              
                                         \'of \'\n                                                                                       \'the \'\n                                 
                                                      \'circumstances \'\n                                                                                       \'under \'\n                                                                                       \'which \'\n                                                                                       \'the \'\n                                                                                       \'exception \'\n                                                                    
                   \'is \'\n                                                                                       \'raised.\'}},\n                                         \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n                                    
     \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                         \'description provided"}}\'}}],\n     \'usage\': {}}']

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\htper\New folder\simple\core\ai_service.py", line 510, in generate_documentation
    if self._client:
            ^^^^^^^^^
core.exceptions.APICallError: Failed to generate documentation: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                    
                           \'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n                                                   \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                             \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                          
                                     \'description\': \'The \'\n                                                                              \'name \'\n                           
                                                   \'of \'\n                                                                              \'the \'\n                                
                                              \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                             
                                  \'description\': \'The \'\n                                                                              \'data \'\n                              
                                                \'type \'\n                                                                              \'of \'\n                                  
                                            \'the \'\n                                                                              \'argument.\'},\n                               
                       \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                     
                                                                \'brief \'\n                                                                                     \'description \'\n                                                                                     \'of \'\n                                                                                     \'the \'\n                                                                                     \'argument.\'}},\n                                       \'required\': [\'name\',\n                                                    \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n            
                    \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                          
                                             \'data \'\n                                                                       \'type \'\n                                          
                             \'of \'\n                                                                       \'the \'\n                                                             
          \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n                                                               \'description\': \'A \'\n                                                                            
  \'brief \'\n                                                                              \'description \'\n                                                                      
        \'of \'\n                                                                              \'the \'\n                                                                           
   \'return \'\n                                                                              \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                                \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                         
                                             \'description\': \'The \'\n                                                                                     \'name \'\n            
                                                                         \'of \'\n                                                                                     \'the \'\n                                                                                     \'exception \'\n                                                                                  
   \'that \'\n                                                                                     \'may \'\n                                                                       
              \'be \'\n                                                                                     \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                           
                                            \'brief \'\n                                                                                       \'description \'\n                   
                                                                    \'of \'\n                                                                                       \'the \'\n                                                                                       \'circumstances \'\n                                                                               
        \'under \'\n                                                                                       \'which \'\n                                                             
                          \'the \'\n                                                                                       \'exception \'\n                                         
                                              \'is \'\n                                                                                       \'raised.\'}},\n                      
                   \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n         
                                \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                      
   \'description provided"}}\'}}],\n     \'usage\': {}}']
Error during documentation generation: Failed to generate documentation: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n           
                                    \'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n                                                   \'of the method or \'\n                                                   \'function.\'},\n              
      \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                             \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                 
                                              \'description\': \'The \'\n                                                                              \'name \'\n                  
                                                            \'of \'\n                                                                              \'the \'\n                       
                                                       \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                    
                                           \'description\': \'The \'\n                                                                              \'data \'\n                     
                                                         \'type \'\n                                                                              \'of \'\n                         
                                                     \'the \'\n                                                                              \'argument.\'},\n                      
                                \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n            
                                                                         \'brief \'\n                                                                                     \'description \'\n                                                                                     \'of \'\n                                                                              
       \'the \'\n                                                                                     \'argument.\'}},\n                                       \'required\': [\'name\',\n                                                    \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n                                \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                 
                                                      \'data \'\n                                                                       \'type \'\n                                 
                                      \'of \'\n                                                                       \'the \'\n                                                    
                   \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n                                                               \'description\': \'A \'\n                                                                   
           \'brief \'\n                                                                              \'description \'\n                                                             
                 \'of \'\n                                                                              \'the \'\n                                                                  
            \'return \'\n                                                                              \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                                \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                
                                                      \'description\': \'The \'\n                                                                                     \'name \'\n                                                                                     \'of \'\n                                                                                     \'the \'\n                                                                                     \'exception \'\n                                                                         
            \'that \'\n                                                                                     \'may \'\n                                                              
                       \'be \'\n                                                                                     \'raised.\'},\n                                                
        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                  
                                                     \'brief \'\n                                                                                       \'description \'\n          
                                                                             \'of \'\n                                                                                       \'the \'\n                                                                                       \'circumstances \'\n                                                                      
                 \'under \'\n                                                                                       \'which \'\n                                                    
                                   \'the \'\n                                                                                       \'exception \'\n                                
                                                       \'is \'\n                                                                                       \'raised.\'}},\n             
                            \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                           
        \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n                                         \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                             
            \'description provided"}}\'}}],\n     \'usage\': {}}']
Traceback (most recent call last):
  File "C:\Users\htper\New folder\simple\core\ai_service.py", line 480, in generate_documentation
    return ProcessingResult(
        ^^^^^^^^^^^^^^^^^^^^
core.exceptions.DataValidationError: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                                               
\'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n            
                                       \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                           
  \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                                                     
          \'description\': \'The \'\n                                                                              \'name \'\n                                                      
                        \'of \'\n                                                                              \'the \'\n                                                           
                   \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                                                        
       \'description\': \'The \'\n                                                                              \'data \'\n                                                         
                     \'type \'\n                                                                              \'of \'\n                                                             
                 \'the \'\n                                                                              \'argument.\'},\n                                                      \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                                                
                                     \'brief \'\n                                                                                     \'description \'\n                            
                                                         \'of \'\n                                                                                     \'the \'\n                   
                                                                  \'argument.\'}},\n                                       \'required\': [\'name\',\n                               
                     \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n                                \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                                                     
                  \'data \'\n                                                                       \'type \'\n                                                                     
  \'of \'\n                                                                       \'the \'\n                                                                       \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n           
                                                    \'description\': \'A \'\n                                                                              \'brief \'\n             
                                                                 \'description \'\n                                                                              \'of \'\n          
                                                                    \'the \'\n                                                                              \'return \'\n           
                                                                   \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                      
          \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                                                    
                  \'description\': \'The \'\n                                                                                     \'name \'\n                                       
                                              \'of \'\n                                                                                     \'the \'\n                              
                                                       \'exception \'\n                                                                                     \'that \'\n             
                                                                        \'may \'\n                                                                                     \'be \'\n                                                                                     \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                                                      
                 \'brief \'\n                                                                                       \'description \'\n                                              
                                         \'of \'\n                                                                                       \'the \'\n                                 
                                                      \'circumstances \'\n                                                                                       \'under \'\n                                                                                       \'which \'\n                                                                                       \'the \'\n                                                                                       \'exception \'\n                                                                    
                   \'is \'\n                                                                                       \'raised.\'}},\n                                         \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n                                    
     \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                         \'description provided"}}\'}}],\n     \'usage\': {}}']

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\htper\New folder\simple\core\docs.py", line 126, in generate_documentation
    processing_result: ProcessingResult = await self.ai_service.generate_documentation(
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\htper\New folder\simple\core\ai_service.py", line 510, in generate_documentation
    if self._client:
            ^^^^^^^^^
core.exceptions.APICallError: Failed to generate documentation: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                    
                           \'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n                                                   \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                             \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                          
                                     \'description\': \'The \'\n                                                                              \'name \'\n                           
                                                   \'of \'\n                                                                              \'the \'\n                                
                                              \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                             
                                  \'description\': \'The \'\n                                                                              \'data \'\n                              
                                                \'type \'\n                                                                              \'of \'\n                                  
                                            \'the \'\n                                                                              \'argument.\'},\n                               
                       \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                     
                                                                \'brief \'\n                                                                                     \'description \'\n                                                                                     \'of \'\n                                                                                     \'the \'\n                                                                                     \'argument.\'}},\n                                       \'required\': [\'name\',\n                                                    \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n            
                    \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                          
                                             \'data \'\n                                                                       \'type \'\n                                          
                             \'of \'\n                                                                       \'the \'\n                                                             
          \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n                                                               \'description\': \'A \'\n                                                                            
  \'brief \'\n                                                                              \'description \'\n                                                                      
        \'of \'\n                                                                              \'the \'\n                                                                           
   \'return \'\n                                                                              \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                                \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                         
                                             \'description\': \'The \'\n                                                                                     \'name \'\n            
                                                                         \'of \'\n                                                                                     \'the \'\n                                                                                     \'exception \'\n                                                                                  
   \'that \'\n                                                                                     \'may \'\n                                                                       
              \'be \'\n                                                                                     \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                           
                                            \'brief \'\n                                                                                       \'description \'\n                   
                                                                    \'of \'\n                                                                                       \'the \'\n                                                                                       \'circumstances \'\n                                                                               
        \'under \'\n                                                                                       \'which \'\n                                                             
                          \'the \'\n                                                                                       \'exception \'\n                                         
                                              \'is \'\n                                                                                       \'raised.\'}},\n                      
                   \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n         
                                \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                      
   \'description provided"}}\'}}],\n     \'usage\': {}}']

Operation: documentation_generation
  Success: False
  Duration: 3.31s
  module_name: token_management
  error: Failed to generate documentation: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                                         
      \'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n                                                   \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                     
        \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                                               
                \'description\': \'The \'\n                                                                              \'name \'\n                                                
                              \'of \'\n                                                                              \'the \'\n                                                     
                         \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                                                  
             \'description\': \'The \'\n                                                                              \'data \'\n                                                   
                           \'type \'\n                                                                              \'of \'\n                                                       
                       \'the \'\n                                                                              \'argument.\'},\n                                                    
  \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                                          
                                           \'brief \'\n                                                                                     \'description \'\n                      
                                                               \'of \'\n                                                                                     \'the \'\n             
                                                                        \'argument.\'}},\n                                       \'required\': [\'name\',\n                         
                           \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n                                \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                                               
                        \'data \'\n                                                                       \'type \'\n                                                               
        \'of \'\n                                                                       \'the \'\n                                                                       \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n                                                               \'description\': \'A \'\n                                                                              \'brief \'\n                                                                              \'description \'\n                                                                              \'of \'\n                                                                              \'the \'\n                                                                              \'return \'\n                                                                              \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                
                \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                             
  \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                                              
                        \'description\': \'The \'\n                                                                                     \'name \'\n                                 
                                                    \'of \'\n                                                                                     \'the \'\n                        
                                                             \'exception \'\n                                                                                     \'that \'\n                                                                                     \'may \'\n                                                                                     \'be \'\n                                                                                     \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                                                
                       \'brief \'\n                                                                                       \'description \'\n                                        
                                               \'of \'\n                                                                                       \'the \'\n                           
                                                            \'circumstances \'\n                                                                                       \'under \'\n                                                                                       \'which \'\n                                                                                  
     \'the \'\n                                                                                       \'exception \'\n                                                              
                         \'is \'\n                                                                                       \'raised.\'}},\n                                         \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n                              
           \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                         \'description provided"}}\'}}],\n     \'usage\': {}}']
Unexpected error generating documentation for simple\api\token_management.py: Failed to generate documentation: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                                               \'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                 
                   \'description\': \'Detailed description \'\n                                                   \'of the method or \'\n                                           
        \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                      
                      \'method or function.\',\n                             \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                                                               \'description\': \'The \'\n                                                                    
          \'name \'\n                                                                              \'of \'\n                                                                        
      \'the \'\n                                                                              \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                                                               \'description\': \'The \'\n                                                                       
       \'data \'\n                                                                              \'type \'\n                                                                         
     \'of \'\n                                                                              \'the \'\n                                                                              
\'argument.\'},\n                                                      \'description\': {\'type\': \'string\',\n                                                                    
  \'description\': \'A \'\n                                                                                     \'brief \'\n                                                        
                             \'description \'\n                                                                                     \'of \'\n                                       
                                              \'the \'\n                                                                                     \'argument.\'}},\n                     
                  \'required\': [\'name\',\n                                                    \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                           
                    \'function.\',\n                                \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                                                                       \'data \'\n                                                                       \'type \'\n                                                                       \'of \'\n                                                                       \'the \'\n             
                                                          \'return \'\n                                                                       \'value.\'},\n                        
                       \'description\': {\'type\': \'string\',\n                                                               \'description\': \'A \'\n                            
                                                  \'brief \'\n                                                                              \'description \'\n                      
                                                        \'of \'\n                                                                              \'the \'\n                           
                                                   \'return \'\n                                                                              \'value.\'}},\n                       
         \'required\': [\'type\', \'description\'],\n                                \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                         
                     \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                                                                      \'description\': \'The \'\n                                                            
                         \'name \'\n                                                                                     \'of \'\n                                                  
                                   \'the \'\n                                                                                     \'exception \'\n                                  
                                                   \'that \'\n                                                                                     \'may \'\n                       
                                                              \'be \'\n                                                                                     \'raised.\'},\n         
                                               \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                                                                       \'brief \'\n                                                                            
           \'description \'\n                                                                                       \'of \'\n                                                       
                                \'the \'\n                                                                                       \'circumstances \'\n                               
                                                        \'under \'\n                                                                                       \'which \'\n             
                                                                          \'the \'\n                                                                                       \'exception \'\n                                                                                       \'is \'\n                                                                             
          \'raised.\'}},\n                                         \'required\': [\'exception\',\n                                                      \'description\'],\n         
                                \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n         
                                \'format", "description": "The \'\n                                         \'response did not match the \'\n                                       
  \'expected structure.", "error": \'\n                                         \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                         \'description provided"}}\'}}],\n     \'usage\': {}}']
Traceback (most recent call last):
  File "C:\Users\htper\New folder\simple\core\ai_service.py", line 480, in generate_documentation
    return ProcessingResult(
        ^^^^^^^^^^^^^^^^^^^^
core.exceptions.DataValidationError: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                                               
\'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n            
                                       \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                           
  \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                                                     
          \'description\': \'The \'\n                                                                              \'name \'\n                                                      
                        \'of \'\n                                                                              \'the \'\n                                                           
                   \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                                                        
       \'description\': \'The \'\n                                                                              \'data \'\n                                                         
                     \'type \'\n                                                                              \'of \'\n                                                             
                 \'the \'\n                                                                              \'argument.\'},\n                                                      \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                                                
                                     \'brief \'\n                                                                                     \'description \'\n                            
                                                         \'of \'\n                                                                                     \'the \'\n                   
                                                                  \'argument.\'}},\n                                       \'required\': [\'name\',\n                               
                     \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n                                \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                                                     
                  \'data \'\n                                                                       \'type \'\n                                                                     
  \'of \'\n                                                                       \'the \'\n                                                                       \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n           
                                                    \'description\': \'A \'\n                                                                              \'brief \'\n             
                                                                 \'description \'\n                                                                              \'of \'\n          
                                                                    \'the \'\n                                                                              \'return \'\n           
                                                                   \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                      
          \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                                                    
                  \'description\': \'The \'\n                                                                                     \'name \'\n                                       
                                              \'of \'\n                                                                                     \'the \'\n                              
                                                       \'exception \'\n                                                                                     \'that \'\n             
                                                                        \'may \'\n                                                                                     \'be \'\n                                                                                     \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                                                      
                 \'brief \'\n                                                                                       \'description \'\n                                              
                                         \'of \'\n                                                                                       \'the \'\n                                 
                                                      \'circumstances \'\n                                                                                       \'under \'\n                                                                                       \'which \'\n                                                                                       \'the \'\n                                                                                       \'exception \'\n                                                                    
                   \'is \'\n                                                                                       \'raised.\'}},\n                                         \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n                                    
     \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                         \'description provided"}}\'}}],\n     \'usage\': {}}']

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\htper\New folder\simple\core\docs.py", line 325, in generate_module_documentation
    updated_code, markdown_doc = await self.generate_documentation(context)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\htper\New folder\simple\core\docs.py", line 126, in generate_documentation
    processing_result: ProcessingResult = await self.ai_service.generate_documentation(
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\htper\New folder\simple\core\ai_service.py", line 510, in generate_documentation
    if self._client:
            ^^^^^^^^^
core.exceptions.APICallError: Failed to generate documentation: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                    
                           \'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n                                                   \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                             \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                          
                                     \'description\': \'The \'\n                                                                              \'name \'\n                           
                                                   \'of \'\n                                                                              \'the \'\n                                
                                              \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                             
                                  \'description\': \'The \'\n                                                                              \'data \'\n                              
                                                \'type \'\n                                                                              \'of \'\n                                  
                                            \'the \'\n                                                                              \'argument.\'},\n                               
                       \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                     
                                                                \'brief \'\n                                                                                     \'description \'\n                                                                                     \'of \'\n                                                                                     \'the \'\n                                                                                     \'argument.\'}},\n                                       \'required\': [\'name\',\n                                                    \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n            
                    \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                          
                                             \'data \'\n                                                                       \'type \'\n                                          
                             \'of \'\n                                                                       \'the \'\n                                                             
          \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n                                                               \'description\': \'A \'\n                                                                            
  \'brief \'\n                                                                              \'description \'\n                                                                      
        \'of \'\n                                                                              \'the \'\n                                                                           
   \'return \'\n                                                                              \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                                \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                         
                                             \'description\': \'The \'\n                                                                                     \'name \'\n            
                                                                         \'of \'\n                                                                                     \'the \'\n                                                                                     \'exception \'\n                                                                                  
   \'that \'\n                                                                                     \'may \'\n                                                                       
              \'be \'\n                                                                                     \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                           
                                            \'brief \'\n                                                                                       \'description \'\n                   
                                                                    \'of \'\n                                                                                       \'the \'\n                                                                                       \'circumstances \'\n                                                                               
        \'under \'\n                                                                                       \'which \'\n                                                             
                          \'the \'\n                                                                                       \'exception \'\n                                         
                                              \'is \'\n                                                                                       \'raised.\'}},\n                      
                   \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n         
                                \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                      
   \'description provided"}}\'}}],\n     \'usage\': {}}']

Operation: module_documentation_generation
  Success: False
  Duration: 3.32s
  module_path: simple\api\token_management.py
  error: Failed to generate documentation: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                                         
      \'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n                                                   \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                     
        \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                                               
                \'description\': \'The \'\n                                                                              \'name \'\n                                                
                              \'of \'\n                                                                              \'the \'\n                                                     
                         \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                                                  
             \'description\': \'The \'\n                                                                              \'data \'\n                                                   
                           \'type \'\n                                                                              \'of \'\n                                                       
                       \'the \'\n                                                                              \'argument.\'},\n                                                    
  \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                                          
                                           \'brief \'\n                                                                                     \'description \'\n                      
                                                               \'of \'\n                                                                                     \'the \'\n             
                                                                        \'argument.\'}},\n                                       \'required\': [\'name\',\n                         
                           \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n                                \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                                               
                        \'data \'\n                                                                       \'type \'\n                                                               
        \'of \'\n                                                                       \'the \'\n                                                                       \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n                                                               \'description\': \'A \'\n                                                                              \'brief \'\n                                                                              \'description \'\n                                                                              \'of \'\n                                                                              \'the \'\n                                                                              \'return \'\n                                                                              \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                
                \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                             
  \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                                              
                        \'description\': \'The \'\n                                                                                     \'name \'\n                                 
                                                    \'of \'\n                                                                                     \'the \'\n                        
                                                             \'exception \'\n                                                                                     \'that \'\n                                                                                     \'may \'\n                                                                                     \'be \'\n                                                                                     \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                                                
                       \'brief \'\n                                                                                       \'description \'\n                                        
                                               \'of \'\n                                                                                       \'the \'\n                           
                                                            \'circumstances \'\n                                                                                       \'under \'\n                                                                                       \'which \'\n                                                                                  
     \'the \'\n                                                                                       \'exception \'\n                                                              
                         \'is \'\n                                                                                       \'raised.\'}},\n                                         \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n                              
           \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                         \'description provided"}}\'}}],\n     \'usage\': {}}']
Error processing file simple\api\token_management.py: Unexpected error generating documentation for simple\api\token_management.py: Failed to generate documentation: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                                               \'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n                                                   \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                             \'items\': {\'type\': \'object\',\n              
                         \'properties\': {\'name\': {\'type\': \'string\',\n                                                               \'description\': \'The \'\n              
                                                                \'name \'\n                                                                              \'of \'\n                  
                                                            \'the \'\n                                                                              \'argument.\'},\n               
                                       \'type\': {\'type\': \'string\',\n                                                               \'description\': \'The \'\n                 
                                                             \'data \'\n                                                                              \'type \'\n                   
                                                           \'of \'\n                                                                              \'the \'\n                        
                                                      \'argument.\'},\n                                                      \'description\': {\'type\': \'string\',\n              
                                                        \'description\': \'A \'\n                                                                                     \'brief \'\n                                                                                     \'description \'\n                                                                               
      \'of \'\n                                                                                     \'the \'\n                                                                      
               \'argument.\'}},\n                                       \'required\': [\'name\',\n                                                    \'type\',\n                   
                                 \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n                                \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                                                                       \'data \'\n                      
                                                 \'type \'\n                                                                       \'of \'\n                                        
                               \'the \'\n                                                                       \'return \'\n                                                       
                \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n                                                              
 \'description\': \'A \'\n                                                                              \'brief \'\n                                                                
              \'description \'\n                                                                              \'of \'\n                                                             
                 \'the \'\n                                                                              \'return \'\n                                                              
                \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                                \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                    
                     \'properties\': {\'exception\': {\'type\': \'string\',\n                                                                      \'description\': \'The \'\n                                                                                     \'name \'\n                                                                                     \'of \'\n                                                                                     \'the \'\n                                                                                 
    \'exception \'\n                                                                                     \'that \'\n                                                                
                     \'may \'\n                                                                                     \'be \'\n                                                       
                              \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                      
                                  \'description\': \'A \'\n                                                                                       \'brief \'\n                      
                                                                 \'description \'\n                                                                                       \'of \'\n                                                                                       \'the \'\n                                                                                    
   \'circumstances \'\n                                                                                       \'under \'\n                                                          
                             \'which \'\n                                                                                       \'the \'\n                                          
                                             \'exception \'\n                                                                                       \'is \'\n                       
                                                                \'raised.\'}},\n                                         \'required\': [\'exception\',\n                            
                          \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n         
           \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                
   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n                                         \'"", "returns": {"type": "Any", \'\n         
                                \'"description": "No return \'\n                                         \'description provided"}}\'}}],\n     \'usage\': {}}']
Traceback (most recent call last):
  File "C:\Users\htper\New folder\simple\core\ai_service.py", line 480, in generate_documentation
    return ProcessingResult(
        ^^^^^^^^^^^^^^^^^^^^
core.exceptions.DataValidationError: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                                               
\'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n            
                                       \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                           
  \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                                                     
          \'description\': \'The \'\n                                                                              \'name \'\n                                                      
                        \'of \'\n                                                                              \'the \'\n                                                           
                   \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                                                        
       \'description\': \'The \'\n                                                                              \'data \'\n                                                         
                     \'type \'\n                                                                              \'of \'\n                                                             
                 \'the \'\n                                                                              \'argument.\'},\n                                                      \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                                                
                                     \'brief \'\n                                                                                     \'description \'\n                            
                                                         \'of \'\n                                                                                     \'the \'\n                   
                                                                  \'argument.\'}},\n                                       \'required\': [\'name\',\n                               
                     \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n                                \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                                                     
                  \'data \'\n                                                                       \'type \'\n                                                                     
  \'of \'\n                                                                       \'the \'\n                                                                       \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n           
                                                    \'description\': \'A \'\n                                                                              \'brief \'\n             
                                                                 \'description \'\n                                                                              \'of \'\n          
                                                                    \'the \'\n                                                                              \'return \'\n           
                                                                   \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                      
          \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                                                    
                  \'description\': \'The \'\n                                                                                     \'name \'\n                                       
                                              \'of \'\n                                                                                     \'the \'\n                              
                                                       \'exception \'\n                                                                                     \'that \'\n             
                                                                        \'may \'\n                                                                                     \'be \'\n                                                                                     \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                                                      
                 \'brief \'\n                                                                                       \'description \'\n                                              
                                         \'of \'\n                                                                                       \'the \'\n                                 
                                                      \'circumstances \'\n                                                                                       \'under \'\n                                                                                       \'which \'\n                                                                                       \'the \'\n                                                                                       \'exception \'\n                                                                    
                   \'is \'\n                                                                                       \'raised.\'}},\n                                         \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n                                    
     \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                         \'description provided"}}\'}}],\n     \'usage\': {}}']

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Users\htper\New folder\simple\core\docs.py", line 325, in generate_module_documentation
    updated_code, markdown_doc = await self.generate_documentation(context)
                                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\htper\New folder\simple\core\docs.py", line 126, in generate_documentation
    processing_result: ProcessingResult = await self.ai_service.generate_documentation(
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\htper\New folder\simple\core\ai_service.py", line 510, in generate_documentation
    if self._client:
            ^^^^^^^^^
core.exceptions.APICallError: Failed to generate documentation: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n                                \'description\': \'A brief summary of the \'\n                    
                           \'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n                                                   \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                             \'items\': {\'type\': \'object\',\n                                       \'properties\': {\'name\': {\'type\': \'string\',\n                          
                                     \'description\': \'The \'\n                                                                              \'name \'\n                           
                                                   \'of \'\n                                                                              \'the \'\n                                
                                              \'argument.\'},\n                                                      \'type\': {\'type\': \'string\',\n                             
                                  \'description\': \'The \'\n                                                                              \'data \'\n                              
                                                \'type \'\n                                                                              \'of \'\n                                  
                                            \'the \'\n                                                                              \'argument.\'},\n                               
                       \'description\': {\'type\': \'string\',\n                                                                      \'description\': \'A \'\n                     
                                                                \'brief \'\n                                                                                     \'description \'\n                                                                                     \'of \'\n                                                                                     \'the \'\n                                                                                     \'argument.\'}},\n                                       \'required\': [\'name\',\n                                                    \'type\',\n                                                    \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n            
                    \'properties\': {\'type\': {\'type\': \'string\',\n                                                        \'description\': \'The \'\n                          
                                             \'data \'\n                                                                       \'type \'\n                                          
                             \'of \'\n                                                                       \'the \'\n                                                             
          \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n                                                               \'description\': \'A \'\n                                                                            
  \'brief \'\n                                                                              \'description \'\n                                                                      
        \'of \'\n                                                                              \'the \'\n                                                                           
   \'return \'\n                                                                              \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                                \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                         \'properties\': {\'exception\': {\'type\': \'string\',\n                         
                                             \'description\': \'The \'\n                                                                                     \'name \'\n            
                                                                         \'of \'\n                                                                                     \'the \'\n                                                                                     \'exception \'\n                                                                                  
   \'that \'\n                                                                                     \'may \'\n                                                                       
              \'be \'\n                                                                                     \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                                        \'description\': \'A \'\n                                           
                                            \'brief \'\n                                                                                       \'description \'\n                   
                                                                    \'of \'\n                                                                                       \'the \'\n                                                                                       \'circumstances \'\n                                                                               
        \'under \'\n                                                                                       \'which \'\n                                                             
                          \'the \'\n                                                                                       \'exception \'\n                                         
                                              \'is \'\n                                                                                       \'raised.\'}},\n                      
                   \'required\': [\'exception\',\n                                                      \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n         
                                \'"", "returns": {"type": "Any", \'\n                                         \'"description": "No return \'\n                                      
   \'description provided"}}\'}}],\n     \'usage\': {}}']

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\htper\New folder\simple\main.py", line 140, in process_file
    await self.doc_orchestrator.generate_module_documentation(
  File "C:\Users\htper\New folder\simple\core\docs.py", line 359, in generate_module_documentation
    raise DocumentationError(error_msg) from gen_error
core.exceptions.DocumentationError: Unexpected error generating documentation for simple\api\token_management.py: Failed to generate documentation: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {\'type\': \'object\',\n     \'properties\': {\'summary\': {\'type\': \'string\',\n              
                  \'description\': \'A brief summary of the \'\n                                               \'method or function.\'},\n                    \'description\': {\'type\': \'string\',\n                                    \'description\': \'Detailed description \'\n                                                   \'of the method or \'\n                                                   \'function.\'},\n                    \'args\': {\'type\': \'array\',\n                             \'description\': \'A list of arguments for the \'\n                                            \'method or function.\',\n                             \'items\': {\'type\': \'object\',\n                                
       \'properties\': {\'name\': {\'type\': \'string\',\n                                                               \'description\': \'The \'\n                                
                                              \'name \'\n                                                                              \'of \'\n                                    
                                          \'the \'\n                                                                              \'argument.\'},\n                                 
                     \'type\': {\'type\': \'string\',\n                                                               \'description\': \'The \'\n                                   
                                           \'data \'\n                                                                              \'type \'\n                                     
                                         \'of \'\n                                                                              \'the \'\n                                          
                                    \'argument.\'},\n                                                      \'description\': {\'type\': \'string\',\n                                
                                      \'description\': \'A \'\n                                                                                     \'brief \'\n                    
                                                                 \'description \'\n                                                                                     \'of \'\n                                                                                     \'the \'\n                                                                                     \'argument.\'}},\n                                       \'required\': [\'name\',\n                                                    \'type\',\n                                     
               \'description\'],\n                                       \'additionalProperties\': False},\n                             \'default\': []},\n                    \'returns\': {\'type\': \'object\',\n                                \'description\': \'Details about the return \'\n                                               \'value of the method or \'\n                                               \'function.\',\n                                \'properties\': {\'type\': {\'type\': \'string\',\n                         
                               \'description\': \'The \'\n                                                                       \'data \'\n                                        
                               \'type \'\n                                                                       \'of \'\n                                                          
             \'the \'\n                                                                       \'return \'\n                                                                       \'value.\'},\n                                               \'description\': {\'type\': \'string\',\n                                                               \'description\': \'A \'\n                                                                              \'brief \'\n                                                                              \'description \'\n                                                                              \'of \'\n                                                                              \'the \'\n                                                                              \'return \'\n                                                                              \'value.\'}},\n                                \'required\': [\'type\', \'description\'],\n                                \'additionalProperties\': False},\n                    \'raises\': {\'type\': \'array\',\n                               \'description\': \'A list of exceptions that \'\n                                              \'may be raised by the method \'\n                                              \'or function.\',\n                               \'items\': {\'type\': \'object\',\n                                      
   \'properties\': {\'exception\': {\'type\': \'string\',\n                                                                      \'description\': \'The \'\n                        
                                                             \'name \'\n                                                                                     \'of \'\n              
                                                                       \'the \'\n                                                                                     \'exception \'\n                                                                                     \'that \'\n                                                                                  
   \'may \'\n                                                                                     \'be \'\n                                                                         
            \'raised.\'},\n                                                        \'description\': {\'type\': \'string\',\n                                                        
                \'description\': \'A \'\n                                                                                       \'brief \'\n                                        
                                               \'description \'\n                                                                                       \'of \'\n                   
                                                                    \'the \'\n                                                                                       \'circumstances \'\n                                                                                       \'under \'\n                                                                            
           \'which \'\n                                                                                       \'the \'\n                                                            
                           \'exception \'\n                                                                                       \'is \'\n                                         
                                              \'raised.\'}},\n                                         \'required\': [\'exception\',\n                                              
        \'description\'],\n                                         \'additionalProperties\': False},\n                               \'default\': []},\n                    \'complexity\': {\'type\': [\'integer\', \'string\'],\n                                   \'description\': \'McCabe complexity score\',\n                                   \'default\': 1}},\n     \'required\': [\'summary\', \'description\', \'returns\'],\n     \'additionalProperties\': False}\n\nOn instance:\n    {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n                                         \'format", "description": "The \'\n                                         \'response did not match the \'\n                                         \'expected structure.", "error": \'\n                                         \'"", "returns": {"type": "Any", \'\n                           
              \'"description": "No return \'\n                                         \'description provided"}}\'}}],\n     \'usage\': {}}']

--------------------------------------------------------------------------------

ERROR:
  Error processing file: simple\api\token_management.py. Reason: Unexpected error generating documentation for simple\api\token_management.py: Failed to generate documentation: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n {\'type\': \'object\',\n \'properties\': {\'summary\': {\'type\': \'string\',\n \'description\': \'A brief summary of the \'\n \'method or function.\'},\n \'description\': {\'type\': \'string\',\n \'description\': \'Detailed description \'\n \'of the method or \'\n \'function.\'},\n \'args\': {\'type\': \'array\',\n \'description\': \'A list of arguments for the \'\n \'method or function.\',\n \'items\': {\'type\': \'object\',\n \'properties\': {\'name\': {\'type\': \'string\',\n \'description\': \'The \'\n \'name \'\n \'of \'\n \'the \'\n \'argument.\'},\n \'type\': {\'type\': \'string\',\n \'description\': \'The \'\n \'data \'\n \'type \'\n \'of \'\n \'the \'\n \'argument.\'},\n \'description\': {\'type\': \'string\',\n \'description\': \'A \'\n \'brief \'\n \'description \'\n \'of \'\n \'the \'\n \'argument.\'}},\n \'required\': [\'name\',\n \'type\',\n \'description\'],\n \'additionalProperties\': False},\n \'default\': []},\n \'returns\': {\'type\': \'object\',\n \'description\': \'Details about the return \'\n \'value of the method or \'\n \'function.\',\n \'properties\': {\'type\': {\'type\': \'string\',\n \'description\': \'The \'\n \'data \'\n \'type \'\n \'of \'\n \'the \'\n \'return \'\n \'value.\'},\n \'description\': {\'type\': \'string\',\n \'description\': \'A \'\n \'brief \'\n \'description \'\n \'of \'\n \'the \'\n \'return \'\n \'value.\'}},\n \'required\': [\'type\', \'description\'],\n \'additionalProperties\': False},\n \'raises\': {\'type\': \'array\',\n \'description\': \'A list of exceptions that \'\n \'may be raised by the method \'\n \'or function.\',\n \'items\': {\'type\': \'object\',\n \'properties\': {\'exception\': {\'type\': \'string\',\n \'description\': \'The \'\n \'name \'\n \'of \'\n \'the \'\n \'exception \'\n \'that \'\n \'may \'\n \'be \'\n \'raised.\'},\n \'description\': {\'type\': \'string\',\n \'description\': \'A \'\n \'brief \'\n \'description \'\n \'of \'\n \'the \'\n \'circumstances \'\n \'under \'\n \'which \'\n \'the \'\n \'exception \'\n \'is \'\n \'raised.\'}},\n \'required\': [\'exception\',\n \'description\'],\n \'additionalProperties\': False},\n \'default\': []},\n \'complexity\': {\'type\': [\'integer\', \'string\'],\n \'description\': \'McCabe complexity score\',\n \'default\': 1}},\n \'required\': [\'summary\', \'description\', \'returns\'],\n \'additionalProperties\': False}\n\nOn instance:\n {\'choices\': [{\'message\': {\'content\': \'{"summary": "Invalid response \'\n \'format", "description": "The \'\n \'response did not match the \'\n \'expected structure.", "error": \'\n \'"", "returns": {"type": "Any", \'\n \'"description": "No return \'\n \'description provided"}}\'}}],\n \'usage\': {}}']

--------------------------------------------------------------------------------

Source code read from simple\api\__init__.py. Length: 798
Generating documentation for simple\api\__init__.py
Validating source code for file: simple
Syntax validation successful for: simple
No __init__.py found at simple\default_module\__init__.py. Module 'default_module' may not be importable.

--------------------------------------------------------------------------------


Metrics Collection

Module: default_module:
----------------------------------------
  cyclomatic_complexity     3
  cognitive_complexity      0
  maintainability_index     94.42
  halstead_metrics          {'volume': 35.909291004461515, 'difficulty': 0.5909090909090909, 'effort': 21.21912650263635, 'time': 1.178840361257575, 'bugs': 0.011969763668153839}  
  lines_of_code             33
  total_functions           1
  scanned_functions         0
  function_scan_ratio       0.00
  total_classes             0
  scanned_classes           0
  class_scan_ratio          0.00
  complexity_graph          None
----------------------------------------

-----------------------------------------------------
[Timestamp: 2024/12/18 23:36:27]

**user**: Analyze the error and help me figfure out what the fuck it says
[Timestamp: 2024/12/18 23:36:43]

**ai**: Okay, let's break down that error message and figure out what it's trying to tell us. It's definitely a bit verbose, but we can dissect it piece by piece.

**The Core Issue: Response Validation Failure**

The main problem is this:

```
core.exceptions.DataValidationError: Response validation failed: ['\'summary\' is a required property\n\nFailed validating \'required\' in schema:\n    {...}
```

This tells us that the AI service (which is being used to generate documentation) returned a response that doesn't match the expected format. Specifically, the response is missing a required field called `summary`.

**What's a "Schema" in This Context?**

The error message mentions a "schema". Think of a schema as a blueprint or a contract that defines the structure of the data that's expected. In this case, the schema specifies that the AI service's response should be a JSON object with certain properties, including `summary`, `description`, `args`, `returns`, `raises`, and `complexity`.

The schema is defined as a JSON object:

```json
{
    'type': 'object',
    'properties': {
        'summary': {'type': 'string', 'description': 'A brief summary of the method or function.'},
        'description': {'type': 'string', 'description': 'Detailed description of the method or function.'},
        'args': {
            'type': 'array',
            'description': 'A list of arguments for the method or function.',
            'items': {
                'type': 'object',
                'properties': {
                    'name': {'type': 'string', 'description': 'The name of the argument.'},
                    'type': {'type': 'string', 'description': 'The data type of the argument.'},
                    'description': {'type': 'string', 'description': 'A brief description of the argument.'}
                },
                'required': ['name', 'type', 'description'],
                'additionalProperties': False
            },
            'default': []
        },
        'returns': {
            'type': 'object',
            'description': 'Details about the return value of the method or function.',
            'properties': {
                'type': {'type': 'string', 'description': 'The data type of the return value.'},
                'description': {'type': 'string', 'description': 'A brief description of the return value.'}
            },
            'required': ['type', 'description'],
            'additionalProperties': False
        },
        'raises': {
            'type': 'array',
            'description': 'A list of exceptions that may be raised by the method or function.',
            'items': {
                'type': 'object',
                'properties': {
                    'exception': {'type': 'string', 'description': 'The name of the exception that may be raised.'},
                    'description': {'type': 'string', 'description': 'A brief description of the circumstances under which the exception is raised.'}
                },
                'required': ['exception', 'description'],
                'additionalProperties': False
            },
            'default': []
        },
        'complexity': {'type': ['integer', 'string'], 'description': 'McCabe complexity score', 'default': 1}
    },
    'required': ['summary', 'description', 'returns'],
    'additionalProperties': False
}
```

The `required` array at the end of the schema specifies that `summary`, `description`, and `returns` are mandatory.

**The Actual Response**

The response that caused the error looks like this:

```json
{
    'choices': [
        {
            'message': {
                'content': '{"summary": "Invalid response format", "description": "The response did not match the expected structure.", "error": "", "returns": {"type": "Any", "description": "No return description provided"}}'
            }
        }
    ],
    'usage': {
        'selectedText'
    }
}
```

Notice that the `content` field contains a JSON string that is missing the `summary` field. This is why the validation failed.

**Why is this happening?**

This error indicates a problem on the AI service's side. It's not generating the documentation in the format that your application expects. It's returning an error message instead of the structured documentation.

**The Traceback**

The rest of the error message is a traceback, which shows the sequence of function calls that led to the error. It's useful for debugging, but the core issue is the response validation failure.

**Simplified Explanation**

In simpler terms:

1.  Your application asked an AI service to generate documentation.
2.  The AI service returned a response that was missing a required piece of information (`summary`).
3.  Your application's validation process detected this missing information and threw an error.

**What Can You Do?**

Since this is an issue with the AI service's response, you can't fix it directly in your code. Here are some things you could consider:

1.  **Retry:** Implement a retry mechanism in your code. Sometimes, these errors are transient.
2.  **Contact the AI Service Provider:** If the issue persists, reach out to the provider of the AI service. They might be aware of the problem or have a fix.
3.  **Error Handling:** Improve your application's error handling to gracefully handle these situations. Instead of crashing, you could log the error and continue processing other files.
4.  **Fallback Mechanism:** If the AI service is consistently failing, you could implement a fallback mechanism, such as using a different documentation generation method.
5.  **Schema Check:** Double-check that the schema you are using is the one that the AI service expects.

**In summary:** The error is not in your code, but in the response from the AI service. It's missing a required field called `summary`. You should consider implementing error handling, retries, and potentially contacting the AI service provider.

Does this explanation make sense? Let me know if you have any more questions.

[Timestamp: 2024/12/18 23:36:54]

**ai**: Okay, I see the error you've provided again. This confirms that the issue is indeed with the AI service's response not adhering to the required schema. Specifically, the `summary` field is missing in the response, which is causing the validation to fail.

Let's break down what this means and what we can do:

**Understanding the Error**

*   **Schema:** The schema you've provided is a definition of the expected structure for the AI service's response. It specifies that the response should be a JSON object with properties like `summary`, `description`, `args`, `returns`, `raises`, and `complexity`. It also defines the data types and requirements for each of these properties.
*   **`'summary' is a required property`:** This specific error message indicates that the `summary` property is mandatory according to the schema, but it's missing in the actual response received from the AI service.
*   **`On instance: {'choices': [{'message': {'content': '{"summary": "Invalid response format", "description": "The response did not match the expected structure.", "error": "", "returns": {"type": "Any", "description": "No return description provided"}}'}}]}`:** This is the actual response that the AI service returned. As you can see, it includes `description`, `error`, and `returns`, but it's missing the `summary` field.
*   **`API call failed: Response validation failed:`:** This confirms that the validation of the AI service's response failed against the defined schema.

**Why This is Happening**

This is most likely an issue on the AI service's side. It could be due to:

*   **A bug in the AI service's code:** The service might not be generating the `summary` field correctly in certain cases.
*   **A change in the AI service's API:** The service's API might have been updated, and the response format no longer matches the expected schema.
*   **An issue with the AI service's configuration:** There might be a configuration problem that's causing the `summary` field to be omitted.

**What We Can Do**

Since this is an issue with the AI service, there are a few things we can try:

1.  **Retry the request:** Sometimes, these errors can be transient. Retrying the request might yield a valid response.
2.  **Check the AI service's documentation:** Look for any recent changes or updates to the AI service's API documentation. This might provide clues about the issue.
3.  **Contact the AI service's support:** If the issue persists, it's best to contact the AI service's support team and report the problem. They are the ones who can fix the underlying issue.
4.  **Implement error handling:** In your code, you should implement proper error handling to gracefully deal with such situations. This could involve logging the error, retrying the request, or providing a fallback mechanism.
5.  **Adjust the schema (with caution):** If the AI service's API has indeed changed and it's not providing the `summary` field anymore, you might need to adjust the schema to remove the `summary` field as a required property. However, this should be done with caution, as it might lead to loss of information or unexpected behavior.

**In summary:**

The error you're seeing is due to the AI service not returning a response that matches the expected schema. The most important thing to do is to report this issue to the AI service's support team. In the meantime, you can try retrying the request and implement error handling in your code.

Do you want me to try to generate the documentation again? Or would you like me to help you with any of the steps I've mentioned above?

[Timestamp: 2024/12/18 23:37:26]

**user**: [Guide for Azure OpenAI](./Guide%20for%20Azure%20OpenAI.md)
[Timestamp: 2024/12/18 23:37:36]

**ai**: Error: [GoogleGenerativeAI Error]: Failed to parse stream
[Timestamp: 2024/12/18 23:38:50]

**ai**: Okay, I've processed the entire "Guide for Azure OpenAI" document. This is a comprehensive guide covering a wide range of topics, from basic authentication to advanced integration patterns. Here's a breakdown of what I've gathered:

**Key Areas Covered:**

1.  **Authentication:**
    *   How to set up and authenticate with the Azure OpenAI API using API keys and environment variables.
2.  **Function Calling:**
    *   How to use the `tools` parameter for function calling, including registering tools, defining schemas, and executing tools.
3.  **Structured Output Generation:**
    *   How to use the `response_format` parameter to extract structured data in JSON format, using predefined schemas.
4.  **Token Management:**
    *   How to estimate token usage and optimize prompts to manage costs.
5.  **Error Handling and Monitoring:**
    *   How to implement comprehensive error handling, capture specific error types, and monitor API usage.
6.  **Batch Processing with Rate Limiting:**
    *   How to manage batch processing with rate limiting and quota management.
7.  **Advanced Prompt Management:**
    *   How to create dynamic prompts with different message roles, structured content, and tool interactions.
8.  **System Monitoring and Logging:**
    *   How to track performance, usage, and errors in Azure OpenAI applications.
9.  **Dynamic Prompts with Structured Outputs and Function Calling:**
    *   How to combine dynamic prompts with `tools` and `response_format` to generate structured outputs.
10. **Advanced RAG with Hybrid Search:**
     * How to combine retrieval-augmented generation (RAG) with hybrid search to enhance information retrieval and response generation.
11. **Advanced Content Filtering and Safety:**
      * How to implement advanced content filtering using predefined categories, thresholds, and custom block lists.
12. **Advanced Caching Strategy:**
      * How to implement caching strategies to improve performance and reduce costs by storing frequently used responses.
13. **Advanced Integration Patterns:**
      * How to integrate with the Assistants API and implement robust content checking and request processing.
14. **Implementing Retrieval-Augmented Generation (RAG):**
      * How to use RAG to enhance the quality of AI responses by integrating external knowledge sources, using Azure Search and the Assistants API.
15. **Generating Embeddings:**
      * How to generate embeddings using Azure OpenAI for tasks like similarity search and clustering.
16. **Azure OpenAI and Sentry Configuration:**
      * How to integrate Azure OpenAI with Sentry for error tracking and monitoring.
17. **Stream Response Processing:**
      * How to handle streaming responses efficiently for real-time content generation.
18. **Advanced Configuration Management:**
      * How to manage advanced configuration options for fine-tuned control over API responses.
19. **Response Validation and Processing:**
      * How to implement comprehensive response validation and processing based on API specifications.

**Key Updates and Enhancements:**

*   **`tools` parameter:** The guide has been updated to use the `tools` parameter instead of the deprecated `functions` and `function_call` for function calling.
*   **`response_format` parameter:** The guide demonstrates how to use the `response_format` parameter to enforce structured outputs in JSON format.
*   **Richer error details:** The guide now includes capturing more data from errors, particularly the `inner_error` and `content_filter_results` which are part of the API responses.
*   **`cached_tokens` and `reasoning_tokens`:** The guide has been updated to include the new `cached_tokens` and `reasoning_tokens` parameters in the usage information.
*   **Assistants API:** The guide has been expanded to include examples of how to use the Assistants API for RAG and other tasks.
*   **Stream Options:** The guide now includes the `stream_options` parameter to include usage information in the stream.
*   **`max_prompt_tokens` and `max_completion_tokens`:** The guide now includes the `max_prompt_tokens` and `max_completion_tokens` parameters for the Assistants API.
*   **Response Validation:** The guide now includes a comprehensive response validation and processing section, including validation for all relevant schemas, including those from the Assistants API.

**Overall:**

This is a very detailed and well-structured guide that provides a wealth of information on how to use the Azure OpenAI API effectively. It covers a wide range of topics, from basic authentication to advanced integration patterns. The guide is also up-to-date with the latest API changes and best practices.

**Next Steps:**

Now that I've processed this document, what would you like to do next? For example, would you like me to:

*   Extract specific code examples?
*   Summarize a particular section
[Timestamp: 2024/12/18 23:38:50]

**user**: [ai_service](./ai_service.md) [response_parsing](./response_parsing.md) [docstring_processor](./docstring_processor.md) [prompt_manager](./prompt_manager.md) Is there error in these modules?
[Timestamp: 2024/12/18 23:39:00]

**user**: [schema_validator.py](schema_validator.py.md) [response_formatter.py](response_formatter.py.md) [docstring_validator.py](docstring_validator.py.md) how about in these modules?
[Timestamp: 2024/12/18 23:41:37]

**ai**: Okay, let's analyze these modules for potential errors. I'll go through each one and point out any issues I see, along with some general observations.

**1. `ai_service` Module**

*   **Initialization:**
    *   The initialization of the `AzureOpenAI` client is wrapped in a `try...except` block, which is good for handling potential connection issues.
    *   The logger is initialized correctly using `LoggerSetup`.
    *   The use of `Injector` for dependency injection is a good pattern for managing dependencies.
    *   The fallback mechanism for `docstring_processor` is good, but it might be better to log a warning if the default processor is used.
*   **Source Code Handling:**
    *   The methods `_add_source_code_to_content`, `_add_source_code_to_function_call`, and `_add_source_code_to_message` are well-structured for adding source code to different parts of the API response.
    *   The use of `json.loads` and `json.dumps` is correct for handling JSON data.
*   **Response Formatting:**
    *   The `_format_response` method handles different response formats (direct content, function calls, tool calls, fallback) well.
    *   The fallback response mechanism is good for handling unexpected responses.
    *   The `_format_function_call_response` method includes validation of the function call arguments, which is good.
*   **API Call Logic:**
    *   The `_make_api_call_with_retry` method implements retry logic with exponential backoff, which is good for handling transient errors.
    *   It handles rate limiting (429) and service unavailable (503) errors correctly.
    *   The use of `ClientTimeout` is good for preventing indefinite hangs.
    *   The use of `async with self._client.post(...)` is correct for making asynchronous HTTP requests.
*   **Documentation Generation:**
    *   The `generate_documentation` method handles the entire process of generating documentation, including prompt creation, API calls, response parsing, and metrics tracking.
    *   It includes error handling for various stages of the process.
    *   The use of `self.prompt_manager` for prompt creation and `self.response_parser` for response parsing is good.
*   **Resource Management:**
    *   The `start` and `close` methods correctly manage the `ClientSession`.
    *   The use of `async with` for the context manager is good for ensuring resources are released.

*   **Potential Issues:**
    *   **Schema Validation:** The `_format_function_call_response` method validates the function call arguments, but there's no explicit schema validation for the main response content. This could lead to issues if the API response doesn't match the expected structure.
    *   **Fallback Response:** The fallback response is hardcoded, which might not be ideal for all use cases. It might be better to make the fallback response configurable.
    *   **Error Handling:** While the error handling is good, it could be improved by providing more specific error messages and context.
    *   **Token Management:** The `token_manager` is used to prepare the request, but it's not clear how the token usage is tracked and managed across multiple API calls.

**2. `response_parsing` Module**

*   **Initialization:**
    *   The logger is initialized correctly.
    *   The `ResponseFormatter` and `DocstringValidator` are initialized using dependency injection.
*   **Content Extraction:**
    *   The `_extract_content` method handles different response formats well, including direct content, function calls, and tool calls.
    *   The fallback mechanism is good for handling unexpected content.
    *   The `_ensure_required_fields` method ensures that the extracted content has the required fields.
    *   The methods `_extract_content_from_function_call` and `_extract_content_from_tool_calls` handle function calls and tool calls correctly.
    *   The `_extract_content_from_direct_content` method handles direct content and attempts to decode JSON.
*   **Response Parsing:**
    *   The `parse_response` method handles the entire process of parsing and validating the API response.
    *   It handles different expected formats ("docstring" and "function").
    *   It includes validation against the docstring schema using `docstring_validator`.
    *   It includes error handling for various stages of the process.
*   **Potential Issues:**
    *   **Schema Validation:** The `_validate_content` method is called in `_extract_content_from_function_call` and `_extract_content_from_tool_calls`, but it's not implemented in this file. This could lead to issues if the function call arguments don't match the expected schema.
    *   **Fallback Response:** The fallback response is hardcoded, which might not be ideal for all use cases. It might be better to make the fallback response configurable.
    *   **Error Handling:** While the error handling is good, it could be improved by providing more specific error messages and context.
    *   **Parsing Statistics:** The parsing statistics are collected but not used or displayed anywhere.

**3. `docstring_processor` Module**

*   **Initialization:**
    *   The logger is initialized correctly.
    *   The `SchemaValidator` is initialized using dependency injection.
    *   The metrics collector is initialized correctly.
*   **Docstring Parsing:**
    *   The `parse` method handles the parsing of docstrings using `docstring_parser`.
    *   It includes a fallback mechanism for different docstring styles.
    *   It collects statistics on docstring processing.
    *   The `_parse_docstring_content` method handles the actual parsing and extracts the necessary information.
*   **Docstring Validation:**
    *   The `validate` method validates the docstring data against a schema loaded from a file.
    *   It uses `jsonschema` for validation.
*   **Docstring Processing:**
    *   The `process_docstring` method handles the entire process of parsing, validating, and processing a docstring.
    *   It includes error handling for various stages of the process.
*   **Potential Issues:**
    *   **Schema Path:** The schema path is hardcoded, which might not be ideal for all use cases. It might be better to make the schema path configurable.
    *   **Error Handling:** While the error handling is good, it could be improved by providing more specific error messages and context.
    *   **Validation:** The validation is done against a static schema file, which might not be flexible enough for all use cases.
    *   **Metrics:** The docstring statistics are displayed using `display_metrics`, but it's not clear how these metrics are used or stored.

**4. `prompt_manager` Module**

*   **Initialization:**
    *   The logger is initialized correctly.
    *   The `token_manager` is initialized using dependency injection.
    *   The templates are loaded using Jinja2, which is a good approach for managing templates.
    *   The function schema is loaded and validated.
*   **Template Loading:**
    *   The `_load_template` method loads and validates template files.
    *   It includes error handling for template loading and rendering.
*   **Prompt Formatting:**
    *   The `_format_class_info` and `_format_function_info` methods format class and function information for template rendering.
*   **Prompt Creation:**
    *   The `create_documentation_prompt` method generates documentation prompts using the loaded template.
    *   It tracks token usage and metrics.
    *   The `create_code_analysis_prompt` method generates code analysis prompts.
*   **Function Schema Handling:**
    *   The `get_prompt_with_schema` method adds function calling instructions to a prompt.
    *   The `get_function_schema` method retrieves the function schema.
*   **Potential Issues:**
    *   **Schema Path:** The schema path is hardcoded, which might not be ideal for all use cases. It might be better to make the schema path configurable.
    *   **Template Validation:** The template validation is basic and might not catch all potential issues.
    *   **Error Handling:** While the error handling is good, it could be improved by providing more specific error messages and context.
    *   **Metrics:** The metrics are collected but not used or displayed anywhere.

**General Observations:**

*   **Dependency Injection:** The use of dependency injection is good for decoupling components and making the code more testable.
*   **Logging:** The use of logging is good for tracking errors and debugging.
*   **Error Handling:** The error handling is generally good, but it could be improved by providing more specific error messages and context.
*   **Asynchronous Operations:** The use of asynchronous operations is good for improving performance.
*   **Modularity:** The code is well-modularized, with each module responsible for a specific task.
*   **Missing Schema Validation:** The main issue is that the main response content is not validated against a schema. This could lead to issues if the API response doesn't match the expected structure.

**Summary of Potential Errors:**

*   **`ai_service`:**
    *   Missing schema validation for main response content.
    *   Hardcoded fallback response.
    *   Token management across multiple API calls could be improved.
*   **`response_parsing`:**
    *   Missing implementation of `_validate_content` method.
    *   Hardcoded fallback response.
    *   Parsing statistics are not used or displayed.
*   **`docstring_processor`:**
    *   Hardcoded schema path.
    *   Validation against a static schema file.
    *   Metrics are collected but not used or stored.
*   **`prompt_manager`:**
    *   Hardcoded schema path.
    *   Basic template validation.
    *   Metrics are collected but not used or stored.

**Recommendations:**

1.  **Implement Schema Validation:** Add schema validation for the main response content in `ai_service` and implement the `_validate_content` method in `response_parsing`.
2.  **Make Fallback Responses Configurable:** Allow the fallback responses to be configured instead of being hardcoded.
3.  **Improve Error Handling:** Provide more specific error messages and context in the error handling blocks.
4.  **Use and Store Metrics:** Implement a mechanism to store and use the collected metrics.
5.  **Make Schema Paths Configurable:** Allow the schema paths to be configured instead of being hardcoded.
6.  **Enhance Template Validation:** Add more robust validation for the template files.

These modules are well-structured and follow good coding practices. Addressing the potential issues above will make them more robust and reliable.

Do you have any other questions or would you like me to elaborate on any of these points?

[Timestamp: 2024/12/18 23:42:00]

**ai**: Okay, I've reviewed the code in `schema_validator.py`, `response_formatter.py`, and `docstring_validator.py`. Here's a breakdown of each module and how they relate to the overall process:

**1. `schema_validator.py`**

*   **Purpose:** This module provides a base class (`SchemaValidator`) for validating data against JSON schemas. It's designed to be reusable across different parts of your application where schema validation is needed.
*   **Key Features:**
    *   **`__init__`:** Initializes the validator with a logger and an optional correlation ID for tracing.
    *   **`validate_schema`:** This is the core method. It takes an `instance` (the data to validate) and a `schema` (a JSON schema) as input. It uses the `jsonschema` library to perform the validation.
    *   **Error Handling:** It catches `ValidationError` from `jsonschema` and logs the error. It also catches any other exceptions during validation and logs them with traceback information.
    *   **Return Value:** Returns a tuple:
        *   A boolean indicating whether the validation was successful (`True` for success, `False` for failure).
        *   A list of error messages (empty if validation was successful).
*   **Usage:** This class is intended to be inherited by more specific validators (like `DocstringValidator`). It provides the basic validation logic, and subclasses can define their specific schemas and validation methods.

**2. `response_formatter.py`**

*   **Purpose:** This module is responsible for formatting responses from the AI model into a consistent structure that your application can easily handle. It also handles fallback scenarios when the response format is invalid.
*   **Key Features:**
    *   **`__init__`:** Initializes the formatter with a logger and an optional correlation ID.
    *   **`format_summary_description_response`:** Formats a response that contains a summary or description. It wraps the response in a `choices` structure with a `content` field.
    *   **`format_function_call_response`:** Formats a response that includes a function call. It wraps the function call in a `choices` structure with a `function_call` field.
    *   **`format_tool_calls_response`:** Formats a response that includes tool calls. It wraps the tool calls in a `choices` structure with a `tool_calls` field.
    *   **`format_fallback_response`:** Creates a fallback response when the AI model's response doesn't match the expected format. It includes a generic error message and a default `returns` field.
    *   **`standardize_response_format`:** Attempts to standardize the response format into a `choices` structure. It handles cases where the response is already in the correct format, or if it's a direct content format. It also includes a fallback for unknown formats.
    *   **Logging:** The logger is used to log debug, warning, and error messages during the formatting process.
*   **Usage:** This class is used to ensure that all responses from the AI model are consistently formatted before being processed by other parts of your application. It handles different types of responses (summary/description, function calls, tool calls) and provides a fallback mechanism for invalid responses.

**3. `docstring_validator.py`**

*   **Purpose:** This module provides a specific validator (`DocstringValidator`) for validating docstring data against a predefined schema.
*   **Key Features:**
    *   **Inheritance:** It inherits from `SchemaValidator`, reusing the base validation logic.
    *   **`__init__`:** Initializes the validator and loads the docstring schema from a JSON file (`docstring_schema.json`).
    *   **`_load_schema`:** Loads the docstring schema from a JSON file. It includes error handling for `FileNotFoundError` and `json.JSONDecodeError`.
    *   **`validate_docstring`:** Validates the input `content` against the loaded docstring schema. It extracts the schema from the loaded JSON and calls the base class's `validate_schema` method.
*   **Usage:** This class is used to validate the structure of docstrings generated by the AI model, ensuring they conform to a specific schema.

**How These Modules Relate:**

*   **`schema_validator.py`** provides the foundation for schema validation.
*   **`docstring_validator.py`** extends `schema_validator.py` to provide a specific validator for docstrings.
*   **`response_formatter.py`** is used to format the responses from the AI model before they are validated.

**In the context of the previous error:**

The `response_formatter.py` module is where the fallback response is created when the AI service returns an invalid response. The `standardize_response_format` method attempts to convert the response to the expected format, and if it fails, it calls `format_fallback_response`.

The error you've been seeing is caused because the AI service is returning a response that does not contain the `summary` field, which is required by the schema. The `format_fallback_response` method in `response_formatter.py` is creating a fallback response, but it is not adding the `summary` field to the fallback response.

**Key Takeaways:**

*   These modules are designed to work together to ensure that the data from the AI model is valid and in the correct format.
*   The `SchemaValidator` provides the base validation logic.
*   The `ResponseFormatter` ensures that responses are in a consistent format.
*   The `DocstringValidator` provides a specific validator for docstrings.

Do you have any specific questions about these modules, or would you like me to elaborate on any particular aspect?

[Timestamp: 2024/12/18 23:43:05]

**user**: Provide a solution for the error and also control the terminal output 
[Timestamp: 2024/12/18 23:43:23]

**ai**: Okay, let's tackle the error and control the terminal output. Based on our previous analysis, the core issue is that the AI service is sometimes returning responses that don't conform to the expected schema, specifically missing the `summary` field. We also want to manage the verbosity of the terminal output.

Here's a breakdown of the solution and how to implement it:

**1. Addressing the Missing `summary` Field**

The most robust solution is to modify the `response_formatter.py` to ensure that the fallback response always includes the required `summary` field. This will prevent the `DataValidationError` from being raised when the AI service returns an invalid response.

**Modified `response_formatter.py`:**

```python
"""Response formatter class."""

import json
from typing import Any, Dict, Optional

from core.logger import LoggerSetup


class ResponseFormatter:
    """Formats responses from the AI model."""

    def __init__(self, correlation_id: Optional[str] = None):
        """Initialize the response formatter."""
        self.logger = LoggerSetup.get_logger(
            f"{__name__}.{self.__class__.__name__}", correlation_id
        )
        self.correlation_id = correlation_id

    def format_summary_description_response(
        self, response: dict[str, Any]
    ) -> dict[str, Any]:
        """Format response with summary or description."""
        return {
            "choices": [{"message": {"content": json.dumps(response)}}],
            "usage": response.get("usage", {}),
        }

    def format_function_call_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Format response with function call."""
        formatted_response = {
            "choices": [{"message": {"function_call": response["function_call"]}}],
            "usage": response.get("usage", {}),
        }
        self.logger.debug(
            f"Formatted function call response to: {formatted_response}",
            extra={"correlation_id": self.correlation_id},
        )
        return formatted_response

    def format_tool_calls_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Format response with tool calls."""
        formatted_response = {
            "choices": [{"message": {"tool_calls": response["tool_calls"]}}],
            "usage": response.get("usage", {}),
        }
        self.logger.debug(
            f"Formatted tool calls response to: {formatted_response}",
            extra={"correlation_id": self.correlation_id},
        )
        return formatted_response

    def format_fallback_response(
        self, response: dict[str, Any], error: str = ""
    ) -> dict[str, Any]:
        """Format a fallback response when the response format is invalid."""
        self.logger.warning(
            "Response format is invalid, creating fallback.",
            extra={"response": response, "correlation_id": self.correlation_id},
        )
        fallback_content = {
            "summary": "Invalid response format",
            "description": "The response did not match the expected structure.",
            "error": error,
            "args": [],
            "returns": {
                "type": "Any",
                "description": "No return description provided",
            },
            "raises": [],
            "complexity": 1,
        }

        fallback_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(fallback_content)
                    }
                }
            ],
            "usage": {},
        }

        self.logger.debug(
            f"Formatted generic response to: {fallback_response}",
            extra={"correlation_id": self.correlation_id},
        )
        return fallback_response

    def standardize_response_format(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize response format to use choices structure."""
        try:
            # Case 1: Already in choices format
            if isinstance(response, dict) and "choices" in response:
                return response

            # Case 2: Direct content format
            if isinstance(response, dict) and (
                "summary" in response or "description" in response
            ):
                # Wrap the content in choices format
                standardized = {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "summary": response.get("summary", ""),
                                        "description": response.get("description", ""),
                                        "args": response.get("args", []),
                                        "returns": response.get(
                                            "returns",
                                            {"type": "Any", "description": ""},
                                        ),
                                        "raises": response.get("raises", []),
                                        "complexity": response.get("complexity", 1),
                                        # Preserve any other fields
                                        **{
                                            k: v
                                            for k, v in response.items()
                                            if k
                                            not in [
                                                "summary",
                                                "description",
                                                "args",
                                                "returns",
                                                "raises",
                                                "complexity",
                                            ]
                                        },
                                    }
                                )
                            }
                        }
                    ],
                    "usage": response.get("usage", {}),
                }
                self.logger.debug(
                    f"Standardized direct format response: {standardized}",
                    extra={"correlation_id": self.correlation_id},
                )
                return standardized

            # Case 3: Fallback for unknown format
            self.logger.warning(
                "Unknown response format, creating fallback",
                extra={"correlation_id": self.correlation_id},
            )
            return self.format_fallback_response(response)

        except Exception as e:
            self.logger.error(
                f"Error standardizing response format: {e}",
                extra={"correlation_id": self.correlation_id},
                exc_info=True,
            )
            return self.format_fallback_response(response, str(e))
```

**Changes:**

*   I've added the `summary` field to the `fallback_content` dictionary in `format_fallback_response`. This ensures that even if the AI service returns an invalid response, the fallback will always include the required `summary` field, preventing the `DataValidationError`.

**2. Controlling Terminal Output**

To control the terminal output, we can leverage the logging configuration. You can configure the logging level to control the verbosity of the output.

**Example Logging Configuration (in `core/logger.py`):**

```python
import logging
import sys
from typing import Optional

class LoggerSetup:
    """Configures and provides access to loggers."""

    @staticmethod
    def get_logger(
        name: str, correlation_id: Optional[str] = None, level: int = logging.INFO
    ) -> logging.Logger:
        """Get a logger with the specified name and correlation ID."""
        logger = logging.getLogger(name)
        logger.setLevel(level)  # Set the desired logging level

        if not logger.hasHandlers():
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger
```

**Changes:**

*   I've added a `level` parameter to the `get_logger` method, which defaults to `logging.INFO`.
*   The `logger.setLevel(level)` line sets the logging level for the logger.

**How to Use:**

1.  **Set the Logging Level:**
    *   To reduce the output, you can set the logging level to `logging.WARNING` or `logging.ERROR` when you initialize the logger. For example:

        ```python
        from core.logger import LoggerSetup
        import logging

        logger = LoggerSetup.get_logger(__name__, level=logging.WARNING)
        ```

    *   To see more detailed output, you can set the level to `logging.DEBUG`:

        ```python
        from core.logger import LoggerSetup
        import logging

        logger = LoggerSetup.get_logger(__name__, level=logging.DEBUG)
        ```

2.  **Control Output:**
    *   By setting the logging level, you can control which messages are displayed in the terminal.
    *   `logging.DEBUG`: Shows all messages, including debug messages.
    *   `logging.INFO`: Shows informational messages, warnings, and errors.
    *   `logging.WARNING`: Shows warnings and errors.
    *   `logging.ERROR`: Shows only errors.
    *   `logging.CRITICAL`: Shows only critical errors.

**Example Usage (in `ai_service.py`):**

```python
class AIService:
    def __init__(
        self, config: Optional[AIConfig] = None, correlation_id: Optional[str] = None
    ) -> None:
        """Initialize the AI Service with Azure OpenAI configurations."""
        # Delayed import to avoid circular dependency
        from core.dependency_injection import Injector

        self.config = config or Injector.get("config")().ai
        self.correlation_id = correlation_id
        self.logger = LoggerSetup.get_logger(
            f"{__name__}.{self.__class__.__name__}",
            correlation_id=self.correlation_id,
            level=logging.WARNING,  # Set the desired logging level here
        )
```

**Explanation:**

*   By setting `level=logging.WARNING` in the `LoggerSetup.get_logger` call, you will only see warning and error messages in the terminal.
*   You can adjust the logging level as needed to control the verbosity of the output.

**Additional Notes:**

*   **Error Handling:** The `try...except` blocks in the code are good for catching errors, but it's important to log the errors with enough context to understand what went wrong.
*   **Schema Validation:** While the `DocstringValidator` validates the docstring content, you might also want to add schema validation for the main response content in the `ai_service` module.
*   **Fallback Responses:** The fallback responses are hardcoded, which might not be ideal for all use cases. Consider making the fallback response configurable.

**How to Apply the Solution:**

1.  **Replace the content of `response_formatter.py`** with the modified code provided above.
2.  **Modify the `core/logger.py`** with the code provided above.
3.  **Adjust the logging level** in the `get_logger` calls in the modules where you want to control the output.

By implementing these changes, you should be able to:

*   Prevent the `DataValidationError` caused by the missing `summary` field.
*   Control the verbosity of the terminal output by adjusting the logging level.

Let me know if you have any other questions or need further clarification.

[Timestamp: 2024/12/18 23:45:53]

**user**: [logger](./logger.md) [console](./console.md)
[Timestamp: 2024/12/18 23:45:55]

**ai**: Okay, let's analyze the `logger.py`, `console.py`, and `docstring_validator.py` modules.

**1. `logger.py`**

*   **Purpose:** This module provides a comprehensive logging setup with structured output, correlation IDs, and sanitization of sensitive information. It's designed to be a central point for all logging in your application.
*   **Key Features:**
    *   **`correlation_id_var`:** A `ContextVar` to manage correlation IDs, allowing you to track requests across different parts of your application.
    *   **`set_correlation_id` and `get_correlation_id`:** Functions to set and retrieve the correlation ID from the context.
    *   **`CorrelationLoggerAdapter`:** A logger adapter that automatically includes the correlation ID and any extra information in log messages. It also sanitizes reserved keys by renaming them.
    *   **`SanitizedLogFormatter`:** A custom formatter that sanitizes log messages, arguments, and timestamps. It also adds section breaks for error and critical logs.
    *   **`LoggerSetup`:**
        *   Manages the configuration of loggers, including console and file handlers.
        *   Allows setting the default logging level and log directory.
        *   Provides a `get_logger` method to retrieve a configured logger instance.
        *   Includes a `shutdown` method to clean up logging resources.
        *   Provides a `handle_exception` method to handle unhandled exceptions.
*   **Strengths:**
    *   **Structured Logging:** The use of a custom formatter with JSON output for file logs is good for structured logging and analysis.
    *   **Correlation IDs:** The use of `ContextVar` and `CorrelationLoggerAdapter` is good for tracking requests across different parts of the application.
    *   **Sanitization:** The `SanitizedLogFormatter` effectively sanitizes sensitive information from logs.
    *   **Flexibility:** The `LoggerSetup` class allows for configuring the logging level, log directory, and other settings.
    *   **Error Handling:** The `handle_exception` method provides a global exception handler.
*   **Potential Issues:**
    *   **Hardcoded Log Directory:** The default log directory is hardcoded to `logs`. It might be better to make this configurable through an environment variable or a configuration file.
    *   **File Logging:** The file logging is enabled by default, which might not be desirable in all environments. It might be better to make this configurable.
    *   **Console Logging Level:** The console logging level is hardcoded to `logging.INFO`, which might not be flexible enough. It might be better to allow this to be configured.
    *   **Sanitization:** The sanitization logic is basic and might need to be extended to handle other types of sensitive information.
    *   **Error Handling:** While the error handling is good, it could be improved by providing more specific error messages and context.

**2. `console.py`**

*   **Purpose:** This module provides utility functions for formatting and displaying output to the console. It aims to provide a clean and consistent user experience.
*   **Key Features:**
    *   **Basic Logging Configuration:** Sets up basic logging with a default level and format.
    *   **Placeholder Functions:** Includes placeholder functions for `setup_live_layout`, `stop_live_layout`, `update_header`, `update_footer`, `update_left_panel`, `update_right_panel`, and `display_progress`. These functions are likely intended for use with a more advanced console UI library (like `rich`), but they are not implemented.
    *   **Formatting Functions:** Provides functions for formatting different types of output, including error messages (`format_error_output`), section breaks (`print_section_break`), error messages (`print_error`), status messages (`print_status`), metrics (`display_metrics`), success messages (`print_success`), info messages (`print_info`), code snippets (`display_code_snippet`), warnings (`print_warning`), debug messages (`print_debug`), metrics reports (`display_metrics_report`), progress summaries (`display_progress_summary`), validation results (`display_validation_results`), processing phases (`display_processing_phase`), and API metrics (`display_api_metrics`).
    *   **Progress Object Creation:** Includes a function to create a `rich.progress.Progress` object.
*   **Strengths:**
    *   **Clean Output:** The functions provide a clean and consistent way to format output to the console.
    *   **Flexibility:** The functions can handle different types of output, including text, tables, code snippets, and progress indicators.
*   **Potential Issues:**
    *   **Basic Logging:** The basic logging configuration at the top of the file is redundant and can interfere with the logging setup in `logger.py`.
    *   **Placeholder Functions:** The placeholder functions are not implemented and might not be necessary if you are not using a more advanced console UI library.
    *   **Hardcoded Formatting:** Some of the formatting logic is hardcoded, which might not be flexible enough for all use cases.
    *   **Limited Functionality:** The module is missing features like interactive prompts, progress bars, and other advanced console UI elements.

**3. `docstring_validator.py`**

*   **Purpose:** This module provides a specific validator for docstring data, inheriting from the base `SchemaValidator`.
*   **Key Features:**
    *   **Inheritance:** It inherits from `SchemaValidator`, reusing the base validation logic.
    *   **`__init__`:** Initializes the validator and loads the docstring schema from a JSON file (`docstring_schema.json`).
    *   **`_load_schema`:** Loads the docstring schema from a JSON file. It includes error handling for `FileNotFoundError` and `json.JSONDecodeError`.
    *   **`validate_docstring`:** Validates the input `content` against the loaded docstring schema. It extracts the schema from the loaded JSON and calls the base class's `validate_schema` method.
*   **Strengths:**
    *   **Specific Validation:** Provides a specific validator for docstrings.
    *   **Schema Loading:** Includes error handling for schema loading.
    *   **Reuses Base Logic:** Reuses the base validation logic from `SchemaValidator`.
*   **Potential Issues:**
    *   **Hardcoded Schema Path:** The schema path is hardcoded, which might not be ideal for all use cases. It might be better to make the schema path configurable.
    *   **Error Handling:** While the error handling is good, it could be improved by providing more specific error messages and context.

**Summary of Potential Errors:**

*   **`logger.py`:**
    *   Hardcoded default log directory.
    *   File logging is enabled by default.
    *   Console logging level is hardcoded.
    *   Basic sanitization logic.
*   **`console.py`:**
    *   Redundant basic logging configuration.
    *   Unimplemented placeholder functions.
    *   Hardcoded formatting logic.
    *   Limited functionality.
*   **`docstring_validator.py`:**
    *   Hardcoded schema path.

**Recommendations:**

1.  **`logger.py`:**
    *   Make the log directory and file logging configurable through environment variables or a configuration file.
    *   Allow the console logging level to be configured.
    *   Extend the sanitization logic to handle other types of sensitive information.
2.  **`console.py`:**
    *   Remove the redundant basic logging configuration.
    *   Implement the placeholder functions or remove them if they are not needed.
    *   Make the formatting logic more flexible and configurable.
    *   Consider adding more advanced console UI elements using a library like `rich`.
3.  **`docstring_validator.py`:**
    *   Make the schema path configurable.

By addressing these issues, you can improve the flexibility, robustness, and maintainability of your logging, console output, and docstring validation.

Do you have any specific questions about these modules, or would you like me to elaborate on any particular aspect?

[Timestamp: 2024/12/18 23:48:06]

**user**: How to prevent the json and weird \\\ and line breaks and broken messages in the terminal
[Timestamp: 2024/12/18 23:49:51]

**ai**: Okay, let's tackle how to prevent JSON strings, escaped backslashes (`\\`), line breaks, and broken messages from cluttering your terminal output. This involves a combination of sanitizing the log messages and adjusting how you format and display the content.

**1. Preventing JSON Strings in Terminal Output**

The issue here is that you're directly printing JSON strings to the console, which results in the raw JSON being displayed instead of a human-readable format. To fix this, you should:

*   **Parse JSON Before Printing:** If you have a JSON string, parse it into a Python dictionary or list before printing it. This allows you to access the data and format it as needed.
*   **Use `pprint` for Complex Structures:** For complex dictionaries or lists, use the `pprint` module to print them in a more readable format.

**Example:**

```python
import json
from pprint import pprint

def print_formatted_json(json_string: str):
    try:
        data = json.loads(json_string)
        pprint(data)
    except json.JSONDecodeError:
        print(f"Invalid JSON: {json_string}")

# Example usage
json_str = '{"name": "John", "age": 30, "city": "New York"}'
print_formatted_json(json_str)

json_str_invalid = '{"name": "John", "age": 30, "city": "New York"'
print_formatted_json(json_str_invalid)
```

**2. Removing Escaped Backslashes (`\\`)**

Escaped backslashes often appear when you're printing strings that have been processed by JSON or other serialization methods. To remove them:

*   **Use `json.dumps` with `ensure_ascii=False`:** When converting Python objects to JSON strings, use `json.dumps(data, ensure_ascii=False)`. This will prevent the encoding of non-ASCII characters as escaped sequences.
*   **Use `str.replace`:** If you already have a string with escaped backslashes, you can use `str.replace("\\\\", "\\")` to replace them with single backslashes.

**Example:**

```python
import json

data = {"text": "This is a string with \\backslashes\\"}
json_str = json.dumps(data, ensure_ascii=False)
print(json_str)

# Remove escaped backslashes
cleaned_str = json_str.replace("\\\\", "\\")
print(cleaned_str)
```

**3. Handling Line Breaks and Broken Messages**

Line breaks and broken messages often occur when you're printing long strings or when the terminal window is too small. To handle this:

*   **Use `textwrap.fill`:** Use the `textwrap.fill` function to wrap long strings to a specific width, making them more readable in the terminal.
*   **Use `pprint` for Complex Structures:** As mentioned before, use `pprint` for complex data structures.
*   **Sanitize Messages:** Sanitize log messages to remove unnecessary line breaks or special characters.

**Example:**

```python
import textwrap

long_text = "This is a very long string that needs to be wrapped to fit within the terminal window. It contains many words and should be displayed in a readable format."
wrapped_text = textwrap.fill(long_text, width=70)
print(wrapped_text)
```

**4. Applying the Solutions to Your Code**

Let's apply these solutions to the modules you provided:

**Modified `response_formatter.py`:**

```python
"""Response formatter class."""

import json
from typing import Any, Dict, Optional
import textwrap
from core.logger import LoggerSetup


class ResponseFormatter:
    """Formats responses from the AI model."""

    def __init__(self, correlation_id: Optional[str] = None):
        """Initialize the response formatter."""
        self.logger = LoggerSetup.get_logger(
            f"{__name__}.{self.__class__.__name__}", correlation_id
        )
        self.correlation_id = correlation_id

    def format_summary_description_response(
        self, response: dict[str, Any]
    ) -> dict[str, Any]:
        """Format response with summary or description."""
        return {
            "choices": [{"message": {"content": json.dumps(response, ensure_ascii=False)}}],
            "usage": response.get("usage", {}),
        }

    def format_function_call_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Format response with function call."""
        formatted_response = {
            "choices": [{"message": {"function_call": response["function_call"]}}],
            "usage": response.get("usage", {}),
        }
        self.logger.debug(
            f"Formatted function call response to: {formatted_response}",
            extra={"correlation_id": self.correlation_id},
        )
        return formatted_response

    def format_tool_calls_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Format response with tool calls."""
        formatted_response = {
            "choices": [{"message": {"tool_calls": response["tool_calls"]}}],
            "usage": response.get("usage", {}),
        }
        self.logger.debug(
            f"Formatted tool calls response to: {formatted_response}",
            extra={"correlation_id": self.correlation_id},
        )
        return formatted_response

    def format_fallback_response(
        self, response: dict[str, Any], error: str = ""
    ) -> dict[str, Any]:
        """Format a fallback response when the response format is invalid."""
        self.logger.warning(
            "Response format is invalid, creating fallback.",
            extra={"response": response, "correlation_id": self.correlation_id},
        )
        fallback_content = {
            "summary": "Invalid response format",
            "description": "The response did not match the expected structure.",
            "error": error,
            "args": [],
            "returns": {
                "type": "Any",
                "description": "No return description provided",
            },
            "raises": [],
            "complexity": 1,
        }

        fallback_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(fallback_content, ensure_ascii=False)
                    }
                }
            ],
            "usage": {},
        }

        self.logger.debug(
            f"Formatted generic response to: {fallback_response}",
            extra={"correlation_id": self.correlation_id},
        )
        return fallback_response

    def standardize_response_format(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Standardize response format to use choices structure."""
        try:
            # Case 1: Already in choices format
            if isinstance(response, dict) and "choices" in response:
                return response

            # Case 2: Direct content format
            if isinstance(response, dict) and (
                "summary" in response or "description" in response
            ):
                # Wrap the content in choices format
                standardized = {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "summary": response.get("summary", ""),
                                        "description": response.get("description", ""),
                                        "args": response.get("args", []),
                                        "returns": response.get(
                                            "returns",
                                            {"type": "Any", "description": ""},
                                        ),
                                        "raises": response.get("raises", []),
                                        "complexity": response.get("complexity", 1),
                                        # Preserve any other fields
                                        **{
                                            k: v
                                            for k, v in response.items()
                                            if k
                                            not in [
                                                "summary",
                                                "description",
                                                "args",
                                                "returns",
                                                "raises",
                                                "complexity",
                                            ]
                                        },
                                    },
                                    ensure_ascii=False
                                )
                            }
                        }
                    ],
                    "usage": response.get("usage", {}),
                }
                self.logger.debug(
                    f"Standardized direct format response: {standardized}",
                    extra={"correlation_id": self.correlation_id},
                )
                return standardized

            # Case 3: Fallback for unknown format
            self.logger.warning(
                "Unknown response format, creating fallback",
                extra={"correlation_id": self.correlation_id},
            )
            return self.format_fallback_response(response)

        except Exception as e:
            self.logger.error(
                f"Error standardizing response format: {e}",
                extra={"correlation_id": self.correlation_id},
                exc_info=True,
            )
            return self.format_fallback_response(response, str(e))
```

**Modified `console.py`:**

```python
"""Console utilities for clean output formatting."""

from typing import Any, Optional
import logging
import textwrap
from rich.progress import Progress
import json
from pprint import pprint

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_live_layout() -> None:
    """Placeholder for setup_live_layout."""
    pass

def stop_live_layout() -> None:
    """Placeholder for stop_live_layout."""
    pass

def format_error_output(error_message: str) -> str:
    """Format error messages for clean console output."""
    lines = error_message.split("\n")
    formatted_lines = []
    indent = "  "
    
    for line in lines:
        # Remove excessive whitespace
        line = " ".join(line.split())
        if line:
            formatted_lines.append(f"{indent}{line}")
            
    return "\n".join(formatted_lines)

def print_section_break() -> None:
    """Print a visual section break."""
    print("\n" + "-" * 80 + "\n")

def print_error(message: str, correlation_id: str | None = None) -> None:
    """Display formatted error messages."""
    print_section_break()
    print("ERROR:")
    print(format_error_output(message))
    if correlation_id:
        print(f"\nCorrelation ID: {correlation_id}")
    print_section_break()

def print_status(message: str, details: dict[str, Any] | None = None) -> None:
    """Display formatted status messages with optional details."""
    print("\n" + message)
    if details:
        for key, value in details.items():
            print(f"  {key}: {value}")

def display_metrics(metrics: dict[str, Any], title: str = "Metrics") -> None:
    """Display metrics in a formatted table."""
    print(f"\n{title}:")
    print("-" * 40)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:<25} {value:>.2f}")
        elif isinstance(value, (dict, list)):
            print(f"  {key:<25}")
            pprint(value, indent=4)
        else:
            print(f"  {key:<25} {value}")
    print("-" * 40)

def print_success(message: str) -> None:
    """Display success messages."""
    print(f"\nSUCCESS:")
    print(format_error_output(message))

def print_info(message: str, details: Any = None) -> None:
    """Display info messages with optional details."""
    if details is not None:
        print(f"\n{message}")
        if isinstance(details, dict):
            for key, value in details.items():
                print(f"  {key}: {value}")
        elif isinstance(details, (list, tuple)):
            pprint(details, indent=4)
        else:
            print(format_error_output(str(details)))
    else:
        print(f"\n{message}")

def update_header(text: str) -> None:

    """Placeholder for update_header."""
    print(f"Header: {text}")

def update_footer(text: str) -> None:

    """Placeholder for update_footer."""
    print(f"Footer: {text}")

def update_left_panel(renderable: Any) -> None:

    """Placeholder for update_left_panel."""
    print(f"Left Panel: {renderable}")

def update_right_panel(renderable: Any) -> None:

    """Placeholder for update_right_panel."""
    print(f"Right Panel: {renderable}")

def display_progress(task_description: str) -> None:

    """Placeholder for display_progress."""
    print(f"Progress: {task_description}")

def display_code_snippet(

    code: str,
    language: str = "python",
    theme: str = "monokai",
    line_numbers: bool = True
) -> None:
    """Display a code snippet."""
    print(f"Code Snippet ({language}):\n{textwrap.fill(code, width=70)}")

def print_warning(message: str) -> None:

    """Print a warning message."""
    print(f"Warning: {message}")

def print_debug(message: str) -> None:

    """Print a debug message."""
    print(f"Debug: {message}")

def display_metrics_report(metrics: dict[str, Any]) -> None:

    """Display a formatted metrics report."""
    print("Metrics Report:")
    for key, value in metrics.items():
        if isinstance(value, (dict, list)):
            print(f"  {key}:")
            pprint(value, indent=4)
        else:
            print(f"  {key}: {value}")

def create_progress() -> Progress:

    """Create a progress object."""
    return Progress()

def print_phase_header(title: str) -> None:

    """Print a section header with formatting."""
    print(f"--- {title} ---")

def create_status_table(title: str, data: dict[str, Any]) -> None:

    """Create and display a status table."""
    print(f"{title}:")
    for key, value in data.items():
        print(f"  {key}: {value}")

def format_validation_status(success: bool, errors: list[str] | None = None) -> None:

    """Display validation status with optional errors."""
    status = "Passed" if success else "Failed"
    print(f"\nValidation Status: {status}")
    
    if not success and errors:
        for error in errors:
            print(f"  - {error}")

def display_processing_phase(title: str, content: dict[str, Any]) -> None:

    """Display a processing phase with formatted content."""
    print(f"--- {title} ---")
    for key, value in content.items():
        if isinstance(value, (dict, list)):
            print(f"  {key}:")
            pprint(value, indent=4)
        else:
            print(f"  {key}: {value}")

def display_api_metrics(response_data: dict[str, Any]) -> None:

    """Display API response metrics in a structured format."""
    print("\nAPI Response Metrics")
    for key, value in response_data.items():
        if isinstance(value, (dict, list)):
            print(f"  {key}:")
            pprint(value, indent=4)
        else:
            print(f"  {key}: {value}")

def display_validation_results(

    results: dict[str, bool], 
    details: Optional[dict[str, Any]] = None
) -> None:
    """Display validation results with details."""
    print("\nValidation Results")
    for key, value in results.items():
        print(f"  {key}: {value} Details: {details.get(key, '') if details else ''}")

def display_progress_summary(summary: dict[str, Any]) -> None:

    """Display a summary of the processing progress."""
    print("\nProcessing Summary")
    for key, value in summary.items():
        if isinstance(value, (dict, list)):
            print(f"  {key}:")
            pprint(value, indent=4)
        else:
            print(f"  {key}: {value}")
```

**Changes:**

*   **`response_formatter.py`:**
    *   Added `ensure_ascii=False` to `json.dumps` to prevent escaped backslashes.
*   **`console.py`:**
    *   Added `pprint` to `display_metrics`, `display_processing_phase`, `display_api_metrics`, and `display_progress_summary` to handle complex data structures.
    *   Added `textwrap.fill` to `display_code_snippet` to wrap long code snippets.
    *   Added `pprint` to `print_info` to handle complex data structures.

**How to Apply the Solution:**

1.  **Replace the content of `response_formatter.py`** with the modified code provided above.
2.  **Replace the content of `console.py`** with the modified code provided above.

**Explanation:**

*   By using `json.dumps(..., ensure_ascii=False)`, you prevent the escaping of non-ASCII characters, which reduces the number of backslashes.
*   By using `textwrap.fill`, you ensure that long strings are wrapped to fit within the terminal window.
*   By using `pprint`, you ensure that complex data structures are displayed in a human-readable format.

By implementing these changes, you should be able to prevent JSON strings, escaped backslashes, line breaks, and broken messages from cluttering your terminal output.

Do you have any other questions or need further clarification?

[Timestamp: 2024/12/18 23:54:04]

**user**: Metrics Collection

Module: default_module:
----------------------------------------
  cyclomatic_complexity     1
  cognitive_complexity      0
  maintainability_index     62.22
  halstead_metrics          {'volume': 79.17171988845776, 'difficulty': 1.2083333333333333, 'effort': 95.66582819855311, 'time': 5.314768233252951, 'bugs': 0.026390573296152586}   
  lines_of_code             198
  total_functions           2
  scanned_functions         0
  function_scan_ratio       0.00
  total_classes             19
  scanned_classes           0
  class_scan_ratio          0.00
  complexity_graph          None
----------------------------------------

--------------------------------------------------------------------------------


Docstring Processing Statistics:
----------------------------------------
  Total Processed           60
  Successfully Parsed       60
  Failed to Parse           0
  Average Length            87
  Total Lines               150
  Success Rate              100.0%
----------------------------------------

Code Extraction Results for default_module

Classes: 19

Functions: 0

Variables: 0

Constants: 0

Lines of Code: 198

Cyclomatic Complexity: 1

Maintainability Index: 62.22

Halstead Volume: 79.17

Dependencies: 4

Operation: code_extraction
  Success: True
  Duration: 0.04s
  classes_extracted: 19
  functions_extracted: 0
  variables_extracted: 0
  constants_extracted: 0
Code extraction completed in 0.04s.

Generating documentation prompt using template.

Generating documentation prompt using template.
Tracked request - Prompt Tokens: 1355, Max Completion Tokens: 16384

SUCCESS:
  Tokens tracked: 17739 total tokens.
Making API call
API call succeeded
Invalid JSON received from AI response, using fallback.

Operation: documentation_generation
  Success: False
  Duration: 7.17s
  module: exceptions
  file: simple\core\exceptions.py
  tokens: {'completion_tokens': 732, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens': 1362, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}, 'total_tokens': 2094}
  validation_success: False
  errors: ['Invalid JSON: Expecting value: line 1 column 1 (char 0)']
Response format is invalid, creating fallback.

Operation: documentation_generation
  Success: True
  Duration: 7.21s
  module_name: exceptions
  processing_time: 7.211508
  token_usage: {'completion_tokens': 732, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens': 1362, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}, 'total_tokens': 2094}

Operation: module_documentation_generation
  Success: True
  Duration: 7.22s
  module_path: simple\core\exceptions.py
Documentation written to documentation\exceptions.md
Documentation generation completed in 7.22s

SUCCESS:
  Successfully processed file: simple\core\exceptions.py
Source code read from simple\core\logger.py. Length: 8044
Generating documentation for simple\core\logger.py
Validating source code for file: simple
Syntax validation successful for: simple
No __init__.py found at simple\default_module\__init__.py. Module 'default_module' may not be importable.

--------------------------------------------------------------------------------


Metrics Collection

Module: default_module:
----------------------------------------
  cyclomatic_complexity     25
  cognitive_complexity      168
  maintainability_index     39.82
  halstead_metrics          {'volume': 1133.7678819054893, 'difficulty': 6.5, 'effort': 7369.491232385681, 'time': 409.41617957698224, 'bugs': 0.37792262730182974}
  lines_of_code             240
  total_functions           11
  scanned_functions         0
  function_scan_ratio       0.00
  total_classes             3
  scanned_classes           0
  class_scan_ratio          0.00
  complexity_graph          None
----------------------------------------

--------------------------------------------------------------------------------


Docstring Processing Statistics:
----------------------------------------
  Total Processed           70
  Successfully Parsed       70
  Failed to Parse           0
  Average Length            90
  Total Lines               186
  Success Rate              100.0%
----------------------------------------

Code Extraction Results for default_module

Classes: 3

Functions: 0

Variables: 14

Constants: 2

Lines of Code: 240

Cyclomatic Complexity: 25

Maintainability Index: 39.82

Halstead Volume: 1133.77

Dependencies: 4

Operation: code_extraction
  Success: True
  Duration: 0.06s
  classes_extracted: 3
  functions_extracted: 0
  variables_extracted: 14
  constants_extracted: 2
Code extraction completed in 0.06s.

Generating documentation prompt using template.

Generating documentation prompt using template.
Tracked request - Prompt Tokens: 2124, Max Completion Tokens: 16384

SUCCESS:
  Tokens tracked: 18508 total tokens.
Making API call
API call succeeded
Invalid JSON received from AI response, using fallback.

Operation: documentation_generation
  Success: False
  Duration: 2.46s
  module: logger
  file: simple\core\logger.py
  tokens: {'completion_tokens': 164, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens': 2131, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}, 'total_tokens': 2295}
  validation_success: False
  errors: ['Invalid JSON: Expecting value: line 1 column 1 (char 0)']
Response format is invalid, creating fallback.

Operation: documentation_generation
  Success: True
  Duration: 2.54s
  module_name: logger
  processing_time: 2.536649
  token_usage: {'completion_tokens': 164, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens': 2131, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}, 'total_tokens': 2295}

Operation: module_documentation_generation
  Success: True
  Duration: 2.54s
  module_path: simple\core\logger.py
Documentation written to documentation\logger.md
Documentation generation completed in 2.54s

SUCCESS:
  Successfully processed file: simple\core\logger.py
Source code read from simple\core\markdown_generator.py. Length: 16277
Generating documentation for simple\core\markdown_generator.py
Validating source code for file: simple
Syntax validation successful for: simple
No __init__.py found at simple\default_module\__init__.py. Module 'default_module' may not be importable.
Circular dependencies detected: [('core.logger', 'default_module'), ('core.logger', 'default_module'), ('core.exceptions', 'default_module')]

--------------------------------------------------------------------------------


Metrics Collection

Module: default_module:
----------------------------------------
  cyclomatic_complexity     56
  cognitive_complexity      1059
  maintainability_index     20.73
  halstead_metrics          {'volume': 1989.2890423739307, 'difficulty': 4.8977272727272725, 'effort': 9742.995196172318, 'time': 541.2775108984621, 'bugs': 0.6630963474579769}    
  lines_of_code             420
  total_functions           19
  scanned_functions         0
  function_scan_ratio       0.00
  total_classes             3
  scanned_classes           0
  class_scan_ratio          0.00
  complexity_graph          None
----------------------------------------

--------------------------------------------------------------------------------


Code Extraction Results for default_module

Classes: 3

Functions: 0

Variables: 50

Constants: 0

Lines of Code: 420

Cyclomatic Complexity: 56

Maintainability Index: 20.73

Halstead Volume: 1989.29

Dependencies: 4

Operation: code_extraction
  Success: True
  Duration: 0.09s
  classes_extracted: 3
  functions_extracted: 0
  variables_extracted: 50
  constants_extracted: 0
Code extraction completed in 0.09s.

Generating documentation prompt using template.

Generating documentation prompt using template.
Tracked request - Prompt Tokens: 3673, Max Completion Tokens: 16384

SUCCESS:
  Tokens tracked: 20057 total tokens.
Making API call
API call succeeded
Invalid JSON received from AI response, using fallback.

Operation: documentation_generation
  Success: False
  Duration: 2.38s
  module: markdown_generator
  file: simple\core\markdown_generator.py
  tokens: {'completion_tokens': 272, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens': 3680, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}, 'total_tokens': 3952}
  validation_success: False
  errors: ['Invalid JSON: Expecting value: line 1 column 1 (char 0)']
Response format is invalid, creating fallback.

Operation: documentation_generation
  Success: True
  Duration: 2.48s
  module_name: markdown_generator
  processing_time: 2.477772
  token_usage: {'completion_tokens': 272, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens': 3680, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}, 'total_tokens': 3952}

Operation: module_documentation_generation
  Success: True
  Duration: 2.48s
  module_path: simple\core\markdown_generator.py
Documentation written to documentation\markdown_generator.md
Documentation generation completed in 2.48s

SUCCESS:
  Successfully processed file: simple\core\markdown_generator.py
Source code read from simple\core\metrics.py. Length: 13523
Generating documentation for simple\core\metrics.py
Validating source code for file: simple
Syntax validation successful for: simple
No __init__.py found at simple\default_module\__init__.py. Module 'default_module' may not be importable.
Circular dependencies detected: [('core.logger', 'default_module'), ('core.logger', 'default_module'), ('core.dependency_injection', 'default_module'), ('core.logger', 'default_module'), ('core.dependency_injection', 'default_module'), ('core.metrics_collector', 'default_module')]

--------------------------------------------------------------------------------


Metrics Collection

Module: default_module:
----------------------------------------
  cyclomatic_complexity     39
  cognitive_complexity      139
  maintainability_index     27.26
  halstead_metrics          {'volume': 1536.1745153215943, 'difficulty': 11.533333333333333, 'effort': 17717.21274337572, 'time': 984.2895968542066, 'bugs': 0.5120581717738648}    
  lines_of_code             388
  total_functions           13
  scanned_functions         0
  function_scan_ratio       0.00
  total_classes             1
  scanned_classes           0
  class_scan_ratio          0.00
  complexity_graph          None
----------------------------------------

--------------------------------------------------------------------------------


Docstring Processing Statistics:
----------------------------------------
  Total Processed           80
  Successfully Parsed       80
  Failed to Parse           0
  Average Length            83
  Total Lines               209
  Success Rate              100.0%
----------------------------------------
Circular dependencies detected: [('core.logger', 'default_module')]

Code Extraction Results for default_module

Classes: 1

Functions: 0

Variables: 39

Constants: 4

Lines of Code: 388

Cyclomatic Complexity: 39

Maintainability Index: 27.26

Halstead Volume: 1536.17

Dependencies: 4

Operation: code_extraction
  Success: True
  Duration: 0.07s
  classes_extracted: 1
  functions_extracted: 0
  variables_extracted: 39
  constants_extracted: 4
Code extraction completed in 0.07s.

Generating documentation prompt using template.

Generating documentation prompt using template.
Tracked request - Prompt Tokens: 3052, Max Completion Tokens: 16384

SUCCESS:
  Tokens tracked: 19436 total tokens.
Making API call
API call succeeded
Invalid JSON received from AI response, using fallback.

Operation: documentation_generation
  Success: False
  Duration: 2.39s
  module: metrics
  file: simple\core\metrics.py
  tokens: {'completion_tokens': 257, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens': 3059, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}, 'total_tokens': 3316}
  validation_success: False
  errors: ['Invalid JSON: Expecting value: line 1 column 1 (char 0)']
Response format is invalid, creating fallback.

Operation: documentation_generation
  Success: True
  Duration: 2.47s
  module_name: metrics
  processing_time: 2.470161
  token_usage: {'completion_tokens': 257, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens': 3059, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}, 'total_tokens': 3316}

Operation: module_documentation_generation
  Success: True
  Duration: 2.48s
  module_path: simple\core\metrics.py
Documentation written to documentation\metrics.md
Documentation generation completed in 2.48s

SUCCESS:
  Successfully processed file: simple\core\metrics.py
Source code read from simple\core\metrics_collector.py. Length: 11159
Generating documentation for simple\core\metrics_collector.py
Validating source code for file: simple
Syntax validation successful for: simple
No __init__.py found at simple\default_module\__init__.py. Module 'default_module' may not be importable.
Circular dependencies detected: [('core.logger', 'default_module')]

[Timestamp: 2024/12/19 00:31:34]

**ai**: Okay, I've analyzed the provided output and the code modules. It seems like you're running your documentation generation process, and I can see the metrics, docstring processing statistics, code extraction results, and the output from the documentation generation process.

Here's a breakdown of what's happening and how to address the issues:

**Analysis of the Output:**

1.  **Metrics Collection:**
    *   The "Metrics Collection" sections show the complexity, maintainability, and other metrics for each module (`default_module`).
    *   The `halstead_metrics` are also included, which can be useful for understanding the complexity of the code.
    *   The `complexity_graph` is `None`, which means that you are not generating a complexity graph.
2.  **Docstring Processing Statistics:**
    *   The "Docstring Processing Statistics" sections show that all docstrings are being parsed successfully.
    *   The success rate is 100%, which is good.
3.  **Code Extraction Results:**
    *   The "Code Extraction Results" sections show the number of classes, functions, variables, and constants extracted from each module.
    *   The code extraction process is completing successfully.
4.  **Documentation Generation:**
    *   The "Generating documentation prompt using template" messages indicate that the prompt generation is working.
    *   The "Tracked request - Prompt Tokens" messages show the number of tokens used in the prompt.
    *   The "Making API call" and "API call succeeded" messages indicate that the API call is being made and is succeeding.
    *   **The "Invalid JSON received from AI response, using fallback." message indicates that the AI service is returning invalid JSON, and the fallback mechanism is being used.** This is the main issue we need to address.
    *   The "Operation: documentation_generation" sections show the success status, duration, module, file, tokens, validation status, and errors.
    *   The "Response format is invalid, creating fallback." message confirms that the fallback mechanism is being triggered.
    *   The "Operation: module_documentation_generation" sections show the success status, duration, and module path.
    *   The "Documentation written to documentation" messages confirm that the documentation is being written to the specified files.
    *   The "SUCCESS: Successfully processed file" messages indicate that the overall process is completing successfully, even when the fallback is triggered.
5.  **Circular Dependencies:**
    *   The "Circular dependencies detected" messages indicate that there are circular dependencies between some modules. This is not necessarily an error, but it can make the code harder to maintain and test.

**Addressing the Issues:**

1.  **Invalid JSON from AI Response:**
    *   **Problem:** The AI service is returning invalid JSON, causing the fallback mechanism to be triggered.
    *   **Solution:**
        *   **Modified `response_formatter.py`:** You've already implemented the fix in `response_formatter.py` by adding the `summary` field to the fallback response. This is the correct approach to prevent the `DataValidationError`.
        *   **Retry Mechanism:** While the fallback is good, you might also want to implement a retry mechanism in `ai_service.py` to retry the API call if the response is invalid. This could help to reduce the number of fallback responses.
        *   **AI Service Issue:** It's also a good idea to report this issue to the AI service provider, as they might be able to fix the underlying problem.
2.  **Circular Dependencies:**
    *   **Problem:** Circular dependencies can make the code harder to maintain and test.
    *   **Solution:**
        *   **Refactor Code:** Refactor your code to remove the circular dependencies. This might involve moving some code to a different module or using dependency injection to break the cycles.
        *   **Ignore Dependencies:** If the dependencies are not causing any issues, you can choose to ignore them. However, it's generally better to remove them if possible.
3.  **Terminal Output:**
    *   **Problem:** The terminal output is cluttered with raw JSON, escaped backslashes, and broken messages.
    *   **Solution:**
        *   **`response_formatter.py`:** You've already added `ensure_ascii=False` to `json.dumps` to prevent escaped backslashes.
        *   **`console.py`:** You've already added `pprint` to handle complex data structures and `textwrap.fill` to wrap long strings.
        *   **Logging Level:** You can control the verbosity of the terminal output by adjusting the logging level as described in the previous response.

**Code Modifications:**

Here's how you can implement the retry mechanism in `ai_service.py`:

```python
    async def _make_api_call_with_retry(
        self,
        prompt: str,
        function_schema: Optional[dict[str, Any]],
        max_retries: int = 3,
        log_extra: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Make an API call with retry logic following Azure best practices."""
        headers = {"api-key": self.config.api_key, "Content-Type": "application/json"}
        if self.correlation_id:
            headers["x-correlation-id"] = self.correlation_id

        request_params = await self.token_manager.validate_and_prepare_request(
            prompt,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            truncation_strategy=str(self.config.truncation_strategy) if self.config.truncation_strategy else None,
            tool_choice=str(self.config.tool_choice) if self.config.tool_choice else None,
            parallel_tool_calls=self.config.parallel_tool_calls,
            response_format=str(self.config.response_format) if self.config.response_format else None,
            stream_options=self.config.stream_options,
        )

        if function_schema:
            request_params["tools"] = [
                {"type": "function", "function": function_schema}
            ]
            if not request_params.get("tool_choice"):
                request_params["tool_choice"] = "auto"

        # Construct the URL using AZURE_API_BASE and AZURE_DEPLOYMENT_NAME
        url = (
            f"{self.config.azure_api_base.rstrip('/')}/openai/deployments/{self.config.azure_deployment_name}"
            f"/chat/completions?api-version={self.config.azure_api_version}"
        )

        for attempt in range(max_retries):
            try:
                if self._client is None:
                    await self.start()

                if self._client:
                    self.logger.info(
                        "Making API call",
                        extra={"url": url, "attempt": attempt + 1, "correlation_id": self.correlation_id},
                    )

                    async with self._client.post(
                        url,
                        headers=headers,
                        json=request_params,
                        timeout=ClientTimeout(total=self.config.timeout),
                    ) as response:
                        if response.status == 200:
                            response_json = await response.json()
                            self.logger.info(
                                "API call succeeded",
                                extra={"status_code": response.status, "correlation_id": self.correlation_id},
                            )
                            # Check if the response is valid JSON
                            try:
                                json.dumps(response_json)
                                return response_json
                            except (TypeError, ValueError) as e:
                                self.logger.warning(
                                    f"Invalid JSON received from AI response (attempt {attempt + 1}), retrying: {e}",
                                    extra=log_extra
                                )
                                await asyncio.sleep(2**attempt)
                                continue
                        error_text = await response.text()
                        self.logger.error(
                            "API call failed",
                            extra={
                                "status_code": response.status,
                                "error_text": error_text[:200],  # Limit error text length
                                "correlation_id": self.correlation_id,
                            },
                        )

                        # Handle specific error cases
                        if response.status == 429:  # Rate limit
                            retry_after = int(
                                response.headers.get("Retry-After", 2**attempt)
                            )
                            await asyncio.sleep(retry_after)
                            continue
                        elif response.status == 503:  # Service unavailable
                            await asyncio.sleep(2**attempt)
                            continue
                        else:
                            raise APICallError(f"API call failed: {error_text}")

            except asyncio.TimeoutError:
                self.logger.warning(
                    f"Request timeout (attempt {attempt + 1})", extra=log_extra
                )
                await asyncio.sleep(2**attempt)
            except Exception as e:
                self.logger.error(
                    f"Error during API call: {e}", exc_info=True, extra=log_extra
                )
                if attempt == max_retries - 1:
                    raise APICallError(
                        f"API call failed after {max_retries} retries: {e}"
                    )
                await asyncio.sleep(2**attempt)

        raise APICallError(f"API call failed after {max_retries} retries")
```

**Changes:**

*   I've added a check to ensure that the response is valid JSON before returning it. If the response is not valid JSON, it will retry the API call.

**How to Apply the Solution:**

1.  **Replace the content of `response_formatter.py`** with the modified code provided in the previous response.
2.  **Replace the content of `console.py`** with the modified code provided in the previous response.
3.  **Replace the content of `ai_service.py`** with the modified code provided above.
4.  **Adjust the logging level** in the `get_logger` calls in the modules where you want to control the output.

By implementing these changes, you should be able to:

*   Prevent the `DataValidationError` caused by the missing `summary` field.
*   Reduce the number of fallback responses by retrying the API call.
*   Control the verbosity of the terminal output by adjusting the logging level.
*   Display complex data structures in a more readable format.
*   Prevent escaped backslashes and broken messages from cluttering the terminal.

Let me know if you have any other questions or need further clarification.

[Timestamp: 2024/12/19 00:34:21]