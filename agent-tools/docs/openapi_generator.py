
# ---- 文档生成 ----
# docs/openapi_generator.py
from core.registry import ToolRegistry

class OpenAPIGenerator:
    def generate(self):
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Agent Tools API", "version": "1.0.0"},
            "paths": {}
        }
        
        registry = ToolRegistry()
        for name, data in registry.list_tools().items():
            path = {
                "post": {
                    "summary": data['description'],
                    "parameters": [
                        {"name": param, "schema": {"type": str(typ)}}
                        for param, typ in data['parameters'].items()
                        if param != 'return'
                    ],
                    "responses": {
                        "200": {
                            "description": "Success",
                            "content": {"application/json": {"schema": {"type": str(data['return_type'])}}}
                        }
                    }
                }
            }
            spec['paths'][f"/tools/{name}"] = path
        return spec

