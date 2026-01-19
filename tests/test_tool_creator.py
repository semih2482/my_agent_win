import os
import json
import unittest
from unittest.mock import patch, MagicMock

from agent.tools.tool_creator import run as tool_creator_run

class TestToolCreator(unittest.TestCase):

    def setUp(self):
        self.agent_instance = MagicMock()
        self.agent_instance.available_tools = {}
        self.tools_dir = os.path.join(os.path.dirname(__file__), '..', 'agent', 'tools', 'community_tools')
        os.makedirs(self.tools_dir, exist_ok=True)

    def tearDown(self):
        # Clean up created files
        for item in os.listdir(self.tools_dir):
            if item.endswith(".py"):
                os.remove(os.path.join(self.tools_dir, item))

    @patch('agent.tools.tool_creator.ask')
    def test_create_simple_tool(self, mock_ask):
        # Mock LLM response
        mock_ask.return_value = """
```python
import json
from typing import Dict, Any, Union

TOOL_INFO = {
    "name": "hello_world",
    "description": "A simple tool that prints hello world.",
    "input_schema": {}
}

def run(args: Union[Dict, str], agent_instance=None) -> Dict[str, Any]:
    return {"status": "success", "result": "Hello, World!"}
```
"""

        args = {
            "task_description": "A simple tool that prints hello world.",
            "tool_name": "hello_world",
            "input_schema": {}
        }

        result = tool_creator_run(args, self.agent_instance)

        self.assertEqual(result['status'], 'success')
        self.assertTrue(os.path.exists(os.path.join(self.tools_dir, 'hello_world.py')))

if __name__ == '__main__':
    unittest.main()
