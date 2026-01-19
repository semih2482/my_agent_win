import unittest
from unittest.mock import MagicMock, patch, ANY
import os
import sys

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent.core.agent import Agent, Colors

class TestAgentLoopFix(unittest.TestCase):

    def setUp(self):
        """Set up a mock environment for the agent."""
        self.mock_tools = {
            "tool_creator": {
                "func": MagicMock(return_value={"status": "success", "result": "Yeni araç 'test_tool.py' başarıyla oluşturuldu ve ... 'review_and_approve_tool' aracını kullanarak onaylayın."}),
                "description": "Creates a new tool."
            },
            "review_and_approve_tool": {
                "func": MagicMock(return_value={"status": "success", "result": "Tool approved."}),
                "description": "Reviews and approves a tool."
            },
            "none": {
                "func": None,
                "description": "No action."
            }
        }
        
        # Mock external dependencies
        self.patcher_config = patch('agent.core.agent.config')
        self.patcher_ask = patch('agent.core.agent.ask')
        self.patcher_embed = patch('agent.core.agent.embed')
        self.patcher_detect_intent = patch('agent.core.agent.detect_intent')
        self.patcher_select = patch('agent.core.agent.select')
        self.patcher_tty = patch('agent.core.agent.tty')
        self.patcher_termios = patch('agent.core.agent.termios')

        self.mock_config = self.patcher_config.start()
        self.mock_ask = self.patcher_ask.start()
        self.mock_embed = self.patcher_embed.start()
        self.mock_detect_intent = self.patcher_detect_intent.start()
        self.mock_select = self.patcher_select.start()
        self.mock_tty = self.patcher_tty.start()
        self.mock_termios = self.patcher_termios.start()

        # Configure mocks
        self.mock_config.MEMORY_DB_PATH = ':memory:'
        self.mock_config.PERSONA_DB_PATH = ':memory:'
        self.mock_config.PERSONAL_STORE_PATH = 'tests/mock_personal_store'
        self.mock_config.KG_DB_PATH = ':memory:'
        self.mock_detect_intent.return_value = {"strategy": "reactive"}
        self.mock_embed.return_value = [0.1] * 768 # Dummy embedding

        # Ensure mock store path exists
        if not os.path.exists(self.mock_config.PERSONAL_STORE_PATH):
            os.makedirs(self.mock_config.PERSONAL_STORE_PATH)

        # Instantiate the agent
        self.agent = Agent(available_tools=self.mock_tools, non_cacheable_tools=[], reload_tools_func=None)

    def tearDown(self):
        """Clean up patches."""
        self.patcher_config.stop()
        self.patcher_ask.stop()
        self.patcher_embed.stop()
        self.patcher_detect_intent.stop()
        self.patcher_select.stop()
        self.patcher_tty.stop()
        self.patcher_termios.stop()

    def test_agent_breaks_loop_after_tool_creation(self):
        """
        Verify that the agent calls 'review_and_approve_tool' immediately after 'tool_creator' succeeds.
        """
        # --- Step 1: Simulate the initial tool creation ---
        
        # Mock the LLM to decide to create a tool first
        self.mock_ask.return_value = '''```json
{
  "thought": "This is a coding task, I should create a tool.",
  "action": "tool_creator",
  "input": "Create a test tool.",
  "response": null
}
```'''
        
        # Run the agent for the first step
        self.agent.run("Create a test tool.")
        
        # Assert that tool_creator was called
        self.mock_tools["tool_creator"]["func"].assert_called_once_with("Create a test tool.")
        
        # --- Step 2: Verify the agent's next action is to approve the tool ---
        
        # The agent's internal state (`last_observation`) should now contain the approval message.
        # The hard-coded rule should now trigger.
        
        # We can check if the `review_and_approve_tool` was called.
        # Since the `run` method is complex, we'll check the mock calls.
        # The `run` method completes in one go in this test setup, so we need to trace the calls.
        
        # The second tool call inside the `run` method should be `review_and_approve_tool`.
        # Let's check the call list of the mock functions.
        
        # The first call was to tool_creator.
        self.mock_tools["tool_creator"]["func"].assert_called_once()
        
        # The second call must be to review_and_approve_tool
        self.mock_tools["review_and_approve_tool"]["func"].assert_called_once()
        
        # We can also be more specific about the input to the approval tool.
        self.mock_tools["review_and_approve_tool"]["func"].assert_called_with("approve test_tool.py")

        # Crucially, the LLM should NOT have been called a second time to make a decision.
        # The first call decides to use tool_creator. The second step (approval) should be hard-coded.
        self.assertEqual(self.mock_ask.call_count, 1, "The LLM should only be called once for the initial decision.")

if __name__ == '__main__':
    unittest.main()
