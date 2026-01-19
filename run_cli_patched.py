
import sys
import os
import builtins
import multiprocessing

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Set TOKENIZERS_PARALLELISM to false to avoid deadlocks
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Keep the original input function
original_input = builtins.input

def patched_input(prompt=""):
    # Print the prompt that the user would have seen
    print(prompt, end='')
    # If the prompt is for the proactive assistant, return 'e'
    if "Bu konuda derinlemesine bir araştırma yapıp öğrenmemi ister misiniz? (e/h):" in prompt:
        print('e') # Simulate user typing 'e'
        return 'e'
    # For all other inputs, exit the loop by simulating 'q'
    print('q')
    return 'q'

# Apply the patch
builtins.input = patched_input

from agent.ui import cli

if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn' to avoid CUDA/fork issues.
    # This must be done in the main block before any multiprocessing-related
    # code is executed.
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # The context can only be set once.
        pass

    print("--- Running CLI with patched input ---")
    try:
        cli.main()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Restore the original input function
        builtins.input = original_input
        print("--- CLI execution finished ---")
