# Autonomous AI Agent

This project is an advanced, self-directed artificial intelligence agent capable of autonomously planning complex tasks, utilizing tools, and maintaining long-term memory. Built upon modern "Agentic AI" principles, this system develops dynamic strategies to achieve user goals.

## About the Project

This project aims to go beyond standard chatbots by creating a digital assistant that **"thinks and acts"**. It doesn't just respond; it conducts internet research, writes and executes Python code, performs financial analysis, and stores insights from these processes in its long-term memory (Vector Store & Knowledge Graph).

**Core Philosophy:**
*   **Proactive, not Reactive:** Instead of waiting passively for user input, it determines the necessary steps to complete a task.
*   **Continuous Learning:** It retains information and user preferences gathered from interactions.
*   **Transparent Decision Mechanism:** It explains *why* it chose a specific tool or provided a specific answer using a "chain-of-thought" process.

## Key Features

### Advanced Cognitive Architecture
*   **Hybrid Strategy:** Uses `Reactive Mode` for simple conversations and `Planner Mode` for complex tasks.
*   **Self-Reflection:** The agent critiques its own outputs and corrects itself when errors occur.
*   **Intent Detection:** Analyzes what the user wants (Chat? Transaction? Research?).

### Dual-Layer Memory System
1.  **Vector Memory (Semantic Memory):** Uses `sentence-transformers` to convert text into vectors for semantic search, enabling recall of past conversations and learned information.
2.  **Knowledge Graph:** Structurally stores relationships between concepts (e.g., "Alice" -> "works_at" -> "Google") and performs logical deductions.

### Extensive Toolset
The agent utilizes various tools to expand its capabilities:
*   **Internet Access:** Searches for up-to-date information and reads web pages using `DuckDuckGo Search`.
*   **Code Capabilities:** Writes, analyzes (`AST` analysis), and edits Python files.
*   **Financial Analysis:** Analyzes stock, fund, and crypto data; interprets charts (technical analysis).
*   **File Operations:** Reads/writes files and executes system commands.

## Architecture and Technologies

The project is developed in **Python** with a modular and extensible structure.

*   **LLM Integration:** Works with local LLMs via `llama-cpp-python` (CPU/GPU compatible) and supports OpenAI-compatible APIs.
*   **Embeddings:** High-quality vector representations using `sentence-transformers` (HuggingFace models).
*   **Data Structures:** Custom vector store and JSON-based data storage.
*   **Parallel Processing:** Executes multi-agent/tool tasks simultaneously using `concurrent.futures`.

## Project Structure


my_agent_win/
├── agent/
│   ├── core/           # Main agent loop (Brain), decision mechanisms
│   ├── memory/         # Vector store and Knowledge Graph
│   ├── planner/        # Task planning and strategy modules
│   ├── tools/          # All tools available to the agent
│   ├── models/         # LLM abstraction layer
│   └── ui/             # Command Line Interface (CLI) and entry point
├── data/               # Database and persistent memory files
├── models/             # Local LLM model files
└── run_cli_patched.py  # Entry script to start the agent


## Installation and Usage

### Requirements
*   Python 3.10 or higher
*   Git
*   (Optional) NVIDIA GPU and CUDA Toolkit (For faster local model performance)

### Step-by-Step Installation

1.  **Clone the Project:**

    git clone https://github.com/username/my-agent.git
    cd my-agent


2.  **Create a Virtual Environment:**

    python -m venv .venv
    # For Windows:
    .venv\Scripts\activate
    # For Linux/Mac:
    source .venv/bin/activate
 

3.  **Install Dependencies:**

    pip install -r requirements.txt

    *(Note: `llama-cpp-python` installation may vary based on your hardware (CPU/GPU). Please refer to the relevant documentation for GPU support.)*

### Usage

To start the agent, run the following command in the terminal:

python -m agent.ui.cli


**Example Commands:**
*   `"Research the price of Bitcoin and analyze the changes over the last week."`
*   `"Read the 'test.txt' file on my desktop and summarize its content."`
*   `"Code a snake game in Python and save it as 'snake.py'."`

## License

This project is licensed under the MIT License - see the LICENSE file for details.
