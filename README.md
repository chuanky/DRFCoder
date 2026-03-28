# Dynamic Reasoning Fusion (DRF)

This repository contains the official implementation of **Dynamic Reasoning Fusion (DRF)**, a unified framework that adaptively coordinates diverse reasoning strategies for inference-time scaling on competition-level programming tasks.

## 🚀 Overview

Solving complex algorithmic problems remains a challenge for Large Language Models (LLMs). While inference-time scaling is promising, existing methods often rely on static strategy allocation. 

**DRF** addresses this by organizing the inference process as a **Behavior Tree (BT)**. This modular architecture enables:
- **Adaptive Strategy Prioritization:** Dynamically selecting the most suitable reasoning path (e.g., CoT, Plan-to-Code, Analogy) based on task difficulty.
- **Reactive Fallback:** Systematically recovering from errors through closed-loop refinement and strategy switching.
- **Efficient Resource Allocation:** Outperforming strong baselines while consuming comparable or significantly fewer tokens.

Experimental results on **LiveCodeBench** show that DRF achieves Pass@1 improvements of **6.11%, 8.9%, and 5.6%** over the strongest baselines for Qwen3-4B, Qwen3.5-9B, and DeepSeek-V3, respectively.

---

## 🛠️ Installation

This project uses `uv` for dependency management.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh # install uv
uv sync  # install dependencies for this repo
```
---

## 💻 Usage

### 1. Setup Inference Server
DRF supports both local inference via `vLLM` and API-based providers. To start a local server (e.g., for Qwen3.5-9B):

```bash
bash run_vllm.sh # Edit run_vllm.sh with your model path
```

### 2. Run DRF
You can run the framework in different modes by modifying `src/main.py`:

```bash
uv run src/main.py
```

- **Run Mode**: Executes the full DRF pipeline on the benchmark.
- **Debug Mode**: Visualizes the Behavior Tree execution for a single problem.
- **Ablation Mode**: Runs experiments with specific strategy subsets.

### 3. Evaluation
The framework integrates with `LiveCodeBench` (included in `lcb_runner/`). Results are saved to the `outputs/` directory.

---

## 🏗️ Project Structure

```text
├── lcb_runner/         # Evaluation infrastructure (LiveCodeBench fork)
├── src/                # Core DRF implementation
│   ├── main.py         # Entry point and experiment management
│   ├── level1_actions.py  # Leaf nodes (Generation, Reflection, Testing)
│   ├── level2_actions.py  # high level nodes (Fusion nodes, strategy routing)
│   ├── client.py       # LLM client abstractions (vLLM, OpenAI API)
│   ├── lcb_env.py      # LiveCodeBench environment wrapper
│   └── prompt_utils.py # Modular prompt templates
├── datasets/           # Pre-processed benchmark data
└── outputs/            # Execution traces and evaluation results
```

---

## 📊 Additional Resources

- [py_tree](https://py-trees.readthedocs.io/en/devel/index.html): Official documentation for py_tree library. 
- [Introduction to Behavior Trees](https://roboticseabass.com/2021/05/08/introduction-to-behavior-trees/): a gentle and practical introduction for Behavior Tree (2021).