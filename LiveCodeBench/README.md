# LiveCodeBench

Official repository for the paper "LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code"

<p align="center">
    <a href="https://livecodebench.github.io/">🏠 Home Page</a> •
    <a href="https://huggingface.co/livecodebench/">💻 Data </a> •
    <a href="https://livecodebench.github.io/leaderboard.html">🏆 Leaderboard</a> •
    <a href="https://huggingface.co/spaces/livecodebench/code_generation_samples">🔍 Explorer</a>
</p>

## About LiveCodeBench

LiveCodeBench provides **holistic and contamination-free evaluation** of coding capabilities of large language models (LLMs). 

The benchmark continuously collects new problems from programming contests across three platforms:
- **LeetCode**
- **AtCoder** 
- **CodeForces**

LiveCodeBench evaluates models across multiple code-related capabilities:
- **Code Generation** - Generate solutions to programming problems
- **Self-Repair** - Fix errors in generated code
- **Code Execution** - Predict code execution outputs
- **Test Output Prediction** - Predict test case results

The dataset currently contains **1055+ high-quality problems** published between May 2023 and April 2025, with new problems added regularly to prevent contamination.

## Quick Start

**For easy evaluation of locally fine-tuned models**, see the [eval_simple/](./eval_simple/) directory for a streamlined setup with simple CLI and SLURM scripts.

For detailed information about the full benchmark including API model support, custom evaluations, and advanced features, please refer to our [website](https://livecodebench.github.io).

## Citation

```bibtex
@article{jain2024livecodebench,
  author    = {Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, Ion Stoica},
  title     = {LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code},
  year      = {2024},
  journal   = {arXiv preprint},
}
```
