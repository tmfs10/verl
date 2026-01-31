# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

## Project Overview

verl (Volcano Engine Reinforcement Learning) is a production-ready RL training library for large language models, developed by ByteDance Seed team. It implements the HybridFlow framework (EuroSys 2025) for efficient RLHF training.

**Key capabilities:**
- Training backends: FSDP, FSDP2, Megatron-LM
- Inference engines: vLLM, SGLang, HuggingFace Transformers
- RL algorithms: PPO, GRPO, GSPO, ReMax, REINFORCE++, RLOO, PRIME, DAPO, etc.
- Supports HuggingFace models: Qwen, Llama, Gemma, DeepSeek, etc.

## Common Commands

### Installation
```bash
pip install -e .                    # Basic install
pip install -e .[test,vllm]         # With test suite + vLLM backend
pip install -e .[test,sglang]       # With test suite + SGLang backend
pip install -e .[gpu,prime,math]    # GPU + PRIME + math verification extras
```

### Linting and Formatting
```bash
pre-commit install                  # Set up pre-commit hooks (required)
pre-commit run                      # Run on staged changes
pre-commit run --all-files          # Run on entire codebase
pre-commit run --all-files ruff     # Run specific hook
```

### Testing
```bash
pytest tests/                       # Run test suite
```
CI workflows run on GitHub Actions: GPU unit tests, CPU unit tests, vLLM tests, SGLang tests.

### Building Documentation
```bash
pip install -r requirements-docs.txt
cd docs && make clean && make html
python -m http.server -d _build/html/
```

## Architecture

### Distributed Master-Worker Pattern

The codebase uses a Ray-based distributed architecture with specialized workers:

1. **Trainer (Controller)** - Entry points in `verl/trainer/main_*.py`
   - `main_ppo.py`: PPO training entry point
   - `main_generation.py`: Generation/inference entry point
   - `ppo/ray_trainer.py`: PPO trainer implementation
   - `main_grad_projection.py`: Rademacher projection of gradients entry point
   - `ppo/ray_grad_projection_trainer.py`: Rademacher projection of gradients implementation
   - Uses Hydra for configuration management

2. **Workers** - Located in `verl/workers/`
   - `actor/`: Policy model training
   - `critic/`: Value function training
   - `rollout/`: Sample generation from policy
   - `reward_model/`: Reward computation
   - `reward_manager/`: Reward orchestration
   - `fsdp_workers.py`: FSDP-based distributed training
   - `megatron_workers.py`: Megatron-LM backend

3. **Data Protocol** - `verl/protocol.py`
   - `DataProto`: Standardized data exchange format between workers
   - Built on TensorDict for efficient batch handling

### Configuration System

Uses Hydra with YAML configs in `verl/trainer/config/`:
- `ppo_trainer.yaml`: Main PPO configuration
- `ppo_megatron_trainer.yaml`: Megatron variant
- Subdirectories: `actor/`, `critic/`, `reward_model/`, `data/`, `optim/`, `engine/`
- Auto-generated configs: `_generated_*.yaml` (created by pre-commit hook)

### Key Directories

- `verl/trainer/ppo/`: PPO algorithm implementation
- `verl/models/`: Model integrations (transformers patches, Llama, Qwen2, Megatron core)
- `verl/utils/`: Dataset utilities, checkpointing, reward scoring, profiling
- `examples/`: Training script examples for different algorithms
- `recipe/`: Reproducible algorithm implementations (DAPO, PRIME, SPPO, etc.)

## Code Style

- **Line length**: 120 characters
- **Linting**: Ruff (pycodestyle, Pyflakes, pyupgrade, flake8-bugbear, isort)
- **Type checking**: MyPy (enabled on specific modules in `verl/trainer/config/`, `verl/workers/reward_manager/`)
- **License**: Apache 2.0 headers required on all source files
- **Pre-commit hooks**: Ruff, MyPy, trainer config auto-generation, docstring coverage, license checking

## Installation Extras

| Extra | Description |
|-------|-------------|
| `test` | pytest, pre-commit, py-spy |
| `vllm` | vLLM inference engine (>=0.7.3) |
| `sglang` | SGLang inference engine |
| `gpu` | liger-kernel, flash-attn |
| `prime` | PRIME algorithm dependencies |
| `math` | math-verify for math tasks |
| `mcore` | Megatron-LM via mbridge |
