# RLHF 

# LLM_SFT_PPO_DPO

Exploring the alignment pipeline from **SFT** to **RLHF-style optimisation**, using **PPO** and **DPO** on an open-source LLM.

## Overview

This repository contains an end-to-end coursework project for building and evaluating an instruction-following and preference-aligned language model pipeline.

The project follows this sequence:

1. **SFT**: Supervised fine-tuning on instruction data
2. **Reward Modeling**: Train a reward model on preference data
3. **PPO**: RLHF training with the reward model
4. **DPO**: Direct preference optimisation
5. **Evaluation**: Generate outputs, build blind pairwise comparisons, score results, and plot figures

### Base components

- **Base model**: `Qwen/Qwen2.5-1.5B`
- **SFT dataset**: `nvidia/Daring-Anteater`
- **Preference dataset**: `Anthropic/hh-rlhf`


## What is included and what is not

This GitHub repository contains:

- training / evaluation scripts
- plotting notebook and plotting scripts
- pairwise evaluation results and plots
- small processed samples and result summaries

This repository does **not** include:

- raw datasets
- Hugging Face cache
- base model weights
- large training checkpoints
- full processed training corpora

Those large artifacts should be downloaded or generated locally.

---

## 1. Environment configuration

### 1.1 Create a clean environment

It is recommended to create a dedicated Conda environment first.

```bash
conda create -n rlhf3090 python=3.10 -y
conda activate rlhf3090
```

### 1.2 Install PyTorch

Install PyTorch first using the official PyTorch selector for your machine.

For Linux + CUDA 12.8, one example is:

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 1.3 Install the remaining requirements

After PyTorch is installed:

```bash
python -m pip install -r requirements.txt
```

### 1.4 Environment variables

Load the project environment variables:

```bash
source scripts/set_env.sh
```

This script is used to set:
- `PROJECT_ROOT`
- Hugging Face cache paths
- temporary directories
- other project-level environment variables

### 1.5 Verify the environment

A quick check:

```bash
which python
python --version
python -c "import torch; print(torch.cuda.is_available())"
python -c "import transformers, datasets, trl, peft, accelerate; print('imports OK')"
```

---

## 2. Data processing pipeline

### 2.1 Download the base model

```bash
python scripts/00_download_model.py
```

This downloads the base model snapshot into the local project directory.

### 2.2 Download the datasets

```bash
python scripts/01_download_datasets.py
```

This downloads:
- `nvidia/Daring-Anteater`
- `Anthropic/hh-rlhf`

### 2.3 Verify model and datasets

```bash
python scripts/02_verify_assets.py
```

This checks:
- tokenizer loading
- dataset splits
- dataset accessibility

### 2.4 Prepare the SFT dataset

```bash
python scripts/03_prepare_sft_data.py
```

This converts the raw Daring-Anteater data into a clean conversational format for SFT training.

Outputs include:
- processed SFT dataset
- sample rows for inspection

### 2.5 Prepare the HH preference dataset

```bash
python scripts/04_prepare_hh_preferences.py
```

This converts HH-RLHF into explicit preference format:

- `prompt`
- `chosen`
- `rejected`

It also keeps conversational variants for later use in DPO and evaluation.

---

## 3. Purpose of each script in `scripts/`

### Setup and preprocessing

#### `00_download_model.py`
Downloads the base model into the local project directory.

#### `01_download_datasets.py`
Downloads the raw SFT and preference datasets.

#### `02_verify_assets.py`
Checks whether the downloaded model and datasets can be loaded correctly.

#### `03_prepare_sft_data.py`
Processes the SFT dataset into the format used by SFT training.

#### `04_prepare_hh_preferences.py`
Processes HH-RLHF into explicit preference format for reward modeling, PPO, DPO, and evaluation.

### Training

#### `05_train_sft.py`
Trains the **SFT baseline** using:
- Qwen2.5-1.5B
- 4-bit loading
- LoRA
- the processed Daring-Anteater dataset

#### `06_train_reward_hh.py`
Trains the **reward model** using:
- the processed HH preference dataset
- a sequence-classification objective
- LoRA + 4-bit setup

#### `07_train_ppo_hh.py`
Runs **PPO-based RLHF** using:
- the SFT model as the policy and reference starting point
- the trained reward model
- prompt-only rollouts derived from HH prompts

#### `08_train_dpo_hh.py`
Runs **DPO** using:
- the SFT checkpoint as the starting policy
- HH preference pairs in `prompt / chosen / rejected` format

### Evaluation

#### `09_generate_eval_outputs.py`
Generates outputs from:
- SFT
- PPO-HH
- DPO-HH

on the same held-out evaluation prompt set.

#### `10_pairwise_compare.py`
Builds blind pairwise comparison files:
- SFT vs PPO
- SFT vs DPO
- PPO vs DPO

It creates:
- a blind file for human judging
- a private answer key

#### `11_score_pairwise_results.py`
Scores completed blind evaluations and computes:
- wins
- losses
- ties
- invalid counts
- summary JSON

#### `12_plot_pairwise_results.py`
Plots pairwise evaluation results and summary figures.

### Utility

#### `set_env.sh`
Shell script for setting project-level environment variables.

#### `talk2.py`
Interactive terminal chat tool for talking to a selected model checkpoint.

Examples:

```bash
python scripts/talk2.py --model SFT
python scripts/talk2.py --model PPO
python scripts/talk2.py --model DPO
python scripts/talk2.py --model ls
python scripts/talk2.py --config
```

---

## 4. How to train and run the models

### 4.1 Train SFT

```bash
python scripts/05_train_sft.py
```

Main output:
- SFT final adapter checkpoint

### 4.2 Train reward model

```bash
python scripts/06_train_reward_hh.py
```

Main output:
- reward model checkpoint
- training and evaluation metrics

### 4.3 Train PPO-HH

```bash
python scripts/07_train_ppo_hh.py
```

Typical optional controls:

```bash
export PPO_TOTAL_EPISODES=10000
python scripts/07_train_ppo_hh.py
```

Main output:
- PPO final adapter
- train metrics
- rollout and PPO log history

### 4.4 Train DPO-HH

```bash
python scripts/08_train_dpo_hh.py
```

Main output:
- DPO final adapter
- train metrics
- eval metrics
- log history

---

## 5. How to run evaluation

### 5.1 Generate model outputs on held-out prompts

```bash
export EVAL_PROMPT_LIMIT=50
export EVAL_SPLIT=test
python scripts/09_generate_eval_outputs.py
```

This creates:
- evaluation prompts
- SFT outputs
- PPO outputs
- DPO outputs

### 5.2 Build blind pairwise comparisons

```bash
export PAIRWISE_LIMIT_PER_COMPARISON=50
python scripts/10_pairwise_compare.py
```

This creates:
- `blind_eval.*`
- `answer_key.*`

### 5.3 Score completed pairwise evaluations

After filling the `winner` field in the blind file:

```bash
BLIND_EVAL_PATH=outputs/pairwise_compare/blind_eval_filled.jsonl \
ANSWER_KEY_PATH=outputs/pairwise_compare/answer_key.jsonl \
python scripts/11_score_pairwise_results.py
```

### 5.4 Plot results

```bash
python scripts/12_plot_pairwise_results.py
```

Outputs include:
- pairwise win-rate figure
- Wilson confidence interval plot
- result tables

---

## 6. Interactive chatting with trained models

Use `talk2.py` to talk to the trained adapters directly in the terminal.

### 6.1 List available models

```bash
python scripts/talk2.py --model ls
```

### 6.2 Chat with a model

```bash
python scripts/talk2.py --model SFT
python scripts/talk2.py --model PPO
python scripts/talk2.py --model DPO
```

### 6.3 Configure chat decoding parameters

```bash
python scripts/talk2.py --config
```

This allows you to set:
- `temperature`
- `top_p`
- `max_new_tokens`
- `system_prompt`

Press `Ctrl + C` to exit the chat.

---

## 7. Recommended run order

For full reproduction, run the scripts in this order:

```text
00_download_model.py
01_download_datasets.py
02_verify_assets.py
03_prepare_sft_data.py
04_prepare_hh_preferences.py
05_train_sft.py
06_train_reward_hh.py
07_train_ppo_hh.py
08_train_dpo_hh.py
09_generate_eval_outputs.py
10_pairwise_compare.py
11_score_pairwise_results.py
12_plot_pairwise_results.py
```

---

## 8. Notes and limitations

- Raw data and large checkpoints are intentionally not committed to GitHub.
- PPO and DPO in this project may be run under different compute budgets; comparisons should be described honestly.
- Blind pairwise evaluation is currently the main preference-based evaluation method in this repository.
- Results from very small prompt sets should be treated as pilot results, not final claims.

In other words, if the preference comparison is denoted by

$$
\text{SFT} \;\text{vs}\; \text{PPO}, \qquad
\text{SFT} \;\text{vs}\; \text{DPO}, \qquad
\text{PPO} \;\text{vs}\; \text{DPO},
$$

then the resulting win rates should be interpreted as **evaluation outcomes under the current experimental configuration**, not as universal conclusions about the methods.

---

## 9. Citation / acknowledgment

This repository uses:

- Hugging Face Transformers
- Datasets
- PEFT
- TRL
- PyTorch
- Rich
- Matplotlib
- Seaborn

Please consult the original package documentation for full API details.

If you use this repository structure or scripts in your own coursework or experiments, please also acknowledge the original datasets and model sources:

- `Qwen/Qwen2.5-1.5B`
- `nvidia/Daring-Anteater`
- `Anthropic/hh-rlhf`
