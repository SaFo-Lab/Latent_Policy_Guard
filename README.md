# Latent Policy Guard (LPG)

**Latent Policy Guard** is a guardrail model that performs *semantic latent deliberation* over dynamic safety policies. Given a content snippet and an indexed policy list, LPG reasons internally over the user's **intent** and the **risk** of policy violation as continuous latent states, and emits only a compact verdict anchored to violated policy indices — preserving auditability while avoiding the latency cost of explicit chain-of-thought guardrails.


## Installation

```bash
pip install -r requirements.txt
```

## Model

The trained **LPG-4B** checkpoint is released on the Hugging Face Hub at
[`andyc03/LPG_4B`](https://huggingface.co/andyc03/LPG_4B). It is the evaluated
checkpoint (a Qwen3-4B base with LoRA and the latent projection bundled into a
single `model.safetensors`). The base model (`Qwen/Qwen3-4B`) is loaded
separately for the architecture and tokenizer, and the released weights are
applied on top.

```bash
pip install -U "huggingface_hub[cli]"
hf download andyc03/LPG_4B --local-dir checkpoints/LPG_4B
```

## Training

The full training set (**40k records**) is hosted on the Hugging Face Hub at [`andyc03/latent-policy-guard-40k`](https://huggingface.co/datasets/andyc03/latent-policy-guard-40k). It is assembled from DynaBench, GuardSet-X, BeaverTails, Aegis v2, SaladBench, Toxic-Chat and XSTest v2, with per-example randomised policy lists and teacher-grounded LLM reasoning. See the [dataset card](https://huggingface.co/datasets/andyc03/latent-policy-guard-40k) for the full per-source breakdown and source-dataset attribution.

Download it into `training/data/` before training:

```bash
cd training
pip install -U "huggingface_hub[cli]"
hf download andyc03/latent-policy-guard-40k train_data_lpg_40k.jsonl \
    --repo-type dataset --local-dir data
```

Each line is one example with the schema:

```json
{
  "annotation_input":    "**Safety policies (indexed from 0):**\n0: ...\n1: ...\n**Content to evaluate:**\n<content>",
  "generated_reasoning": "<Intent>...</Intent>\n\n<Risk>...</Risk>\n\n<Output>unsafe, policy 0</Output>",
  "teacher_summaries":   {"intent_summary": "<IntentSummary>...</IntentSummary>", "risk_summary": "<RiskSummary>...</RiskSummary>"}
}
```

To train:

```bash
cd training
MODEL_PATH=Qwen/Qwen3-4B \
DATA_PATH=data/train_data_lpg_40k.jsonl \
NUM_GPUS=4 \
bash scripts/train.sh
```

`train.sh` is a thin wrapper around `python train.py …`. Any dataclass field declared in `src/model.py` (`ModelArguments`, `TrainingArguments`, `DataArguments`) can be overridden by appending its `--flag` to the command line.

## Evaluation

Models and datasets are plugged together through lightweight registries; adding a new system means dropping in one file under `evaluation/models/` or `evaluation/datasets/`.

```bash
cd evaluation
python evaluate.py \
    --model latent_policy_guard \
    --model_path Qwen/Qwen3-4B \
    --ckpt_dir   /path/to/checkpoint \
    --dataset    wildguard \
    --dataset_path allenai/wildguardmix \
    --output     results/lpg_wildguard.json \
    --no_system_prompt
```

Use `python evaluate.py --list_models` and `--list_datasets` to enumerate everything that is registered.

### Evaluate on DynaBench(Aug)

The augmented DynaBench test set from the paper (Section 5.1) is bundled at
[`evaluation/dataset/dynabench_latest.json`](evaluation/dataset/dynabench_latest.json)
(543 examples, with shuffled policy lists and counterfactual variants). With the
released `andyc03/LPG_4B` checkpoint:

```bash
cd evaluation
python evaluate.py \
    --model latent_policy_guard \
    --model_path Qwen/Qwen3-4B \
    --ckpt_dir   /path/to/LPG_4B \
    --dataset    dynabench \
    --dataset_path dataset/dynabench_latest.json \
    --output     results/lpg_dynabench.json \
    --no_system_prompt \
    --num_latent_per_stage "4,6" --stage_names "intent,risk" \
    --lora_r 128 --lora_alpha 32 \
    --use_prj True --prj_dim 2560 \
    --greedy True --remove_eos True \
    --model_max_length 1024 --max_new_tokens 160
```

## License

Released under the MIT License — see [`LICENSE`](LICENSE).

## Citation

```bibtex
@misc{li2026lpgbalancingefficiencypolicy,
      title={LPG: Balancing Efficiency and Policy Reasoning in Latent Policy Guardrails}, 
      author={Nanxi Li and Zhengyue Zhao and Chaowei Xiao},
      year={2026},
      eprint={2605.17329},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2605.17329}, 
}
```
