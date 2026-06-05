# Latent Policy Guard (LPG)

**Latent Policy Guard** is a guardrail model that performs *semantic latent deliberation* over dynamic safety policies. Given a content snippet and an indexed policy list, LPG reasons internally over the user's **intent** and the **risk** of policy violation as continuous latent states, and emits only a compact verdict anchored to violated policy indices — preserving auditability while avoiding the latency cost of explicit chain-of-thought guardrails.


## Installation

```bash
pip install -r requirements.txt
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
