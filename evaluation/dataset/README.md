# Augmented DynaBench for Latent Policy Guard

This directory holds the augmented DynaBench evaluation set used for the
Latent Policy Guard (LPG) policy-grounded evaluation in the paper
(Section 5.1, "DynaBench (Aug)"). Relative to the original DynaBench test
release, two perturbations are applied: (i) each policy list is randomly
shuffled, and (ii) a subset receives counterfactual variants in which the
violated rule is removed from the policy list, so the corresponding label
flips to safe. The shuffling tests whether the model anchors on policy
content rather than position, and the counterfactuals test whether verdicts
are grounded in the specific violated clause.

## Files

- `dynabench_latest.json`: 543 examples (326 `PASS` / 217 `FAIL`).

## Schema

Each example contains:

- `policy`: policy text (numbered rule list) shown to the guard model.
- `transcript`: the user/agent conversation to evaluate.
- `label`: `PASS` (safe) or `FAIL` (a rule is violated).
- `metadata`: serialized JSON with `rules_violated` and generation attributes.
- `base_id`: source/example identifier.

## Usage

The dataset is registered as `dynabench`, so the evaluation harness reads it
directly:

```bash
cd evaluation
python evaluate.py \
    --model latent_policy_guard \
    --model_path Qwen/Qwen3-4B \
    --ckpt_dir   /path/to/LPG_4B \
    --dataset    dynabench \
    --dataset_path dataset/dynabench_latest.json \
    --no_system_prompt
```

## Related links

- Model: https://huggingface.co/andyc03/LPG_4B
- Training data: https://huggingface.co/datasets/andyc03/latent-policy-guard-40k
- Code: https://github.com/SaFo-Lab/Latent_Policy_Guard
- Paper: https://arxiv.org/abs/2605.17329
