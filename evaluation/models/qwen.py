"""
Qwen model implementation for policy evaluation.

This module provides implementations for Qwen models using HuggingFace
transformers and VLLM backends.
"""

import re
import json
import time
from typing import Dict, Any, Optional

import torch

from .base import BaseModel, ModelInfo, ModelOutput, register_model


def normalize_top_level_policy_headers(policies: str) -> str:
    """
    Normalize top-level numbered policy headers from ``N. ...`` to ``N: ...``.

    Training data uses ``N:`` for top-level policy entries, but raw DynaBench
    policy strings arrive as ``N.`` blocks. Nested numbered sublists are left
    untouched by only rewriting the next expected top-level index.
    """
    lines = policies.splitlines()
    expected_index = 0

    for idx, line in enumerate(lines):
        match = re.match(r"^(\d+)\.\s+(.*)$", line)
        if not match:
            continue

        rule_index = int(match.group(1))
        if rule_index != expected_index:
            continue

        lines[idx] = f"{rule_index}: {match.group(2)}"
        expected_index += 1

    return "\n".join(lines)


def build_user_input(
    content: str,
    policy: str = None,
    policies: list = None,
    response: str = None,
    dataset_type: str = "single_policy"
) -> str:
    """
    Build user input from content, policy, and optional response.
    
    Args:
        content: The main content to evaluate
        policy: The policy text (for single-policy datasets)
        policies: List of policies (for multi-policy datasets)
        response: Optional model response
        dataset_type: "single_policy" or "multi_policy"
    
    Returns:
        Formatted user input string
    """
    if dataset_type == "single_policy":
        content_to_eval = content
        if response:
            content_to_eval = f"{content}\n\nModel response: {response}"

        if policy:
            return (
                f"Policy:\n{policy}\n\n"
                f"Transcript:\n{content_to_eval}"
            )
        else:
            return f"Transcript:\n{content_to_eval}"
    else:
        content_to_eval = content
        if response:
            content_to_eval = f"{content}\n\nModel response: {response}"
        
        if isinstance(policies, str):
            policies_block = normalize_top_level_policy_headers(policies)
        else:
            policies_str_lines = [
                f"{idx}: {p}" for idx, p in enumerate(policies or [])
            ]
            policies_block = "\n".join(policies_str_lines)

        if policies_block:
            return (
                "**Content to evaluate:**\n"
                f"{content_to_eval}\n\n"
                "**Safety policies (indexed from 0):**\n"
                f"{policies_block}"
            )
        else:
            return (
                "**Content to evaluate:**\n"
                f"{content_to_eval}"
            )


def extract_json_from_text(text: str) -> dict:
    """Extract JSON object from model output text."""
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    violation = False
    violated_indices = []
    
    if '"violation":' in text or "'violation':" in text:
        violation_match = re.search(r'["\']violation["\']\s*:\s*(true|false)', text, re.IGNORECASE)
        if violation_match:
            violation = violation_match.group(1).lower() == "true"
    
    if '"violated_rules":' in text or "'violated_rules':" in text:
        indices_match = re.search(r'["\']violated_rules["\']\s*:\s*\[([^\]]*)\]', text)
        if indices_match:
            indices_str = indices_match.group(1)
            try:
                violated_indices = [int(x.strip()) for x in indices_str.split(",") if x.strip().isdigit()]
            except ValueError:
                pass
    
    return {
        "violation": violation,
        "violated_rules": violated_indices
    }


@register_model
class QwenHuggingFaceModel(BaseModel):
    """
    Qwen model using HuggingFace transformers.
    
    This model loads Qwen models using the transformers library
    and supports both standard and trained variants.
    """
    
    @classmethod
    def get_info(cls) -> ModelInfo:
        return ModelInfo(
            name="qwen_hf",
            description="Qwen model using HuggingFace transformers",
            model_type="huggingface",
            supports_system_prompt=True,
            max_context_length=32768,
        )
    
    def load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
        
        enable_thinking = self.config.get("enable_thinking", False)
        
        try:
            from transformers import AutoModelForVision2Seq
            processor = AutoProcessor.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            self._tokenizer = processor
        except Exception:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True,
                enable_thinking=enable_thinking
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        
        model.eval()
        self._model = model
    
    def generate(
        self,
        system_prompt: str,
        user_input: Optional[str] = None,
        content: Optional[str] = None,
        policy: Optional[str] = None,
        policies: Optional[list] = None,
        response: Optional[str] = None,
        dataset_type: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        think: Optional[str] = None,
        **kwargs
    ) -> ModelOutput:
        if hasattr(self._tokenizer, "tokenizer"):
            tokenizer = self._tokenizer.tokenizer
        else:
            tokenizer = self._tokenizer
        
        if user_input is None:
            user_input = build_user_input(
                content=content,
                policy=policy,
                policies=policies,
                response=response,
                dataset_type=dataset_type or "single_policy"
            )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        enable_thinking = self.config.get("enable_thinking", False)
        if think == "free":
            enable_thinking = True
        elif think == "structured":
            enable_thinking = False
        elif think == "nothink":
            enable_thinking = False
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            enable_thinking=enable_thinking,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(text, return_tensors="pt")
        input_length = inputs.input_ids.shape[1]
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        top_p = kwargs.get("top_p", 0.8)
        top_k = kwargs.get("top_k", 20)
        repetition_penalty = kwargs.get("repetition_penalty", 1.15)
        
        with torch.no_grad():
            start_time = time.perf_counter()
            do_sample = temperature > 0.0
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
            }
            if do_sample:
                gen_kwargs.update({
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "min_p": 0.0,
                    "repetition_penalty": repetition_penalty,
                })
            output_ids = self._model.generate(**inputs, **gen_kwargs)
            inference_time = time.perf_counter() - start_time
            
            response_ids = output_ids[0][input_length:].tolist()
            response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
            if tokenizer.eos_token:
                response_text = response_text.replace(tokenizer.eos_token, "")
        
        original_output = response_text.strip()
        prediction_json = extract_json_from_text(original_output)
        
        tokens_generated = len(response_ids)
        
        return ModelOutput(
            prediction=prediction_json,
            raw_output=original_output,
            inference_time=inference_time,
            tokens_generated=tokens_generated,
        )


@register_model
class QwenVLLMModel(BaseModel):
    """
    Qwen model using VLLM server.
    
    This model connects to a VLLM server and uses the OpenAI-compatible API
    for inference.
    """
    
    @classmethod
    def get_info(cls) -> ModelInfo:
        return ModelInfo(
            name="qwen_vllm",
            description="Qwen model using VLLM server",
            model_type="vllm",
            supports_system_prompt=True,
            max_context_length=32768,
        )
    
    def load(self) -> None:
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for VLLM support. "
                "Install with: pip install openai"
            )
        
        vllm_url = self.config.get("vllm_url", "http://localhost:8000")
        self._client = OpenAI(
            base_url=f"{vllm_url}/v1",
            api_key="EMPTY"
        )
        self._model_name = self.config.get("model_name", self.model_path)
        self._model = True

    def _build_messages(self, system_prompt: str, user_input: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
    
    def generate(
        self,
        system_prompt: str,
        user_input: Optional[str] = None,
        content: Optional[str] = None,
        policy: Optional[str] = None,
        policies: Optional[list] = None,
        response: Optional[str] = None,
        dataset_type: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        think: Optional[str] = None,
        **kwargs
    ) -> ModelOutput:
        if user_input is None:
            user_input = build_user_input(
                content=content,
                policy=policy,
                policies=policies,
                response=response,
                dataset_type=dataset_type or "single_policy"
            )

        # The Qwen3 chat template controls thinking via `enable_thinking`,
        # but the OpenAI-compatible vLLM API doesn't expose that flag, so
        # we toggle thinking from the client via the /think /no_think
        # suffix. Default is /no_think because guardrail evaluation wants
        # deterministic short outputs; --think free switches to /think.
        if think == "free":
            user_input = user_input + " /think"
        else:
            user_input = user_input + " /no_think"

        messages = self._build_messages(system_prompt, user_input)
                
        top_p = kwargs.get("top_p", 0.8)
        
        start_time = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        inference_time = time.perf_counter() - start_time
        
        original_output = response.choices[0].message.content
        prediction_json = extract_json_from_text(original_output)
        
        tokens_generated = response.usage.completion_tokens if response.usage else None
        
        return ModelOutput(
            prediction=prediction_json,
            raw_output=original_output,
            inference_time=inference_time,
            tokens_generated=tokens_generated,
        )


@register_model
class TrainedQwenModel(QwenVLLMModel):
    """
    Trained Qwen model using VLLM server with custom system prompt support.
    
    This model extends the base Qwen VLLM model with additional support
    for trained checkpoints and custom system prompt configurations.
    """
    
    @classmethod
    def get_info(cls) -> ModelInfo:
        return ModelInfo(
            name="trained_qwen",
            description="Trained Qwen model using VLLM with custom system prompt support",
            model_type="vllm",
            supports_system_prompt=True,
            max_context_length=32768,
        )
    
    def __init__(self, model_path: str, **kwargs):
        super().__init__(model_path, **kwargs)
        self.checkpoint_path = kwargs.get("checkpoint_path", None)
        self.system_prompt_file = kwargs.get("system_prompt_file", None)
    
    def load(self) -> None:
        super().load()
        
        if self.system_prompt_file:
            with open(self.system_prompt_file, 'r') as f:
                self.default_system_prompt = f.read().strip()
        else:
            self.default_system_prompt = None

    def _build_messages(self, system_prompt: str, user_input: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": user_input}]
