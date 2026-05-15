"""
Base model interface and registry for the Policy Evaluate framework.

This module provides the abstract base class that all model plugins must implement,
as well as a registry system for discovering and loading models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Type
import time


@dataclass
class ModelOutput:
    """
    Standardized output from model inference.
    
    Attributes:
        prediction: The model's prediction (format depends on task)
        raw_output: The raw text output from the model
        inference_time: Time taken for inference in seconds
        tokens_generated: Number of tokens generated (optional)
        metadata: Additional metadata about the inference
    """
    prediction: Dict[str, Any]
    raw_output: str
    inference_time: float
    tokens_generated: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ModelInfo:
    """
    Metadata about a model.
    
    Attributes:
        name: Unique identifier for the model
        description: Human-readable description
        model_type: Type of model (e.g., "huggingface", "vllm", "api")
        supports_system_prompt: Whether the model supports system prompts
        max_context_length: Maximum context length (if known)
    """
    name: str
    description: str
    model_type: str
    supports_system_prompt: bool = True
    max_context_length: Optional[int] = None


class BaseModel(ABC):
    """
    Abstract base class for all model plugins.
    
    All models must inherit from this class and implement the required methods.
    This ensures a consistent interface across different model backends.
    """
    
    def __init__(self, model_path: str, **kwargs):
        """
        Initialize the model.
        
        Args:
            model_path: Path or identifier for the model
            **kwargs: Additional model-specific configuration
        """
        self.model_path = model_path
        self.config = kwargs
        self._model = None
        self._tokenizer = None
    
    @classmethod
    @abstractmethod
    def get_info(cls) -> ModelInfo:
        """
        Get metadata about this model type.
        
        Returns:
            ModelInfo object with model metadata
        """
        pass
    
    @abstractmethod
    def load(self) -> None:
        """
        Load the model into memory.
        
        This method should load the model and tokenizer/processor
        and store them in self._model and self._tokenizer.
        """
        pass
    
    @abstractmethod
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
        """
        Generate a response from the model.
        
        Args:
            system_prompt: The system prompt to use
            user_input: Pre-formatted user input (for backward compatibility)
            content: The main content to evaluate (transcript, prompt, etc.)
            policy: The policy text (for single-policy datasets)
            policies: List of policies (for multi-policy datasets)
            response: Optional model response associated with the content
            dataset_type: Type of dataset ("single_policy" or "multi_policy")
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            think: Thinking mode (nothink, free, structured)
            **kwargs: Additional generation parameters
            
        Returns:
            ModelOutput containing the prediction and metadata
        """
        pass
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None
    
    def unload(self) -> None:
        """Unload the model from memory."""
        import gc
        import torch
        
        self._model = None
        self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __repr__(self) -> str:
        info = self.get_info()
        return f"{self.__class__.__name__}(name='{info.name}', path='{self.model_path}')"


class ModelRegistry:
    """
    Registry for model plugins.
    
    This class provides a central registry for all available models.
    Models can be registered using the @register_model decorator or
    by calling register() directly.
    """
    
    _models: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register(cls, model_class: Type[BaseModel]) -> Type[BaseModel]:
        """
        Register a model class.
        
        Args:
            model_class: The model class to register
            
        Returns:
            The same model class (for decorator usage)
        """
        info = model_class.get_info()
        cls._models[info.name] = model_class
        return model_class
    
    @classmethod
    def get(cls, name: str) -> Type[BaseModel]:
        """
        Get a model class by name.
        
        Args:
            name: The registered name of the model
            
        Returns:
            The model class
            
        Raises:
            KeyError: If the model is not registered
        """
        if name not in cls._models:
            available = list(cls._models.keys())
            raise KeyError(f"Model '{name}' not registered. Available: {available}")
        return cls._models[name]
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered model names."""
        return list(cls._models.keys())
    
    @classmethod
    def get_model_info(cls, name: str) -> ModelInfo:
        """Get info about a registered model."""
        model_class = cls.get(name)
        return model_class.get_info()
    
    @classmethod
    def create(cls, name: str, model_path: str, **kwargs) -> BaseModel:
        """
        Create an instance of a registered model.
        
        Args:
            name: The registered name of the model
            model_path: Path or identifier for the model
            **kwargs: Additional model configuration
            
        Returns:
            An instance of the model
        """
        model_class = cls.get(name)
        return model_class(model_path, **kwargs)


def register_model(model_class: Type[BaseModel]) -> Type[BaseModel]:
    """
    Decorator to register a model class.
    
    Usage:
        @register_model
        class MyModel(BaseModel):
            ...
    """
    return ModelRegistry.register(model_class)
