"""
Base dataset interface and registry for the Policy Evaluate framework.

This module provides the abstract base class that all dataset plugins must implement,
as well as a registry system for discovering and loading datasets.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Type, Iterator
import json


@dataclass
class DatasetSample:
    """
    Standardized sample format for evaluation.
    
    Attributes:
        sample_id: Unique identifier for the sample
        content: The main text content to evaluate (transcript, prompt, etc.)
        safe: Whether the content is safe (True) or unsafe (False)
        policy: The policy text (for single-policy datasets)
        policies: List of policies (for multi-policy datasets)
        response: Optional model response associated with the content
        violated_rule: The violated rule number (for single-rule datasets)
        violated_rules: List of violated rule numbers (for multi-rule datasets)
        violated_policy_index: Index of violated policy (for multi-policy datasets)
        metadata: Additional metadata about the sample
    """
    sample_id: str
    content: str
    safe: bool
    policy: Optional[str] = None
    policies: Optional[List[str]] = None
    response: Optional[str] = None
    violated_rule: Optional[int] = None
    violated_rules: Optional[List[int]] = None
    violated_policy_index: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary."""
        return {
            "sample_id": self.sample_id,
            "content": self.content,
            "safe": self.safe,
            "policy": self.policy,
            "policies": self.policies,
            "response": self.response,
            "violated_rule": self.violated_rule,
            "violated_rules": self.violated_rules,
            "violated_policy_index": self.violated_policy_index,
            "metadata": self.metadata,
        }


@dataclass
class DatasetInfo:
    """
    Metadata about a dataset.
    
    Attributes:
        name: Unique identifier for the dataset
        description: Human-readable description
        dataset_type: Type of dataset (e.g., "single_policy", "multi_policy")
        num_samples: Number of samples in the dataset
        num_safe: Number of safe samples
        num_unsafe: Number of unsafe samples
        file_format: File format (e.g., "json", "jsonl")
    """
    name: str
    description: str
    dataset_type: str
    num_samples: Optional[int] = None
    num_safe: Optional[int] = None
    num_unsafe: Optional[int] = None
    file_format: str = "json"


class BaseDataset(ABC):
    """
    Abstract base class for all dataset plugins.
    
    All datasets must inherit from this class and implement the required methods.
    This ensures a consistent interface across different dataset formats.
    """
    
    def __init__(self, dataset_path: str, **kwargs):
        """
        Initialize the dataset.
        
        Args:
            dataset_path: Path to the dataset file
            **kwargs: Additional dataset-specific configuration
        """
        self.dataset_path = dataset_path
        self.config = kwargs
        self._data = None
    
    @classmethod
    @abstractmethod
    def get_info(cls) -> DatasetInfo:
        """
        Get metadata about this dataset type.
        
        Returns:
            DatasetInfo object with dataset metadata
        """
        pass
    
    @abstractmethod
    def load(self) -> None:
        """
        Load the dataset into memory.
        
        This method should load the dataset and store it in self._data.
        """
        pass
    
    @abstractmethod
    def __iter__(self) -> Iterator[DatasetSample]:
        """
        Iterate over samples in the dataset.
        
        Yields:
            DatasetSample objects
        """
        pass
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass
    
    def __getitem__(self, index: int) -> DatasetSample:
        """Get a sample by index."""
        for i, sample in enumerate(self):
            if i == index:
                return sample
        raise IndexError(f"Index {index} out of range")
    
    def is_loaded(self) -> bool:
        """Check if the dataset is loaded."""
        return self._data is not None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics about the dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        stats = {
            "total_samples": len(self),
            "safe_samples": 0,
            "unsafe_samples": 0,
        }
        
        for sample in self:
            if sample.safe:
                stats["safe_samples"] += 1
            else:
                stats["unsafe_samples"] += 1
        
        stats["safe_ratio"] = stats["safe_samples"] / stats["total_samples"] if stats["total_samples"] > 0 else 0
        
        return stats
    
    def __repr__(self) -> str:
        info = self.get_info()
        return f"{self.__class__.__name__}(name='{info.name}', path='{self.dataset_path}')"


class DatasetRegistry:
    """
    Registry for dataset plugins.
    
    This class provides a central registry for all available datasets.
    Datasets can be registered using the @register_dataset decorator or
    by calling register() directly.
    """
    
    _datasets: Dict[str, Type[BaseDataset]] = {}
    
    @classmethod
    def register(cls, dataset_class: Type[BaseDataset]) -> Type[BaseDataset]:
        """
        Register a dataset class.
        
        Args:
            dataset_class: The dataset class to register
            
        Returns:
            The same dataset class (for decorator usage)
        """
        info = dataset_class.get_info()
        cls._datasets[info.name] = dataset_class
        return dataset_class
    
    @classmethod
    def get(cls, name: str) -> Type[BaseDataset]:
        """
        Get a dataset class by name.
        
        Args:
            name: The registered name of the dataset
            
        Returns:
            The dataset class
            
        Raises:
            KeyError: If the dataset is not registered
        """
        if name not in cls._datasets:
            available = list(cls._datasets.keys())
            raise KeyError(f"Dataset '{name}' not registered. Available: {available}")
        return cls._datasets[name]
    
    @classmethod
    def list_datasets(cls) -> List[str]:
        """List all registered dataset names."""
        return list(cls._datasets.keys())
    
    @classmethod
    def get_dataset_info(cls, name: str) -> DatasetInfo:
        """Get info about a registered dataset."""
        dataset_class = cls.get(name)
        return dataset_class.get_info()
    
    @classmethod
    def create(cls, name: str, dataset_path: str, **kwargs) -> BaseDataset:
        """
        Create an instance of a registered dataset.
        
        Args:
            name: The registered name of the dataset
            dataset_path: Path to the dataset file
            **kwargs: Additional dataset configuration
            
        Returns:
            An instance of the dataset
        """
        dataset_class = cls.get(name)
        return dataset_class(dataset_path, **kwargs)


def register_dataset(dataset_class: Type[BaseDataset]) -> Type[BaseDataset]:
    """
    Decorator to register a dataset class.
    
    Usage:
        @register_dataset
        class MyDataset(BaseDataset):
            ...
    """
    return DatasetRegistry.register(dataset_class)
