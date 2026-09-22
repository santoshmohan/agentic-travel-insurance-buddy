import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class GoldenDataset:
    """Load and manage golden dataset for evaluation."""
    
    def __init__(self, dataset_path: str = "golden-dataset/golden_dataset.json"):
        self.dataset_path = Path(dataset_path)
        self.data = self._load()
    
    def _load(self) -> List[Dict[str, Any]]:
        """Load golden dataset from JSON."""
        if not self.dataset_path.exists():
            logger.warning(f"Golden dataset not found at {self.dataset_path}")
            return []
        
        with open(self.dataset_path, "r") as f:
            data = json.load(f)
        
        logger.info(f"Loaded {len(data)} golden examples")
        return data
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all examples."""
        return self.data
    
    def get_by_provider(self, provider: str) -> List[Dict[str, Any]]:
        """Get examples for specific provider."""
        return [ex for ex in self.data if ex.get("provider") == provider]
    
    def __len__(self):
        return len(self.data)
    
    def __iter__(self):
        return iter(self.data)