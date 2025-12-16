The architecture of the library that we have created is the following:
```
scr/
└── best_library/
    │
    ├── __init__.py
    ├── base.py
    │
    ├── preprocessing/
    │   ├── __init__.py
    │   └── preprocessing.py
    │       └── Preprocessing class
    |
    ├── split/
    │   ├── __init__.py
    │   └── slpit_train_and_test.py
    │       └── DatasetSplitter class
    |
    ├── data/
    │   ├── __init__.py
    │   └── load_data.py
    │       └── LoadData class
    |
    ├── features/
    │   ├── __init__.py
    │   └── feature_engineering.py
    │       └── FeatureBuilder class
    │
    ├── model/
    │   ├── __init__.py
    │   └── model_resnet18.py
    │       └── ModelResnet18 class
    |
    ├── hyperparameter_tuning/
    │   ├── __init__.py
    │   └── tuner.py
    │       └── HyperparameterTuner class
    │
    └── evaluation/
        ├── __init__.py
        └── evaluate.py
            └── Evaluator class
```

Observations:

1. The structure of the library follows a modern Python package layout, using a src/-based architecture to clearly separate library code from project configuration and notebooks.

2. Each folder and subfolder contains a __init__.py file, which marks the directory as a Python package and allows its modules and classes to be imported correctly.

3. In our project, each subfolder represents a specific step of the machine learning pipeline. This separation allows each component to evolve independently and promotes code reuse. In this case:

    - Preprocessing: Creates the image transformations that will later be applied to the dataset.
    - Split: Splits the raw dataset into train and val folders, each containing alpaca and not_alpaca subfolders.
    - Data: Loads the dataset from disk, applies the preprocessing transformations, and creates iterable DataLoaders for the training and validation sets.
    - Features: Builds five independent handcrafted feature sets from the images.
    - Model: Defines the ResNet18-based model and provides methods to build the model, train it, load a trained model, and generate predictions.
    - Hyperparameter tuning: Performs hyperparameter tuning to select the best set of training parameters.
    - Evaluation: Evaluates the trained model using performance metrics (e.g. validation accuracy).

 ## Scaling Guidelines

The library should follow a **modular design pattern** where **abstract base classes** to extend each part of the library. This README includes concrete examples from the current codebase.

## How to Contribute (Abstract-Class First)

1. **Review the architecture** - Understand which components use abstract classes
2. **Check existing implementations** - Study similar components as templates
3. **Follow the inheritance hierarchy** - Inherit from base classes where they exist
4. **Write tests first** - Test-driven development ensures compatibility
5. **Update documentation** - Add your component to this README

---

## 1) Data
- **Primary method**: `load_and_split`
- **Existing impl**: `LoadData` in `src/best_library/data/load_data.py`

**Abstract contract (suggested)**
```python
from abc import ABC, abstractmethod
from typing import Tuple

class BaseDataLoader(ABC):
    @abstractmethod
    def load_and_split(self, batch_size: int) -> Tuple[object, object, list]:
        """Return train_loader, val_loader, class_names."""
        raise NotImplementedError
```

**Class (current)**
```python
# src/best_library/data/load_data.py
class LoadData:
    def __init__(self, work_dir: str, transform):
        self.work_dir = work_dir
        self.transform = transform

    def load_and_split(self, batch_size: int):
        """Return train_loader, val_loader, class_names."""
        ...
```

**Class (adapted from abstract)**
```python
# src/best_library/data/load_data.py
from best_library.base import BaseDataLoader

class LoadData(BaseDataLoader):
    def __init__(self, work_dir: str, transform):
        self.work_dir = work_dir
        self.transform = transform

    def load_and_split(self, batch_size: int) -> Tuple[object, object, list]:
        """Return train_loader, val_loader, class_names."""
        ...
```


---

## 2) Split
- **Primary method**: `split`
- **Existing impl**: `DatasetSplitter` in `src/best_library/split/split_train_test.py`

**Abstract contract (suggested)**
```python
from abc import ABC, abstractmethod

class BaseSplitter(ABC):
    @abstractmethod
    def split(self) -> None:
        """Create train/val folders from a raw dataset."""
        raise NotImplementedError
```

**Class (current)**
```python
# src/best_library/split/split_train_test.py
class DatasetSplitter:
    def __init__(self, dataset_dir: str, work_dir: str, train_ratio: float = 0.8, seed: int = 42):
        ...

    def split(self) -> None:
        ...
```

**Class (adapted from abstract)**
```python
# src/best_library/split/split_train_test.py
from best_library.base import BaseSplitter

class DatasetSplitter(BaseSplitter):
    def __init__(self, dataset_dir: str, work_dir: str, train_ratio: float = 0.8, seed: int = 42):
        ...

    def split(self) -> None:
        """Create train/val folders from a raw dataset."""
        ...
```


---

## 3) Preprocessing
- **Primary method**: `get_transform`
- **Existing impl**: `Preprocessing` in `src/best_library/preprocessing/preprocessing.py`

**Abstract contract (suggested)**
```python
from abc import ABC, abstractmethod

class BasePreprocessing(ABC):
    @abstractmethod
    def get_transform(self):
        """Return a callable (e.g., torchvision.transforms.Compose)."""
        raise NotImplementedError
```

**Class (current)**
```python
# src/best_library/preprocessing/preprocessing.py
class Preprocessing:
    def __init__(self, img_size: int = 224):
        ...

    def get_transform(self):
        ...
```

**Class (adapted from abstract)**
```python
# src/best_library/preprocessing/preprocessing.py
from best_library.base import BasePreprocessing

class Preprocessing(BasePreprocessing):
    def __init__(self, img_size: int = 224):
        ...

    def get_transform(self):
        """Return a callable (e.g., torchvision.transforms.Compose)."""
        ...
```


---

## 4) Features
- **Primary method**: `extract_features`
- **Existing impl**: `FeatureBuilder` in `src/best_library/features/feature_engineering.py`

**Abstract contract (suggested)**
```python
from abc import ABC, abstractmethod

class BaseFeatureExtractor(ABC):
    @abstractmethod
    def extract_features(self, image) -> object:
        """Return a feature vector/tensor for a single image tensor (C,H,W)."""
        raise NotImplementedError
```

**Class (current)**
```python
# src/best_library/features/feature_engineering.py
class FeatureBuilder:
    def extract_features(self, image):
        ...
```

**Class (adapted from abstract)**
```python
# src/best_library/features/feature_engineering.py
from best_library.base import BaseFeatureExtractor

class FeatureBuilder(BaseFeatureExtractor):
    def extract_features(self, image) -> object:
        """Return a feature vector/tensor for a single image tensor (C,H,W)."""
        ...
```

---

## 5) Model
- **Primary methods**: `build_model`, `train_model`, `load_trained_model`, `predict`
- **Existing impl**: `ModelResnet18` in `src/best_library/model/model_resnet18.py`

**Abstract contract (suggested)**
```python
from abc import ABC, abstractmethod

class BaseModel(ABC):
    def __init__(self, device: str, class_names: list):
        self.device = device
        self.class_names = class_names

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @abstractmethod
    def build_model(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train_model(self, model, train_loader, val_loader, epochs: int, lr: float):
        raise NotImplementedError

    @abstractmethod
    def load_trained_model(self, model_path: str, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def predict(self, image_path: str, model, transform):
        raise NotImplementedError
```

**Class (current)**
```python
# src/best_library/model/model_resnet18.py
class ModelResnet18:
    def build_model(self, num_classes: int | None = None, weights: str | None = "IMAGENET1K_V1"):
        ...
    def train_model(self, model, train_loader, val_loader, epochs: int, lr: float) -> float:
        ...
    def load_trained_model(self, model_path: str, num_classes: int | None = None):
        ...
    def predict(self, image_path: str, model, transform):
        ...
```

**Class (adapted from abstract)**
```python
# src/best_library/model/model_resnet18.py
from best_library.base import BaseModel

class ModelResnet18(BaseModel):
    def __init__(self, device: str, class_names: list):
        self.device = device
        self.class_names = class_names

    def build_model(self, num_classes: int | None = None, weights: str | None = "IMAGENET1K_V1"):
        ...
    
    def train_model(self, model, train_loader, val_loader, epochs: int, lr: float) -> float:
        ...
    
    def load_trained_model(self, model_path: str, num_classes: int | None = None):
        ...
    
    def predict(self, image_path: str, model, transform):
        ...
```

---

## 6) Evaluation
- **Primary method**: `evaluate`
- **Existing impl**: `Evaluator` in `src/best_library/evaluation/evaluate.py`

**Abstract contract (suggested)**
```python
from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, model, val_loader) -> float:
        """Return an evaluation score (e.g., accuracy)."""
        raise NotImplementedError
```

**Class (current)**
```python
# src/best_library/evaluation/evaluate.py
class Evaluator:
    def __init__(self, device: str):
        ...
    def evaluate(self, model, val_loader) -> float:
        ...
```

**Class (adapted from abstract)**
```python
# src/best_library/evaluation/evaluate.py
from best_library.base import BaseEvaluator

class Evaluator(BaseEvaluator):
    def __init__(self, device: str):
        ...
    
    def evaluate(self, model, val_loader) -> float:
        """Return an evaluation score (e.g., accuracy)."""
        ...
```

---

## 7) Hyperparameter Tuning
- **Primary method**: `tune`
- **Existing impl**: `HyperparameterTuner` in `src/best_library/hyperparameter_tuning/tuner.py`

**Abstract contract (suggested)**
```python
from abc import ABC, abstractmethod

class BaseTuner(ABC):
    @abstractmethod
    def tune(self, train_loader, val_loader, save_path: str):
        """Search hyperparameters, return best_params and best_score."""
        raise NotImplementedError
```

**Class (current)**
```python
# src/best_library/hyperparameter_tuning/tuner.py
class HyperparameterTuner:
    def __init__(self, param_grid: dict, device: str | None = None):
        ...
    def tune(self, train_loader, val_loader, save_path: str):
        ...
```

**Class (adapted from abstract)**
```python
# src/best_library/hyperparameter_tuning/tuner.py
from best_library.base import BaseTuner

class HyperparameterTuner(BaseTuner):
    def __init__(self, param_grid: dict, device: str | None = None):
        ...
    
    def tune(self, train_loader, val_loader, save_path: str):
        """Search hyperparameters, return best_params and best_score."""
        ...
```

---

## Quick Contribution Checklist
- Inherit from the appropriate base class for the component you add.
- Implement the single primary method listed above (plus required helpers).
- Add a unit test in `tests/` covering the primary method.
- Update docstrings with inputs/outputs and defaults.
- If you add dependencies, update `pyproject.toml`.

## Contributing

When contributing new components:
1. Follow the established patterns and interfaces
2. Write comprehensive tests
3. Document all public APIs
4. Ensure backward compatibility
5. Update this README with usage examples
