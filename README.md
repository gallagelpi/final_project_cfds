The architecture of the library that we have created is the following:

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

The library follows a **modular design pattern** where **abstract base classes** are used to extend each part of the library. This README includes concrete examples from the current codebase.

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

**Concrete example (current)**
```python
from best_library.data.load_data import LoadData
from best_library.preprocessing.preprocessing import Preprocessing

transform = Preprocessing(img_size=224).get_transform()
loader = LoadData(work_dir="data", transform=transform)
train_loader, val_loader, class_names = loader.load_and_split(batch_size=32)
```

Tips: validate directories, surface clear errors, keep batch_size configurable.

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

**Concrete example (current)**
```python
from best_library.split.split_train_test import DatasetSplitter

splitter = DatasetSplitter(
    dataset_dir="dataset",
    work_dir="data",
    train_ratio=0.8,
    seed=42,
)
splitter.split()
```

Tips: support custom class maps, ensure idempotent outputs (clear/rebuild folders safely).

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

**Concrete example (current)**
```python
from best_library.preprocessing.preprocessing import Preprocessing

prep = Preprocessing(img_size=224)
transform = prep.get_transform()
```

Tips: keep transforms composable; accept img_size/augmentation flags; document expected input (PIL) and output (torch.Tensor).

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

**Concrete example (current)**
```python
from best_library.features.feature_engineering import FeatureBuilder

fb = FeatureBuilder()
# given a batch from a dataloader
# images: torch.Tensor of shape (B, C, H, W)
image_tensor = next(iter(train_loader))[0][0]  # first image from first batch
feature_vec = fb.extract_features(image_tensor)  # concatenates color, histogram, edges, entropy, freq
```

Tips: keep functions pure; return consistent shapes; support GPU tensors; document feature meaning.

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

**Concrete example (current)**
```python
from best_library.model.model_resnet18 import ModelResnet18

model_api = ModelResnet18(device="cuda", class_names=["alpaca", "not_alpaca"])
model = model_api.build_model()
best_val = model_api.train_model(model, train_loader, val_loader, epochs=5, lr=1e-3)
label, score = model_api.predict(image_path, model, transform)
```

Tips: support pretrained weights, checkpointing, early stopping, and device placement.

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

**Concrete example (current)**
```python
from best_library.evaluation.evaluate import Evaluator

evaluator = Evaluator(device="cuda")
acc = evaluator.evaluate(model, val_loader)
```

Tips: extend with precision/recall/F1; ensure no_grad; keep outputs typed (float or dict).

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

**Concrete example (current)**
```python
from best_library.hyperparameter_tuning.tuner import HyperparameterTuner

grid = {"lr": [1e-3, 1e-4], "epochs": [3, 5]}
tuner = HyperparameterTuner(param_grid=grid, device="cuda")
best_params, best_acc = tuner.tune(train_loader, val_loader, save_path="models/best.pth")
```

Tips: log trials, fix seeds, and make saving optional/configurable.

---

## Contribution Checklist
- Inherit from the appropriate base class for the component you add.
- Implement the single primary method listed above (plus required helpers).
- Add a unit test in `tests/` covering the primary method.
- Update docstrings with inputs/outputs and defaults.
- If you add dependencies, update `pyproject.toml`.
