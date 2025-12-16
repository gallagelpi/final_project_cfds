# final_project_cfds

## Scaling Guidelines

The library follows a **modular design pattern** where **abstract base classes** are used strategically for components requiring strict contracts, while **flexible patterns** are used where variation is expected.

### When to Use Abstract Base Classes (Strict Contracts)
- **Models**: Enforce consistent training/prediction interfaces for interchangeability
- **Evaluators**: Standardize metric computation for pipeline consistency

### When Flexibility is Preferred (Informal Interfaces)
- **Preprocessors**: Use-case specific transformations vary greatly across domains
- **Feature Engineering**: Different features have different signatures and outputs
- **Data Loaders**: Dataset structure determines implementation
- **Utilities**: Helper functions don't need rigid contracts

**Key Insight**: Only enforce abstraction where polymorphism is essential. Too many abstract classes create unnecessary rigidity.

### Contribution Workflow

Before contributing, new team members should:

1. **Review the architecture** - Understand which components use abstract classes
2. **Check existing implementations** - Study similar components as templates
3. **Follow the inheritance hierarchy** - Inherit from base classes where they exist
4. **Write tests first** - Test-driven development ensures compatibility
5. **Update documentation** - Add your component to this README

---

## Abstract Base Classes Pattern

### Core Principle
Use abstract base classes **only when necessary** - specifically for components that:
1. Need to be **swappable** in the pipeline (e.g., different model architectures)
2. Require **guaranteed interfaces** for integration
3. Benefit from **type checking** and IDE support

**Don't overuse abstraction** - it can hinder flexibility and make simple additions complex.

### Example: Model Base Class (RECOMMENDED for abstraction)

```python
# src/best_library/base.py

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import torch.nn as nn

class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, device: str, class_names: List[str]):
        self.device = device
        self.class_names = class_names
    
    @property
    def num_classes(self) -> int:
        return len(self.class_names)
    
    @abstractmethod
    def build_model(self, **kwargs) -> nn.Module:
        """Build and return the model architecture."""
        pass
    
    @abstractmethod
    def train_model(self, model: nn.Module, train_loader, val_loader, 
                    epochs: int, learning_rate: float, **kwargs) -> Dict[str, Any]:
        """Train the model and return training history."""
        pass
    
    @abstractmethod
    def load_trained_model(self, model_path: str) -> nn.Module:
        """Load model from checkpoint."""
        pass
    
    @abstractmethod
    def predict(self, model: nn.Module, image, transform) -> Dict[str, Any]:
        """Predict single image and return results with confidence."""
        pass
```

### Benefits of Abstract Base Classes
✅ **Interface Guarantee**: All models have the same methods  
✅ **Polymorphism**: Swap models without changing pipeline code  
✅ **Type Safety**: IDE autocomplete and static type checking  
✅ **Documentation**: Clear contract for implementers  
✅ **Consistency**: Enforced by Python's ABC module  

---

## Adding New Preprocessors

### Location
`src/best_library/preprocessing/`

### Why Preprocessors Are NOT Abstract

Preprocessors are **highly use-case dependent** and benefit from flexibility:
- Medical imaging needs different normalization than natural images
- Different augmentation strategies for different domains
- Custom transformations for specific datasets
- Integration with external libraries (Albumentations, imgaug)

### Recommended Pattern (Duck Typing)

Instead of strict inheritance, follow the **informal interface**:

```python
class CustomPreprocessing:
    """
    Informal interface: Must provide get_transform() method.
    No inheritance required - composition over inheritance.
    """
    def __init__(self, **kwargs):
        # Initialize parameters specific to your use case
        pass
    
    def get_transform(self):
        # Return torchvision.transforms.Compose object
        # Or any callable that transforms PIL Image -> torch.Tensor
        pass
```

### Guideline for Contributors

**When adding a preprocessor:**
1. ✅ Implement `get_transform()` method returning a callable
2. ✅ Accept flexible `**kwargs` for use-case specific parameters
3. ✅ Document expected input/output formats in docstrings
4. ✅ Maintain compatibility with PyTorch transforms when possible
5. ❌ Don't force inheritance from a base class
6. ✅ Do ensure your transform is composable with others

### Example: Medical Imaging Preprocessor

```python
import torchvision.transforms as transforms

class MedicalImagePreprocessing:
    """
    Specialized preprocessing for medical images.
    Uses different normalization and no color augmentation.
    """
    def __init__(self, img_size=224, window_level=None, window_width=None):
        self.img_size = img_size
        self.window_level = window_level
        self.window_width = window_width
    
    def get_transform(self):
        transforms_list = [transforms.Resize((self.img_size, self.img_size))]
        
        # Custom windowing for medical images
        if self.window_level and self.window_width:
            transforms_list.append(
                transforms.Lambda(lambda x: self._apply_window(x))
            )
        
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Different normalization
        ])
        
        return transforms.Compose(transforms_list)
    
    def _apply_window(self, image):
        # Custom windowing logic
        pass
```

### Example: Augmentation Preprocessor

```python
class AugmentedPreprocessing:
    """Standard preprocessing with optional augmentation."""
    
    def __init__(self, img_size=224, augment=True, augmentation_strength='medium'):
        self.img_size = img_size
        self.augment = augment
        self.augmentation_strength = augmentation_strength
    
    def get_transform(self):
        if self.augment:
            return self._get_augmented_transform()
        return self._get_standard_transform()
    
    def _get_augmented_transform(self):
        strength_params = {
            'light': {'rotation': 5, 'brightness': 0.1},
            'medium': {'rotation': 15, 'brightness': 0.2},
            'heavy': {'rotation': 30, 'brightness': 0.3}
        }
        params = strength_params[self.augmentation_strength]
        
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(params['rotation']),
            transforms.ColorJitter(brightness=params['brightness']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _get_standard_transform(self):
        return transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
```

---

## Adding New Features

### Location
`src/best_library/features/`

### Why Features Use Functional Pattern

Feature extraction functions are **stateless operations** that don't require class hierarchy:
- Pure functions: Same input → Same output
- Composable: Can combine multiple features
- Flexible: Easy to add without architectural changes
- Testable: Simple unit tests

### Recommended Pattern (Functional)

Use **standalone functions** rather than classes:

```python
def new_feature(image: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Extract feature from image.
    
    Args:
        image: torch.Tensor of shape (C, H, W)
        **kwargs: Additional parameters
    
    Returns:
        torch.Tensor: Feature vector (1D tensor)
    """
    # Implementation
    pass
```

### Example: Texture Features

```python
import torch
import torch.nn.functional as F

def texture_variance(image: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Compute local variance as texture measure.
    
    Higher variance indicates more texture detail.
    """
    gray = image.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, H, W)
    
    # Local mean using convolution
    kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size**2)
    kernel = kernel.to(image.device)
    local_mean = F.conv2d(gray, kernel, padding=kernel_size//2)
    
    # Local variance
    local_sq_mean = F.conv2d(gray**2, kernel, padding=kernel_size//2)
    variance = local_sq_mean - local_mean**2
    
    return variance.mean().unsqueeze(0)  # Return scalar as 1D tensor

def frequency_features(image: torch.Tensor) -> torch.Tensor:
    """Extract frequency domain features using FFT."""
    gray = image.mean(dim=0)  # (H, W)
    
    # 2D FFT
    fft = torch.fft.fft2(gray)
    magnitude = torch.abs(fft)
    
    # Divide into frequency bands
    h, w = magnitude.shape
    low_freq = magnitude[:h//4, :w//4].mean()
    high_freq = magnitude[3*h//4:, 3*w//4:].mean()
    
    return torch.tensor([low_freq, high_freq])
```

### Guidelines for Contributors

**When adding features:**
1. ✅ Keep functions pure (no side effects)
2. ✅ Always return torch.Tensor with consistent shape
3. ✅ Support GPU by using `.to(device)` for created tensors
4. ✅ Document what the feature captures
5. ✅ Handle edge cases (empty images, different sizes)
6. ✅ Optimize for batch processing when possible

---

## Adding New Models

### Location
`src/best_library/model/`

### Guidelines

1. **Create a model class** following the established pattern
2. **Required methods**:
   ```python
   class CustomModel:
       def __init__(self, device: str, class_names: List[str]):
           self.device = device
           self.class_names = class_names
       
       @property
       def num_classes(self) -> int:
           return len(self.class_names)
       
       def build_model(self, **kwargs):
           """Build and return model architecture."""
           pass
       
       def train_model(self, model, train_loader, val_loader, 
                      epochs, learning_rate, **kwargs):
           """Train the model."""
           pass
       
       def load_trained_model(self, model_path: str):
           """Load model from checkpoint."""
           pass
       
       def predict(self, model, image: Image.Image, transform):
           """Predict single image."""
           pass
   ```

3. **Example: Adding EfficientNet**:
   ```python
   import torch.nn as nn
   from torchvision import models
   
   class ModelEfficientNet:
       def __init__(self, device: str, class_names: List[str]):
           self.device = device
           self.class_names = class_names
       
       @property
       def num_classes(self) -> int:
           return len(self.class_names)
       
       def build_model(self, weights="IMAGENET1K_V1"):
           model = models.efficientnet_b0(weights=weights)
           in_features = model.classifier[1].in_features
           model.classifier[1] = nn.Linear(in_features, self.num_classes)
           return model.to(self.device)
       
       def train_model(self, model, train_loader, val_loader,
                      epochs=10, learning_rate=0.001):
           # Implement training loop
           pass
   ```

4. **Best Practices**:
   - Support pretrained weights
   - Implement early stopping
   - Save checkpoints during training
   - Log training metrics
   - Support resume from checkpoint
   - Handle device placement consistently

---

## Adding New Metrics

### Location
`src/best_library/evaluation/`

### Guidelines

1. **Create metric functions** that compute evaluation scores
2. **Function signature pattern**:
   ```python
   def new_metric(predictions, labels, **kwargs):
       """
       Compute metric.
       
       Args:
           predictions: Model predictions
           labels: Ground truth labels
           **kwargs: Additional parameters
       
       Returns:
           float or dict: Metric value(s)
       """
       pass
   ```

3. **Example: Adding Precision, Recall, F1**:
   ```python
   import torch
   from sklearn.metrics import precision_recall_fscore_support
   
   def compute_classification_metrics(model, val_loader, device: str):
       """Compute precision, recall, F1 for each class."""
       model.eval()
       all_preds = []
       all_labels = []
       
       with torch.no_grad():
           for images, labels in val_loader:
               images = images.to(device)
               outputs = model(images)
               preds = torch.argmax(outputs, dim=1)
               all_preds.extend(preds.cpu().numpy())
               all_labels.extend(labels.numpy())
       
       precision, recall, f1, support = precision_recall_fscore_support(
           all_labels, all_preds, average=None
       )
       
       return {
           'precision': precision,
           'recall': recall,
           'f1': f1,
           'support': support,
           'macro_f1': f1.mean()
       }
   ```

4. **Extending the Evaluator Class**:
   ```python
   class AdvancedEvaluator:
       def __init__(self, device: str):
           self.device = device
       
       def evaluate(self, model, val_loader):
           # Basic accuracy
           accuracy = evaluate_model(model, val_loader, self.device)
           return accuracy
       
       def evaluate_detailed(self, model, val_loader):
           # Comprehensive metrics
           metrics = compute_classification_metrics(model, val_loader, self.device)
           metrics['accuracy'] = self.evaluate(model, val_loader)
           return metrics
   ```

5. **Best Practices**:
   - Support both binary and multi-class scenarios
   - Return structured results (dictionaries)
   - Handle edge cases (empty predictions)
   - Provide per-class and aggregate metrics
   - Consider computational efficiency for large datasets

---

## Integration Example

Here's how to use new components together:

```python
from best_library.preprocessing.preprocessing import AugmentedPreprocessing
from best_library.features.feature_engineering import texture_variance
from best_library.model.model_efficientnet import ModelEfficientNet
from best_library.evaluation.evaluate import AdvancedEvaluator

# Setup
preprocessor = AugmentedPreprocessing(img_size=224, augment=True)
transform = preprocessor.get_transform()

model_builder = ModelEfficientNet(device='cuda', class_names=['alpaca', 'not_alpaca'])
model = model_builder.build_model()

# Training
model_builder.train_model(model, train_loader, val_loader, epochs=20)

# Evaluation
evaluator = AdvancedEvaluator(device='cuda')
metrics = evaluator.evaluate_detailed(model, val_loader)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Macro F1: {metrics['macro_f1']:.4f}")
```

---

## Testing New Components

### Location
`tests/`

### Guidelines

1. **Create unit tests** for each new component
2. **Test file naming**: `test_<component_name>.py`
3. **Example test structure**:
   ```python
   import pytest
   import torch
   
   def test_new_feature():
       """Test new feature extraction."""
       image = torch.rand(3, 224, 224)
       feature = new_feature(image)
       
       assert feature.shape[0] > 0
       assert not torch.isnan(feature).any()
   
   def test_new_preprocessor():
       """Test new preprocessor."""
       preprocessor = CustomPreprocessing(img_size=224)
       transform = preprocessor.get_transform()
       
       assert transform is not None
       # Test on sample image
   ```

4. **Run tests**:
   ```bash
   pytest tests/
   ```

---

## Documentation Standards

When adding new components:

1. **Docstrings**: Use Google-style docstrings
2. **Type hints**: Add type annotations to all functions
3. **Examples**: Include usage examples in docstrings
4. **Dependencies**: Update `pyproject.toml` if new packages needed
5. **README**: Update this file with new capabilities

---

## Project Structure

```
src/best_library/
├── preprocessing/       # Data preprocessing & transforms
├── features/           # Feature extraction functions
├── model/              # Model architectures & training
├── evaluation/         # Metrics & evaluation
├── data/               # Data loading utilities
├── split/              # Train/test splitting
└── hyperparameter_tuning/  # Hyperparameter optimization
```

---

## Contributing

When contributing new components:
1. Follow the established patterns and interfaces
2. Write comprehensive tests
3. Document all public APIs
4. Ensure backward compatibility
5. Update this README with usage examples

---

## Installation

```bash
uv pip install -e .
```

## Running the Pipeline

```bash
python run_pipeline.py
```

## Hyperparameter Tuning

```bash
python run_tuning.py
```
