# Final Project (CFDS) — Alpaca vs Not Alpaca

This repository contains:

- **A Python library (`best_library`)** to train and use an image classifier.
- **A FastAPI service (`app/`)** exposing a `/predict` endpoint.

## Library (`best_library`)

### How the library works (high level)

The library implements an end-to-end image classification workflow:

- **Preprocessing**: defines the image transforms used for both training and inference.
- **Dataset splitting**: creates `train/` and `val/` folders from a raw dataset.
- **Data loading**: builds PyTorch `DataLoader`s for `train` and `val`.
- **Model**: builds a ResNet18, trains it, saves/loads checkpoints, and predicts on images.
- **Tuning & evaluation**: optional hyperparameter search + validation evaluation.

The main runnable example is the notebook at `notebook/notebook.ipynb`.

### Library components (classes) and what they do

#### `ModelResnet18`

Location: `src/best_library/model/model_resnet18.py`

This is the model interface you use from notebooks/apps. It contains the full ResNet18 workflow:

- **`build_model(num_classes=None, weights="IMAGENET1K_V1")`**
  - Builds a torchvision ResNet18.
  - Replaces the final `fc` layer to match `num_classes`.
  - Moves the model to `self.device`.
- **`train_model(model, train_loader, val_loader, epochs, lr)`**
  - Trains the model using cross-entropy + Adam.
  - Prints train/val accuracy per epoch.
  - Returns the best validation accuracy.
- **`load_trained_model(model_path, num_classes=None)`**
  - Loads a checkpoint (expects `model_state_dict`, and optionally `num_classes`).
  - Rebuilds the model and loads weights.
  - Sets `eval()` and returns the ready-to-use model.
- **`predict_image(image_path, model, transform)`**
  - Loads an image from disk, applies `transform`, runs the model in `torch.no_grad()`.
  - Returns `(label, confidence)` using `self.class_names`.
- **`predict(...)`**
  - Alias for `predict_image(...)`.

#### `Preprocessing`

Location: `src/best_library/preprocessing/preprocessing.py`

- Provides `get_transform()` to build the transform pipeline (resize/normalize/etc).
- The same transform should be used in training and inference for consistent results.

#### `DatasetSplitter`

Location: `src/best_library/split/split_train_test.py`

- Splits a raw dataset directory into a working directory with:
  - `WORK_DIR/train/<class_name>/...`
  - `WORK_DIR/val/<class_name>/...`

#### `LoadData`

Location: `src/best_library/data/load_data.py`

- Builds PyTorch datasets and returns:
  - `train_loader`, `val_loader`, `class_names`

#### `FeatureBuilder` (optional)

Location: `src/best_library/features/feature_engineering.py`

- Provides feature utilities (used in the notebook as an example).

#### `HyperparameterTuner` (optional)

Location: `src/best_library/hyperparameter_tuning/tuner.py`

- Runs a simple grid search over `{lr, epochs}`.
- Trains each candidate and tracks validation accuracy.
- Saves the final (best) model checkpoint to the given `save_path`.

#### `Evaluator`

Location: `src/best_library/evaluation/evaluate.py`

- Evaluates a trained model on a validation loader and returns accuracy.

### Install / setup

You can install the library in **editable mode** (recommended while developing):

```bash
cd final_project_cfds
python3 -m pip install -e .
```

If you prefer `uv`:

```bash
cd final_project_cfds
uv sync
```

### How to run (Notebook)

The main workflow lives in `notebook/notebook.ipynb`. It:

- Splits the dataset
- Creates transforms (preprocessing)
- Trains a first model
- Runs hyperparameter tuning
- Saves the best model to `models/best_model.pth`
- Evaluates and predicts on a sample image

To run it locally:

```bash
cd final_project_cfds
python3 -m ipykernel install --user --name final-project-cfds --display-name "final-project-cfds"
python3 -m jupyter notebook
```

### Using the library in code

Minimal usage pattern:

```python
from best_library import ModelResnet18

model_api = ModelResnet18(device=DEVICE, class_names=class_names)
model = model_api.build_model()
model_api.train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR)

best_model = model_api.load_trained_model("models/best_model.pth")
label, conf = model_api.predict("dog2.jpg", best_model, transform)
```

### How to update/change the library

- **Edit source code** in `src/best_library/...`
- Keep the package installed in editable mode (`pip install -e .`) so changes are picked up immediately.
- If you add a new public class/function, update exports:
  - `src/best_library/model/__init__.py` (model-related)
  - `src/best_library/__init__.py` (top-level convenience exports)

#### Bump version

Update the version in `pyproject.toml`:

- `[project].version = "0.1.0"` → increment as needed.

#### Quick sanity import (recommended)

```bash
cd final_project_cfds
PYTHONPATH="./src" python3 -c "from best_library import ModelResnet18; m=ModelResnet18('cpu',['a','b']); m.build_model(); print('ok')"
```

## API (`app/`)

### How the API works

The API exposes an endpoint to classify an uploaded image:

- `POST /predict`
- Response: `{ "label": "...", "confidence": 0.0 }`

Internally, the request flow is:

- `app/routers/predict_router.py` receives the uploaded image file.
- It applies the same transform as training (via `Preprocessing`).
- `app/services/predict_service.py`:
  - temporarily saves the uploaded file
  - calls `ModelResnet18.predict(...)`
  - returns `(label, confidence)`

The API loads the trained checkpoint from `models/best_model.pth` at startup.

### Run the API locally

```bash
cd final_project_cfds
uvicorn app.app:app --reload --host 0.0.0.0 --port 8000
```

### Call the API

Example with `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@dog2.jpg"
```

### Notes / troubleshooting

- The API expects a trained model at `models/best_model.pth`. If you don’t have it yet, run the notebook to generate it.
- If you change the class order (`class_names`), you must retrain and save a new checkpoint (or ensure the API uses the same order).