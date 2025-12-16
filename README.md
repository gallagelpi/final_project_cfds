# Final Project (CFDS) — Alpaca vs Not Alpaca

This repository contains:

- **A Python library (`best_library`)** to train and use an image classifier.
- **A FastAPI service (`app/`)** exposing a `/predict` endpoint.

## Library (`best_library`)

### What we changed (refactor summary)

We consolidated the previous model code (previously split across multiple modules/classes) into **one single class**:

- **`ModelResnet18`** in `src/best_library/model/model_resnet18.py`

`ModelResnet18` now contains the full ResNet18 workflow:

- `build_model(...)`: builds a ResNet18 and adapts the final layer to `num_classes`
- `train_model(...)`: trains the model and prints train/val accuracy per epoch
- `load_trained_model(...)`: loads a saved checkpoint and sets `eval()`
- `predict_image(...)`: predicts a single image on disk
- `predict(...)`: alias for `predict_image(...)`

As a result, these files were removed from `src/best_library/model/`:

- `model_definition.py`
- `train.py`
- `predict.py`

### Install / setup

You can install the library in **editable mode** (recommended while developing):

```bash
cd "/Users/felipe/Documents/FINAL PROJECT TEST CFDS/final_project_cfds"
python3 -m pip install -e .
```

If you prefer `uv`:

```bash
cd "/Users/felipe/Documents/FINAL PROJECT TEST CFDS/final_project_cfds"
uv sync
```

### How to run training (Notebook)

The main workflow lives in `notebook/notebook.ipynb`. It:

- Splits the dataset
- Creates transforms (preprocessing)
- Trains a first model
- Runs hyperparameter tuning
- Saves the best model to `models/best_model.pth`
- Evaluates and predicts on a sample image

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
- If you change public APIs/exports, update:
  - `src/best_library/model/__init__.py`
  - `src/best_library/__init__.py`

#### Bump version

Update the version in `pyproject.toml`:

- `[project].version = "0.1.0"` → increment as needed.

#### Quick sanity import (recommended)

```bash
cd "/Users/felipe/Documents/FINAL PROJECT TEST CFDS/final_project_cfds"
PYTHONPATH="/Users/felipe/Documents/FINAL PROJECT TEST CFDS/final_project_cfds/src" python3 -c "from best_library import ModelResnet18; m=ModelResnet18('cpu',['a','b']); m.build_model(); print('ok')"
```

## API (`app/`)

### What it does

The API exposes an endpoint to classify an uploaded image:

- `POST /predict`
- Response: `{ "label": "...", "confidence": 0.0 }`

Internally, the API uses:

- `ModelResnet18` to load `models/best_model.pth`
- the same preprocessing transform used during training (see `app/routers/predict_router.py`)

### Run the API locally

```bash
cd "/Users/felipe/Documents/FINAL PROJECT TEST CFDS/final_project_cfds"
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