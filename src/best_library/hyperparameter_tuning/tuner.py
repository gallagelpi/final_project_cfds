import torch
from best_library.model.model_definition import build_model
from best_library.model.train import Trainer
from best_library.evaluation.evaluate import evaluate_model
import itertools

# --- Pure functions ---

def generate_param_combinations(param_grid: dict):
    """Generate all combinations of hyperparameters."""
    keys = param_grid.keys()
    values = param_grid.values()
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def train_and_evaluate_model(model, train_loader, val_loader, trainer: Trainer, lr: float, epochs: int):
    """
    Train model (without saving) using Trainer and return validation accuracy.
    """
    trainer.train(model, train_loader, val_loader, epochs, lr)
    val_acc = evaluate_model(model, val_loader, trainer.device)
    return val_acc


def select_best_model(results: list):
    """Select the best hyperparameters and accuracy from results."""
    if not results:
        raise ValueError("No results to select from.")
    best_params, best_acc = max(results, key=lambda x: x[1])
    return best_params, best_acc


def train_final_model(
    best_params: dict,
    train_loader,
    val_loader,
    trainer: Trainer,
    save_path: str
):
    """Train final model with best hyperparameters and save it."""
    num_classes = len(train_loader.dataset.classes)

    model = build_model(num_classes=num_classes).to(trainer.device)

    trainer.train(
        model,
        train_loader,
        val_loader,
        epochs=best_params.get("epochs", 3),
        lr=best_params.get("lr", 1e-4)
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_params": best_params,
            "num_classes": num_classes
        },
        save_path
    )

    print(f"Final model saved to {save_path}")

    return model

# --- Orchestrator class ---

class HyperparameterTuner:
    """
    Class to perform hyperparameter tuning and save the best model at the end.
    """

    def __init__(self, param_grid: dict, device: str = None):
        self.param_grid = param_grid
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.trainer = Trainer(self.device)

    def tune(self, train_loader, val_loader, save_path: str):
        param_combinations = generate_param_combinations(self.param_grid)
        results = []

        num_classes = len(train_loader.dataset.classes)
        print(f"Starting hyperparameter tuning with {len(param_combinations)} combinations...")

        # Train all combinations and record metrics
        for i, params in enumerate(param_combinations):
            print(f"\n--- Trial {i+1}/{len(param_combinations)}: {params} ---")
            model = build_model(num_classes=num_classes).to(self.device)
            val_acc = train_and_evaluate_model(
                model, train_loader, val_loader, self.trainer,
                lr=params.get("lr", 1e-4),
                epochs=params.get("epochs", 3)
            )
            results.append((params, val_acc))
            print(f"Trial finished. Validation accuracy: {val_acc:.3f}")

        # Select best hyperparameters
        best_params, best_acc = select_best_model(results)
        print(f"\nBest combination: {best_params} with accuracy {best_acc:.3f}")

        # Train final model with best hyperparameters and save it
        print("Training final model with best hyperparameters...")
        train_final_model(best_params, train_loader, val_loader, self.trainer, save_path)

        return best_params, best_acc
