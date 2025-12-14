from src.best_library.hyperparameter_tuning.tuner import HyperparameterTuner

# CONFIG
WORK_DIR = "data" # Assumes data is already split in 'data/train' and 'data/val'

def main():
    # Define the grid of hyperparameters to search
    param_grid = {
        'lr': [1e-3, 1e-4, 1e-5],
        'batch_size': [8, 16],
        'epochs': [3] # Keep epochs low for speed during demonstration
    }
    
    print("Initializing Tuner...")
    tuner = HyperparameterTuner(WORK_DIR, param_grid)
    
    best_params, best_acc = tuner.tune()
    
    print(f"Optimization finished! You should train your final model with: {best_params}")

if __name__ == "__main__":
    main()
