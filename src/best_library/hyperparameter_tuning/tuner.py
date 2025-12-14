import itertools
import torch
from src.best_library.model.model_definition import build_model
from src.best_library.model.train import train_model
from src.best_library.data.load_data import load_data
from src.best_library.preprocessing.preprocessing import Preprocessing

class HyperparameterTuner:
    def __init__(self, work_dir, param_grid, img_size=224):
        """
        Args:
            work_dir (str): Path to the split dataset.
            param_grid (dict): Dictionary where keys are parameter names and values are lists of values to try.
                               Example: {'lr': [1e-3, 1e-4], 'batch_size': [16, 32]}
            img_size (int): Image size for preprocessing.
        """
        self.work_dir = work_dir
        self.param_grid = param_grid
        self.img_size = img_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def tune(self):
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        combinations = list(itertools.product(*values))
        
        best_acc = 0.0
        best_params = None
        
        print(f"Starting Grid Search with {len(combinations)} combinations...")
        
        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            print(f"\n--- Trial {i+1}/{len(combinations)}: {params} ---")
            
            # Extract params
            lr = params.get('lr', 1e-4)
            batch_size = params.get('batch_size', 16)
            epochs = params.get('epochs', 3) # Default small for tuning
            
            # Setup Data
            preprocessing = Preprocessing(img_size=self.img_size)
            transform = preprocessing.get_transform()
            
            try:
                train_loader, val_loader, class_names = load_data(self.work_dir, batch_size, transform)
            except FileNotFoundError:
                print("Data not found. Skipping.")
                continue
                
            # Build Model
            model = build_model(self.device, num_classes=len(class_names))
            
            # Train
            # We don't save every model, only the best one conceptually, but here we just want the score
            val_acc = train_model(model, train_loader, val_loader, epochs, lr, self.device, save_path=None)
            
            if val_acc > best_acc:
                best_acc = val_acc
                best_params = params
                print(f"New Best Accuracy: {best_acc:.3f} with params {best_params}")
                
        print("\n============================================")
        print(f"Tuning Complete.")
        print(f"Best Accuracy: {best_acc:.3f}")
        print(f"Best Parameters: {best_params}")
        print("============================================")
        
        return best_params, best_acc
