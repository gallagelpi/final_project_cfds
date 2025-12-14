import unittest
import os
import shutil
import torch
from PIL import Image
from src.best_library.features.feature_engineering import compute_dataset_stats

class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        """Create a temporary dataset for testing."""
        self.test_dir = "test_dataset_temp"
        self.class_dir = os.path.join(self.test_dir, "class_a")
        os.makedirs(self.class_dir, exist_ok=True)
        
        # Create a few dummy images
        for i in range(5):
            img = Image.new('RGB', (100, 100), color=(i*50, 100, 150))
            img.save(os.path.join(self.class_dir, f"img_{i}.jpg"))

    def tearDown(self):
        """Clean up the temporary dataset."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_compute_dataset_stats(self):
        """Test if mean and std are computed correctly."""
        mean, std = compute_dataset_stats(self.test_dir, img_size=64, batch_size=2)
        
        # Check if they are not None
        self.assertIsNotNone(mean)
        self.assertIsNotNone(std)
        
        # Check shapes (should be 3 channels)
        self.assertEqual(mean.shape, (3,))
        self.assertEqual(std.shape, (3,))
        
        # Check if values are within reasonable range (0 to 1 for mean of tensors usually, but normalized? 
        # Wait, compute_dataset_stats uses ToTensor(), so values are [0,1].
        self.assertTrue(torch.all(mean >= 0.0) and torch.all(mean <= 1.0))
        self.assertTrue(torch.all(std >= 0.0))

    def test_empty_directory(self):
        """Test behavior with an empty directory."""
        empty_dir = "empty_temp_dir"
        os.makedirs(empty_dir, exist_ok=True)
        try:
            # Should handle gracefully or return None/Error
            # The current implementation prints "No images found" and returns None, None
            mean, std = compute_dataset_stats(empty_dir, img_size=64)
            self.assertIsNone(mean)
            self.assertIsNone(std)
        finally:
            shutil.rmtree(empty_dir)

if __name__ == '__main__':
    unittest.main()
