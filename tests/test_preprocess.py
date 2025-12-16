import unittest
import torch
from PIL import Image
from src.best_library.preprocessing.preprocessing import Preprocessing

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.img_size = 224
        self.preprocessing = Preprocessing(img_size=self.img_size)

    def test_transform_creation(self):
        """Test if the transform is created successfully."""
        transform = self.preprocessing.get_transform()
        self.assertIsNotNone(transform)

    def test_transform_output_shape(self):
        """Test if the transform produces a tensor of the correct shape."""
        # Create a dummy RGB image
        dummy_img = Image.new('RGB', (500, 500), color='red')
        
        transform = self.preprocessing.get_transform()
        output_tensor = transform(dummy_img)
        
        # Check type
        self.assertIsInstance(output_tensor, torch.Tensor)
        
        # Check shape: (Channels, Height, Width)
        expected_shape = (3, self.img_size, self.img_size)
        self.assertEqual(output_tensor.shape, expected_shape)

    def test_normalization(self):
        """Test if normalization is applied (roughly)."""
        # A pure white image [255, 255, 255] should not be all 1.0s after normalization
        dummy_img = Image.new('RGB', (100, 100), color='white')
        transform = self.preprocessing.get_transform()
        output_tensor = transform(dummy_img)
        
        # If it was just ToTensor, max would be 1.0. With Normalize, it should be different.
        # Mean=[0.485...], Std=[0.229...] -> (1 - 0.485)/0.229 ≈ 2.25
        self.assertTrue(torch.max(output_tensor) > 1.0)

if __name__ == '__main__':
    unittest.main()
