import torchvision.transforms as transforms


def build_transform(img_size: int = 224):
    """Create image preprocessing transformations."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


class Preprocessing:
    """Class to build and provide image transformations."""

    def __init__(self, img_size: int = 224):
        """Initialize with desired image size."""
        self.img_size = img_size

    def get_transform(self):
        """Return the composed image transform."""
        return build_transform(self.img_size)
