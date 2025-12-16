import torch

from best_library.features.feature_engineering import color_features, grayscale_histogram, edge_density, image_entropy, frequency_features, FeatureBuilder



# ---------- Fixtures simples ----------

def dummy_image():
    torch.manual_seed(42)
    return torch.rand(3, 64, 64)


# ---------- Individual feature tests ----------

def test_color_features_shape():
    image = dummy_image()
    features = color_features(image)
    assert features.shape[0] == 6  # mean(3) + std(3)


def test_color_features_no_nan():
    image = dummy_image()
    features = color_features(image)
    assert not torch.isnan(features).any()


def test_grayscale_histogram_shape():
    image = dummy_image()
    hist = grayscale_histogram(image, bins=16)
    assert hist.shape[0] == 16


def test_grayscale_histogram_sum_positive():
    image = dummy_image()
    hist = grayscale_histogram(image, bins=16)
    assert hist.sum() > 0


def test_edge_density_scalar():
    image = dummy_image()
    edge = edge_density(image)
    assert edge.shape == torch.Size([1])


def test_edge_density_positive():
    image = dummy_image()
    edge = edge_density(image)
    assert edge.item() >= 0.0


def test_image_entropy_shape():
    image = dummy_image()
    ent = image_entropy(image)
    assert ent.shape == torch.Size([1])


def test_image_entropy_positive():
    image = dummy_image()
    ent = image_entropy(image)
    assert ent.item() >= 0.0


def test_frequency_features_shape():
    image = dummy_image()
    freq = frequency_features(image)
    assert freq.shape[0] == 2


def test_frequency_features_no_nan():
    image = dummy_image()
    freq = frequency_features(image)
    assert not torch.isnan(freq).any()


# ---------- FeatureBuilder tests ----------

def test_feature_builder_output_shape():
    image = dummy_image()
    builder = FeatureBuilder()
    features = builder.extract_features(image)

    expected_dim = (
        6   # color_features
        + 16  # grayscale_histogram
        + 1   # edge_density
        + 1   # image_entropy
        + 2   # frequency_features
    )

    assert features.shape[0] == expected_dim


def test_feature_builder_no_nan():
    image = dummy_image()
    builder = FeatureBuilder()
    features = builder.extract_features(image)
    assert not torch.isnan(features).any()
