"""Task 2.2 / 2.3 - model definitions.

Every model takes raw pixels (0..255) and does its own scaling as its first layer,
so the saved .keras file is self-contained and serving cannot "forget" a step.
"""
import keras
from keras import layers

# First layer of the last block of each backbone ("unfreeze only the last block")
LAST_BLOCK_START = {"mobilenetv2": "block_16_expand", "efficientnetb0": "block7a_expand_conv"}


def build_cnn(img_size: int, dropout: float = 0.3) -> keras.Model:
    """Small CNN from scratch: 4 conv blocks of increasing width, then a tiny classifier."""
    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1 / 255.0, name="scale_0_1")(inputs)
    for i, filters in enumerate([32, 64, 128, 256], start=1):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.Activation("relu", name=f"relu{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    x = layers.Dense(64, activation="relu", name="dense")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_ai")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


def build_transfer(img_size: int, backbone: str = "mobilenetv2", dropout: float = 0.3,
                   weights="imagenet"):
    """Pretrained backbone (no classifier) + our own head. Returns (model, base)."""
    inputs = keras.Input((img_size, img_size, 3), name="image")
    if backbone == "mobilenetv2":
        # MobileNetV2 was trained on pixels in [-1, 1]; this layer is exactly
        # keras.applications.mobilenet_v2.preprocess_input (x / 127.5 - 1).
        x = layers.Rescaling(1 / 127.5, offset=-1.0, name="mobilenetv2_preprocess")(inputs)
        base = keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3),
                                              include_top=False, weights=weights)
    elif backbone == "efficientnetb0":
        # EfficientNet includes its own rescaling/normalisation and expects 0..255 input.
        x = inputs
        base = keras.applications.EfficientNetB0(input_shape=(img_size, img_size, 3),
                                                 include_top=False, weights=weights)
    else:
        raise ValueError(f"unknown backbone {backbone}")
    base.trainable = False
    # training=False keeps BatchNorm statistics frozen even when we fine-tune later.
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="p_ai")(x)
    model = keras.Model(inputs, outputs, name=f"{backbone}_transfer")
    return model, base


def unfreeze_top(base: keras.Model, backbone: str, n_layers: int = None) -> int:
    """Unfreeze the last block (default) or the last n_layers of the backbone.
    BatchNormalization layers stay frozen: re-estimating their statistics on ~700 images
    would destroy what they learned on ImageNet. Returns the number of unfrozen layers."""
    if n_layers is None:
        names = [l.name for l in base.layers]
        start = names.index(LAST_BLOCK_START[backbone])
    else:
        start = max(0, len(base.layers) - n_layers)
    base.trainable = True
    unfrozen = 0
    for i, layer in enumerate(base.layers):
        layer.trainable = i >= start and not isinstance(layer, layers.BatchNormalization)
        unfrozen += int(layer.trainable and bool(layer.weights))
    return unfrozen


def trainable_report(model: keras.Model) -> str:
    """Text list of every layer that has trainable weights (evidence for Task 2.3)."""
    lines = []
    for layer in model.layers:
        sub = layer.layers if isinstance(layer, keras.Model) else [layer]
        for l in sub:
            if l.trainable_weights:
                n = sum(int(w.numpy().size) for w in l.trainable_weights)
                lines.append(f"{l.name:<35} {n:>10,}")
    total = sum(int(w.numpy().size) for w in model.trainable_weights)
    return "\n".join(lines + [f"{'TOTAL trainable params':<35} {total:>10,}"])


def compile_model(model: keras.Model, lr: float):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
    )
