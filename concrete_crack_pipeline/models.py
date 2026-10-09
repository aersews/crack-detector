from __future__ import annotations

from typing import Callable


def build_models(img_size: tuple[int, int]) -> dict[str, Callable]:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import ResNet50, MobileNetV2, MobileNetV3Small, EfficientNetB0, DenseNet121
    from tensorflow.keras.regularizers import l2

    def transfer(backbone_class, name, preprocessing, dropout, l2_reg):
        inputs = layers.Input(shape=(*img_size, 3), name="image_input")
        augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(20 / 360.0),
            layers.RandomTranslation(0.15, 0.15),
            layers.RandomZoom(0.15),
        ], name="augmentation")
        x = augmentation(inputs)
        x = preprocessing(x)
        backbone = backbone_class(weights="imagenet", include_top=False, input_shape=(*img_size, 3), name="backbone")
        backbone.trainable = False
        x = backbone(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation="relu", kernel_regularizer=l2(l2_reg))(x)
        x = layers.Dropout(dropout)(x)
        output = layers.Dense(1, activation="sigmoid", dtype="float32", name="probability")(x)
        return models.Model(inputs, output, name=name)

    def tinycnn(dropout=0.4, l2_reg=1e-4):
        inputs = layers.Input(shape=(*img_size, 3), name="image_input")
        augmentation = keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(20 / 360.0),
            layers.RandomTranslation(0.15, 0.15),
            layers.RandomZoom(0.15),
        ], name="augmentation")
        x = augmentation(inputs)
        for filters in (32, 64, 128):
            x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(64, activation="relu", kernel_regularizer=l2(l2_reg))(x)
        x = layers.Dropout(dropout)(x)
        return models.Model(inputs, layers.Dense(1, activation="sigmoid", dtype="float32", name="probability")(x), name="TinyCNN")

    def resnet_preprocess(x): return tf.keras.applications.resnet50.preprocess_input(x)
    def mobilenetv2_preprocess(x): return tf.keras.applications.mobilenet_v2.preprocess_input(x)
    def mobilenetv3_preprocess(x): return x
    def efficientnet_preprocess(x): return x
    def densenet_preprocess(x): return tf.keras.applications.densenet.preprocess_input(x)

    return {
        "ResNet50": lambda dropout=0.4, l2_reg=1e-4: transfer(ResNet50, "ResNet50", resnet_preprocess, dropout, l2_reg),
        "MobileNetV2": lambda dropout=0.4, l2_reg=1e-4: transfer(MobileNetV2, "MobileNetV2", mobilenetv2_preprocess, dropout, l2_reg),
        "MobileNetV3Small": lambda dropout=0.4, l2_reg=1e-4: transfer(MobileNetV3Small, "MobileNetV3Small", mobilenetv3_preprocess, dropout, l2_reg),
        "EfficientNetB0": lambda dropout=0.4, l2_reg=1e-4: transfer(EfficientNetB0, "EfficientNetB0", efficientnet_preprocess, dropout, l2_reg),
        "DenseNet121": lambda dropout=0.4, l2_reg=1e-4: transfer(DenseNet121, "DenseNet121", densenet_preprocess, dropout, l2_reg),
        "TinyCNN": tinycnn,
    }


def compile_model(model, learning_rate: float) -> None:
    from tensorflow import keras
    from tensorflow.keras import optimizers
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"), keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall"), keras.metrics.AUC(name="auc")],
    )
