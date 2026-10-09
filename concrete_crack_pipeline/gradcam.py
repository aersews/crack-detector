from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def find_last_conv_layer(model):
    from tensorflow import keras
    for layer in reversed(model.layers):
        if isinstance(layer, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D)):
            return layer
        if hasattr(layer, "layers"):
            for sublayer in reversed(layer.layers):
                if isinstance(sublayer, (keras.layers.Conv2D, keras.layers.DepthwiseConv2D)):
                    return sublayer
    raise ValueError("No convolutional layer found.")


def compute_gradcam(model, image, target_class):
    import tensorflow as tf
    from tensorflow import keras
    target_layer = find_last_conv_layer(model)
    grad_model = keras.Model(model.inputs, [target_layer.output, model.output])
    tensor = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tensor, training=False)
        probability = predictions[:, 0]
        score = probability if target_class == 1 else 1.0 - probability
    gradients = tape.gradient(score, conv_outputs)
    if gradients is None:
        raise RuntimeError("Grad-CAM gradients were not available.")
    pooled = tf.reduce_mean(gradients, axis=(1, 2))[0]
    conv_outputs = conv_outputs[0]
    heatmap = tf.maximum(tf.reduce_sum(conv_outputs * pooled, axis=-1), 0)
    maximum = tf.reduce_max(heatmap)
    heatmap = tf.where(maximum > 0, heatmap / (maximum + 1e-8), heatmap)
    return heatmap.numpy(), float(probability.numpy()[0])


def save_examples(model, model_name, test_df, probabilities, threshold, output_dir, examples_per_case, prefix, logger=print):
    predictions = (probabilities >= threshold).astype(int); cases = defaultdict(list)
    for idx in range(len(test_df)):
        true, pred = int(test_df.iloc[idx].label), int(predictions[idx])
        case = "TP" if true == 1 and pred == 1 else "TN" if true == 0 and pred == 0 else "FP" if true == 0 else "FN"
        if len(cases[case]) < examples_per_case: cases[case].append(idx)
    for case, indices in cases.items():
        for number, idx in enumerate(indices):
            with Image.open(Path(test_df.iloc[idx].path)) as img:
                image = np.asarray(img.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS), dtype=np.float32)
            try:
                heatmap, probability = compute_gradcam(model, image, int(predictions[idx]))
            except Exception as exc:
                logger(f"Grad-CAM failed for {model_name}, {case}: {exc}"); continue
            import tensorflow as tf
            heatmap = tf.image.resize(heatmap[..., None], (224, 224)).numpy().squeeze()
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(image.astype(np.uint8)); axes[0].set_title(f"{case}: Original"); axes[0].axis("off")
            axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title(f"{case}: Grad-CAM"); axes[1].axis("off")
            axes[2].imshow(image.astype(np.uint8)); axes[2].imshow(heatmap, cmap="jet", alpha=0.4); axes[2].set_title(f"{case}\nTrue: {'Crack' if int(test_df.iloc[idx].label) else 'No Crack'}\nPred: {'Crack' if predictions[idx] else 'No Crack'}\nP(Crack): {probability:.3f}"); axes[2].axis("off")
            fig.tight_layout(); fig.savefig(output_dir / f"{prefix}_{case}_{number}.png", dpi=300, bbox_inches="tight"); plt.close(fig)
