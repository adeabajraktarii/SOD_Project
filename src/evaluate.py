import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os


# ========= Metrics ========= #

def iou_metric(y_true, y_pred, smooth=1e-6):
    """
    IoU for binary segmentation.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred_bin)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_bin) - intersection
    return (intersection + smooth) / (union + smooth)


def mae_score(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def precision_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp + 1e-8)


def recall_score(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn + 1e-8)


def f1_score(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-8)


# ========= Evaluation with Threshold ========= #

def evaluate_model(model, dataset, threshold=0.5):
    all_prec, all_rec, all_f1, all_iou, all_mae = [], [], [], [], []
    all_time = []

    for imgs, masks in dataset:
        start = time.time()
        pred = model.predict(imgs)[0]
        elapsed = (time.time() - start) * 1000  # ms
        all_time.append(elapsed)

        y_true = masks[0].numpy()
        y_pred = (pred > threshold).astype("float32")  # APPLY THRESHOLD HERE

        all_prec.append(precision_score(y_true, y_pred))
        all_rec.append(recall_score(y_true, y_pred))
        all_f1.append(f1_score(y_true, y_pred))
        all_iou.append(iou_metric(y_true, y_pred).numpy())
        all_mae.append(mae_score(y_true, y_pred))

    return {
        "precision": round(float(np.mean(all_prec)), 4),
        "recall": round(float(np.mean(all_rec)), 4),
        "f1": round(float(np.mean(all_f1)), 4),
        "iou": round(float(np.mean(all_iou)), 4),
        "mae": round(float(np.mean(all_mae)), 4),
        "avg_inference_time_ms": round(float(np.mean(all_time)), 2)
    }


# ========= Visualization ========= #

def visualize_prediction(model, img, mask, threshold=0.5, save_path=None):
    pred = model.predict(img[None, ...])[0]
    pred_bin = (pred > threshold).astype("float32")  # apply threshold

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.title("Image")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("GT Mask")
    plt.imshow(mask[..., 0], cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title(f"Pred Mask (Thr={threshold})")
    plt.imshow(pred_bin[..., 0], cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Overlay")
    plt.imshow(img)
    plt.imshow(pred_bin[..., 0], cmap="jet", alpha=0.5)
    plt.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to: {save_path}")

    plt.show() 

