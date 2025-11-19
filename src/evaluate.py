import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred_bin)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_bin) - intersection
    return (intersection + smooth) / (union + smooth)


def precision_score(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(np.uint8)
    y_true = (y_true > 0.5).astype(np.uint8)
    tp = np.sum(y_true * y_pred)
    fp = np.sum((1 - y_true) * y_pred)
    return tp / (tp + fp + 1e-8)


def recall_score(y_true, y_pred):
    y_pred = (y_pred > 0.5).astype(np.uint8)
    y_true = (y_true > 0.5).astype(np.uint8)
    tp = np.sum(y_true * y_pred)
    fn = np.sum(y_true * (1 - y_pred))
    return tp / (tp + fn + 1e-8)


def f1_score(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return 2 * p * r / (p + r + 1e-8)


def evaluate_model(model, dataset):
    all_prec, all_rec, all_f1, all_iou = [], [], [], []

    for imgs, masks in dataset:
        pred = model.predict(imgs)[0]

        y_true = masks[0].numpy()
        y_pred = pred

        all_prec.append(precision_score(y_true, y_pred))
        all_rec.append(recall_score(y_true, y_pred))
        all_f1.append(f1_score(y_true, y_pred))
        all_iou.append(iou_metric(y_true, y_pred).numpy())

    return {
        "precision": float(np.mean(all_prec)),
        "recall": float(np.mean(all_rec)),
        "f1": float(np.mean(all_f1)),
        "iou": float(np.mean(all_iou))
    }


def visualize_prediction(model, img, mask):
    pred = model.predict(img[None, ...])[0]
    pred_bin = (pred > 0.5).astype("float32")

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
    plt.title("Pred Mask")
    plt.imshow(pred[..., 0], cmap="gray")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Overlay")
    plt.imshow(img)
    plt.imshow(pred_bin[..., 0], cmap="jet", alpha=0.5)
    plt.axis("off")

    plt.show()