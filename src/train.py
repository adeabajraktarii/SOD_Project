import tensorflow as tf
from src.data_loader import build_dataset
from src.sod_model import build_unet

def iou_metric(y_true, y_pred, smooth=1e-6):
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred_bin)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_bin) - intersection
    return (intersection + smooth) / (union + smooth)


def bce_iou_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    iou = iou_metric(y_true, y_pred)
    return bce + 0.5 * (1 - iou)


def train_model(train_imgs, train_masks, val_imgs, val_masks, save_path):
    train_ds = build_dataset(train_imgs, train_masks, batch_size=16, augment_data=True)
    val_ds   = build_dataset(val_imgs, val_masks, batch_size=16)

    model = build_unet()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=bce_iou_loss,
        metrics=["accuracy", iou_metric]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            save_path, save_best_only=True, monitor="val_loss", mode="min"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=25, callbacks=callbacks)
    return model

