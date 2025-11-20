import tensorflow as tf
import os
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
    return 0.7 * bce + 0.3 * (1 - iou)  # Updated weighting

def train_model(train_imgs, train_masks, val_imgs, val_masks, save_path, resume=False):
    train_ds = build_dataset(train_imgs, train_masks, batch_size=8, augment_data=True)
    val_ds   = build_dataset(val_imgs, val_masks, batch_size=8)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=2000,
        decay_rate=0.9
    )

    if resume and os.path.exists(save_path):
        print("Resuming training from checkpoint...")
        model = tf.keras.models.load_model(save_path,
                                           custom_objects={'iou_metric': iou_metric,
                                                           'bce_iou_loss': bce_iou_loss})
    else:
        model = build_unet(input_shape=(224, 224, 3), base_filters=32, dropout_rate=0.4)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        loss=bce_iou_loss,
        metrics=["accuracy", iou_metric]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=save_path,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(log_dir="/content/drive/MyDrive/SOD_Project/logs")
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=callbacks)
    return model, history


