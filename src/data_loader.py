import tensorflow as tf
import os

IMG_SIZE = 128

def load_image_mask(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE))

    mask = tf.cast(mask > 127, tf.float32)
    return img, mask


def augment(img, mask):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    return img, mask


def build_dataset(images_dir, masks_dir, batch_size=16, augment_data=False):
    image_paths = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir)])
    mask_paths  = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir)])

    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    ds = ds.map(load_image_mask, num_parallel_calls=tf.data.AUTOTUNE)

    if augment_data:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

