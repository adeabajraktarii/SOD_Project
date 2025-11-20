import tensorflow as tf
import os

IMG_SIZE = 224  # Updated resolution

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
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)

    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)

    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)

    if tf.random.uniform(()) > 0.3:
        crop_size = tf.random.uniform([], 180, 224, dtype=tf.int32)
        img = tf.image.resize_with_crop_or_pad(img, crop_size, crop_size)
        mask = tf.image.resize_with_crop_or_pad(mask, crop_size, crop_size)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE))

    return img, mask


def build_dataset(images_dir, masks_dir, batch_size=8, augment_data=False):
    image_paths = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir)])
    mask_paths  = sorted([os.path.join(masks_dir, f) for f in os.listdir(masks_dir)])

    ds = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    ds = ds.map(load_image_mask, num_parallel_calls=tf.data.AUTOTUNE)

    if augment_data:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(500)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
