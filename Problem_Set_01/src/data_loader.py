import tensorflow as tf

IMG_SIZE = (150, 150)
BATCH_SIZE = 32
SEED = 42

def get_datasets(data_dir="data/chest_xray"):
    # Val split provided by the dataset is too small (16 images) to be reliable,
    # so we carve our own validation set out of the training data instead.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/train",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        color_mode="grayscale",
        validation_split=0.2,
        subset="training",
        seed=SEED
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/train",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        color_mode="grayscale",
        validation_split=0.2,
        subset="validation",
        seed=SEED
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        f"{data_dir}/test",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        color_mode="grayscale",
        shuffle=False  # preserve order so predictions line up with true labels later
    )

    class_names = train_ds.class_names  # e.g. ['NORMAL', 'PNEUMONIA']

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
    ])
    normalization = tf.keras.layers.Rescaling(1.0 / 255)

    train_ds = train_ds.map(lambda x, y: (augmentation(x, training=True), y))
    train_ds = train_ds.map(lambda x, y: (normalization(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization(x), y))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names