import os
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import tensorflow_transform as tft 

from customer_churn_transform import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    transformed_name,
)

def get_model(show_summary=True):
    """Defines a lightweight Keras model."""
    input_features = []
    for key, dim in CATEGORICAL_FEATURES.items():
        input_features.append(
            tf.keras.Input(shape=(dim + 1,), name=transformed_name(key))
        )
    
    for feature in NUMERICAL_FEATURES:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(feature))
        )
    
    concatenate = tf.keras.layers.concatenate(input_features)
    deep = tf.keras.layers.Dense(128, activation="relu")(concatenate)
    deep = tf.keras.layers.Dropout(0.3)(deep)
    deep = tf.keras.layers.Dense(32, activation="relu")(deep)
    deep = tf.keras.layers.Dropout(0.2)(deep)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(deep)

    model = tf.keras.models.Model(inputs=input_features, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )
    
    if show_summary:
        model.summary()
    
    return model

def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern, tf_transform_output, batch_size=32):
    """Generates dataset with smaller batch size."""
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy()
    )

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
    )
    
    return dataset

def run_fn(fn_args):
    """Train the optimized model."""
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    train_dataset = input_fn(fn_args.train_files, tf_transform_output, 32)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, 32)
    
    model = get_model()
    model.fit(
        train_dataset,
        steps_per_epoch=min(fn_args.train_steps, 500),
        validation_data=eval_dataset,
        validation_steps=min(fn_args.eval_steps, 100),
        epochs=5
    )
    
    model.save(fn_args.serving_model_dir, save_format="tf")
