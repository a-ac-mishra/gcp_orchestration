import tensorflow as tf
import tensorflow_transform as tft
from tfx_bsl.public import tfxio
from absl import logging
from models import features

HIDDEN_UNITS = [16, 8]
LEARNING_RATE = 0.001
TRAIN_BATCH_SIZE = 40
EVAL_BATCH_SIZE = 40


def _get_tf_examples_serving_signature(model, tf_transform_output):
    model.tft_layer_inference = tf_transform_output.transform_features_layer()

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")]
    )
    def serve_tf_examples_fn(serialized_tf_example):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_feature_spec.pop(features.LABEL_KEY)
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_inference(raw_features)
        outputs = model(transformed_features)
        return {"outputs": outputs}

    return serve_tf_examples_fn


def _get_transform_features_signature(model, tf_transform_output):
    model.tft_layer_eval = tf_transform_output.transform_features_layer()

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")]
    )
    def transform_features_fn(serialized_tf_example):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_eval(raw_features)
        return transformed_features

    return transform_features_fn


def _input_fn(file_pattern, data_accessor, tf_transform_output, batch_size=200):
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=batch_size,
            label_key=features.transformed_name(features.LABEL_KEY),
        ),
        tf_transform_output.transformed_metadata.schema,
    ).repeat()


def build_model(hidden_units, learning_rate):
    real_valued_columns = [
        tf.feature_column.numeric_column(feature, shape=())
        for feature in features.transformed_names(features.DENSE_FLOAT_FEATURES)
    ]
    categorical_columns = [
        tf.feature_column.categorical_column_with_identity(
            feature,
            num_buckets=features.VOCAB_SIZE + features.OOV_SIZE,
            default_value=0,
        )
        for feature in features.transformed_names(features.VOCAB_FEATURES)
    ]
    categorical_columns += [
        tf.feature_column.categorical_column_with_identity(
            feature, num_buckets=num_buckets, default_value=0
        )
        for feature, num_buckets in zip(
            features.transformed_names(features.BUCKET_FEATURES),
            features.BUCKET_FEATURE_BUCKET_COUNT,
        )
    ]
    categorical_columns += [
        tf.feature_column.categorical_column_with_identity(
            feature, num_buckets=num_buckets, default_value=0
        )
        for feature, num_buckets in zip(
            features.transformed_names(features.CATEGORICAL_FEATURES),
            features.CATEGORICAL_FEATURE_MAX_VALUES,
        )
    ]
    indicator_column = [
        tf.feature_column.indicator_column(categorical_column)
        for categorical_column in categorical_columns
    ]
    model = dnn_classifier(
        wide_columns=indicator_column,
        deep_columns=real_valued_columns,
        dnn_hidden_units=hidden_units,
        learning_rate=learning_rate,
    )
    return model


def dnn_classifier(wide_columns, deep_columns, dnn_hidden_units, learning_rate):
    input_layers = {
        colname: tf.keras.layers.Input(name=colname, shape=(), dtype=tf.float32)
        for colname in features.transformed_names(features.DENSE_FLOAT_FEATURES)
    }
    input_layers.update(
        {
            colname: tf.keras.layers.Input(name=colname, shape=(), dtype="int32")
            for colname in features.transformed_names(features.VOCAB_FEATURES)
        }
    )
    input_layers.update(
        {
            colname: tf.keras.layers.Input(name=colname, shape=(), dtype="int32")
            for colname in features.transformed_names(features.BUCKET_FEATURES)
        }
    )
    input_layers.update(
        {
            colname: tf.keras.layers.Input(name=colname, shape=(), dtype="int32")
            for colname in features.transformed_names(features.CATEGORICAL_FEATURES)
        }
    )
    deep = tf.keras.layers.DenseFeatures(deep_columns)(input_layers)
    for numnodes in dnn_hidden_units:
        deep = tf.keras.layers.Dense(numnodes)(deep)
    wide = tf.keras.layers.DenseFeatures(wide_columns)(input_layers)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(
        tf.keras.layers.concatenate([deep, wide])
    )
    output = tf.squeeze(output, -1)
    model = tf.keras.Model(input_layers, output)
    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )
    model.summary(print_fn=logging.info)
    return model


def run_fn(fn_args):

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files,
        fn_args.data_accessor,
        tf_transform_output,
        TRAIN_BATCH_SIZE,
    )
    eval_dataset = _input_fn(
        fn_args.eval_files, fn_args.data_accessor, tf_transform_output, EVAL_BATCH_SIZE
    )

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = build_model(hidden_units=HIDDEN_UNITS, learning_rate=LEARNING_RATE)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq="batch"
    )

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback],
    )

    signatures = {
        "serving_default": _get_tf_examples_serving_signature(
            model, tf_transform_output
        ),
        "transform_features": _get_transform_features_signature(
            model, tf_transform_output
        ),
    }
    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)
