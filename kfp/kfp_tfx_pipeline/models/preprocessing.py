import tensorflow as tf
import tensorflow_transform as tft
from models import features


def fill_missing_value(x):
    if not isinstance(x, tf.sparse.SparseTensor):
        return x

    default_value = "" if x.dtype == tf.string else 0
    return tf.squeeze(
        tf.sparse.to_dense(
            tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]), default_value
        ),
        axis=1,
    )


def preprocessing_fn(inputs):
    outputs = {}
    for feature_name in features.DENSE_FLOAT_FEATURES:
        outputs[features.transformed_name(feature_name)] = tft.scale_to_z_score(
            fill_missing_value(inputs[feature_name])
        )

    for feature_name in features.VOCAB_FEATURES:
        outputs[
            features.transformed_name(feature_name)
        ] = tft.compute_and_apply_vocabulary(
            fill_missing_value(inputs[feature_name]),
            top_k=features.VOCAB_SIZE,
            num_oov_buckets=features.OOV_SIZE,
        )

    for feature_name, num_buckets in zip(
        features.BUCKET_FEATURES, features.BUCKET_FEATURE_BUCKET_COUNT
    ):
        outputs[features.transformed_name(feature_name)] = tft.bucketize(
            fill_missing_value(inputs[feature_name]), num_buckets
        )

    for feature_name in features.CATEGORICAL_FEATURES:
        outputs[features.transformed_name(feature_name)] = fill_missing_value(
            inputs[feature_name]
        )

    taxi_fare = fill_missing_value(inputs[features.FARE_KEY])

    tips = fill_missing_value(inputs[features.LABEL_KEY])

    outputs[features.transformed_name(features.LABEL_KEY)] = tf.where(
        tf.math.is_nan(taxi_fare),
        tf.cast(tf.zeros_like(taxi_fare), tf.int64),
        tf.cast(tf.greater(tips, tf.multiply(taxi_fare, tf.constant(0.2))), tf.int64),
    )
    return outputs
