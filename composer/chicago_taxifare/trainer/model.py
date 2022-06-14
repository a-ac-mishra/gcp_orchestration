import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import keras

from tensorflow import feature_column as fc
from tensorflow.keras import layers
from tensorflow.keras import models

from google.cloud import bigquery

CSV_COLUMNS = [
    "fare_amount", 
    "dayofweek",
    "hourofday", 
    "pickup_longitude", 
    "pickup_latitude", 
    "is_luxury",
    "distance", 
    "is_airport", 
    "dropoff_longitude", 
    "dropoff_latitude"
]

NUM_COLS = [
    'pickup_longitude',
    'pickup_latitude',
    'dropoff_longitude',
    'dropoff_latitude',
    'distance',
]

CAT_COLS = [
    'hourofday',
    'dayofweek',
    'is_luxury',
    'is_airport'
]

LABEL_COLUMN = 'fare_amount'
DEFAULTS = [[0.0], [0], [0], [0.0], [0.0], [0], [0.0], [0], [0.0], [0.0]]
OPTIMIZER = 'adam'
LOSS = 'mse'
METRICS = ['RootMeanSquaredError']

def features_and_labels(row_data):
    """ generates features and label from tf dataset

    Args:
        row_data: tensor for attributes

    Returns:
        tuple : features and label
    """    
    label = row_data.pop(LABEL_COLUMN)
    features = row_data
    return features, label


def create_dataset(pattern, batch_size=1, mode=tf.estimator.ModeKeys.EVAL):
    """creates tensorflow dataset

    Args:
        pattern : 
        batch_size (int, optional): batch size . Defaults to 1.
        mode (tf.estimator.ModeKeys, optional): Defaults to tf.estimator.ModeKeys.EVAL.

    Returns:
        tensorflow dataset
    """    
    dataset = tf.data.experimental.make_csv_dataset(
        pattern, batch_size, CSV_COLUMNS, DEFAULTS)
    dataset = dataset.map(features_and_labels)
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(buffer_size=1000).repeat(None)
    # take advantage of multi-threading; 1=AUTOTUNE
    dataset = dataset.prefetch(1)
    return dataset


def euclidean(params):
    """calculate euclidean distance

    Args:
        params (tuple): longitude, latitude, longitude, latitude

    Returns:
        float: euclidean distance
    """    
    lon1, lat1, lon2, lat2 = params
    londiff = lon2 - lon1
    latdiff = lat2 - lat1
    return tf.sqrt(londiff*londiff + latdiff*latdiff)


def transform(inputs, num_cols, cat_cols):
    """_summary_

    Args:
        inputs (_type_): _description_
        num_cols (_type_): _description_
        cat_cols (_type_): _description_

    Returns:
        _type_: _description_
    """    
    # Pass-through columns
    transformed = inputs.copy()
    feature_columns = {
        colname: tf.feature_column.numeric_column(colname) for colname in num_cols }
    # Add Euclidean distance
    transformed['euclidean'] = layers.Lambda(
        euclidean, name='euclidean')([inputs['pickup_longitude'],
                           inputs['pickup_latitude'],
                           inputs['dropoff_longitude'],
                           inputs['dropoff_latitude']])
    feature_columns['euclidean'] = fc.numeric_column('euclidean')
    # Shift 'dayofweek' feature to a value range of 0-6
    transformed['dayofweek'] = transformed['dayofweek'] - 1
    # Create categorical columns (wrapped in indicator columns)
    feature_columns['is_luxury'] = fc.indicator_column(
        fc.categorical_column_with_identity('is_luxury', 2))
    feature_columns['is_airport'] = fc.indicator_column(
        fc.categorical_column_with_identity('is_airport', 2))
    feature_columns['hourofday'] = fc.indicator_column(
        fc.categorical_column_with_identity('hourofday', 24))
    feature_columns['dayofweek'] = fc.indicator_column(
        fc.categorical_column_with_identity('dayofweek', 7))
    return transformed, feature_columns


def build_dnn_model():
    """Building deep neural network using keras

    Returns:
       tf.Model: Compiled DNN model
    """    
    # define the imput layers
    inputs = {
        colname: layers.Input(name=colname, shape=(), dtype='float32')
        for colname in NUM_COLS
    }
    inputs.update({
        colname: layers.Input(name=colname, shape=(), dtype='int32')
        for colname in CAT_COLS
    })
    # transforms the features, here euclidean layer
    transformed, feature_columns = transform(inputs, num_cols=NUM_COLS, cat_cols=CAT_COLS)
    dnn_inputs = layers.DenseFeatures(feature_columns.values())(transformed)
    # adding two hidden layers
    h1 = layers.Dense(32, activation='relu', name='hidden_layer_1')(dnn_inputs)
    h2 = layers.Dense(8, activation='relu', name='hidden_layer_2')(h1)
    # final output is a linear activation because this is regression
    output = layers.Dense(1, activation='linear', name='fare')(h2)
    model = models.Model(inputs, output)
    # Compile model
    model.compile(optimizer=OPTIMIZER, loss=LOSS, metrics=METRICS)
    return model


def train_and_evaluate(hparams):
    """training and evaluation on DNN model

    Args:
        hparams (dict): 
            train_epochs: training epochs
            log_dir: logs directory
            output_dir: output artifacts directory
            train_data_path: training data path
            eval_data_path: evaluation data path
            output_ds: model artifacts
            version_name: version name provided by user


    Returns:
        tf model training history
    """    
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    BATCH_SIZE_PER_REPLICA = 256
    TRAIN_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    NUM_TRAIN_EXAMPLES = 4 * 52000 * hparams['train_epochs']
    NUM_EVALS = hparams['train_epochs']
    NUM_EVAL_EXAMPLES = 52000
    OUTDIR = hparams['output_dir']
    LOGDIR = hparams['log_dir']

    print("building dnn model...")
    with strategy.scope():
        model = build_dnn_model()

    print("training data is getting created...")
    trainds = create_dataset(hparams['train_data_path'], TRAIN_BATCH_SIZE, tf.estimator.ModeKeys.TRAIN)

    print("evaluation data is getting created...")
    evalds = create_dataset(hparams['eval_data_path'], 1000, tf.estimator.ModeKeys.EVAL)

    steps_per_epoch = NUM_TRAIN_EXAMPLES // (TRAIN_BATCH_SIZE * NUM_EVALS)
    validation_steps = NUM_EVAL_EXAMPLES // BATCH_SIZE_PER_REPLICA

    callbacks = [tf.keras.callbacks.ModelCheckpoint(filepath=OUTDIR),
                 tf.keras.callbacks.TensorBoard(log_dir=LOGDIR)]

    history = model.fit(trainds,
                        verbose=2,
                        validation_data=evalds,
                        validation_steps=validation_steps,
                        epochs=NUM_EVALS,
                        steps_per_epoch=steps_per_epoch,
                        callbacks=callbacks)

    tf.saved_model.save(model, hparams['output_dir'])

    val_metric = history.history['val_RootMeanSquaredError'][NUM_EVALS-1]

    client = bigquery.Client()

    sql = """ INSERT `{0}.model_metrics`
              VALUES ('{1}',{2});""".format(hparams[''], hparams['version_name'], val_metric)

    query_job = client.query(sql)
    print(query_job.done())

    return history
