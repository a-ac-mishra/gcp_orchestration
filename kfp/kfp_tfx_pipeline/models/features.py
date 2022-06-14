DENSE_FLOAT_FEATURES = ["trip_miles", "fare", "trip_seconds"]

CATEGORICAL_FEATURES = ["trip_start_hour", "trip_start_day", "trip_start_month"]
CATEGORICAL_FEATURE_MAX_VALUES = [24, 31, 12]

BUCKET_FEATURES = [
    "pickup_latitude",
    "pickup_longitude",
    "dropoff_latitude",
    "dropoff_longitude",
]
BUCKET_FEATURE_BUCKET_COUNT = [10, 10, 10, 10]

VOCAB_FEATURES = ["payment_type", "company"]
VOCAB_SIZE = 1000
OOV_SIZE = 10


def transformed_name(feature_name):
    """changes the name of transform

  Args:
      feature_name (str): name of the feature

  Returns:
      str: renamed feature name
  """
    return feature_name + "_xf"


def vocabulary_name(feature_name):
    """ changes vocabulary name

  Args:
      feature_name (str): change feature name

  Returns:
      str:renamed feature name 
  """
    return feature_name + "_vocab"


def transformed_names(feature_names):
    """changes feature name

  Args:
      feature_names (str): change feature name

  Returns:
      str: renamed feature name
  """
    return [transformed_name(feature_name) for feature_name in feature_names]
