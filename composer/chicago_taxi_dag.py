""""DAG definition for Chicago Taxifare pipeline.
    This pipeline was created for use as a demo in the Data Engineering
    on GCP Course"""

import datetime
from base64 import b64encode as b64e

from airflow import DAG
from airflow.models import Variable
from airflow.contrib.operators.bigquery_check_operator import BigQueryCheckOperator
from airflow.contrib.operators.bigquery_operator import BigQueryOperator
from airflow.contrib.operators.bigquery_to_gcs import BigQueryToCloudStorageOperator
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.contrib.operators.mlengine_operator import MLEngineTrainingOperator
from airflow.contrib.operators.mlengine_operator import MLEngineModelOperator
from airflow.contrib.operators.mlengine_operator import MLEngineVersionOperator


# GCS bucket names and region, can also be changed.
PROJECT_ID = "<your_project_id>"
if PROJECT_ID == "<your_project_id>":
    raise "Please provide PROJECT_ID"
BUCKET = "gs://" + PROJECT_ID
REGION = "us-central1"

# Specify your source BigQuery project, dataset and table names
SOURCE_BQ_PROJECT = "bigquery-public-data"
SOURCE_DATASET_TABLE_NAMES = "chicago_taxi_trips"
DESTINATION_DATASET = "chicago_taxi_fare"

# Define runtime environment versions
TF_RUNTIME_VERSION = "2.1"
PYTHON_VERSION = "3.7"

# directory of the solution code base.
PACKAGE_URI = BUCKET + "/chicago_taxi/code/trainer.tar"
JOB_DIR = BUCKET + "/jobs"

# Define Machine Learning Model name and location to store artifacts
MODEL = "taxi_trips"
MODEL_LOCATION = BUCKET + "/chicago_taxi/saved_model/"

# Define the default arguments required to instantiate the DAG
DEFAULT_ARGS = {
    "owner": "Owner-Name",
    "depends_on_past": False,
    "start_date": datetime.datetime(2022, 3, 1),
    "email": ["email-address@your-domain.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": datetime.timedelta(minutes=2),
}

# Instaniate the DAG Object
# NOTE: schedule_interval is None, trigger DAG manually from Airflow UI.
#       else provide interval.
with DAG(
    "taxi_trips_dag", catchup=False, default_args=DEFAULT_ARGS, schedule_interval=None
) as dag:

    # SQL Query to check for fresh data. "fresh" data is
    # ingested within the past 90 days.
    verify_sql = """
            SELECT
                COUNT(*)
            FROM
                `bigquery-public-data.chicago_taxi_trips.taxi_trips`
            WHERE
                trip_start_timestamp >= TIMESTAMP('{{ macros.ds_add(ds, -90) }}')
        """

    # BigQueryCheckOperator will fail if the result of the query is 0.
    # I.e. if there is no fresh data.
    bq_check_data_op = BigQueryCheckOperator(
        task_id="bq_check_for_latest_data", use_legacy_sql=False, sql=verify_sql,
    )

    # Base query to extract training and validation datasets from Public BigQuery dataset.
    bql = """
            SELECT
                pickup_longitude,
                pickup_latitude ,
                dropoff_longitude,
                dropoff_latitude,
                (IFNULL(tolls,0) + fare) AS fare_amount,
                EXTRACT(DAYOFWEEK FROM trip_start_timestamp) AS dayofweek,
                EXTRACT(HOUR FROM trip_start_timestamp) AS hourofday,
                ST_DISTANCE(ST_GEOGPOINT(pickup_longitude, pickup_latitude), 
                ST_GEOGPOINT(dropoff_longitude, dropoff_latitude)) AS distance,
                IF(company IN ("Blue Ribbon Taxi Association Inc.", "Suburban Dispatch LLC"),1,0) as is_luxury,
                CASE  
                    WHEN (pickup_community_area IN (56, 64, 76)) OR (dropoff_community_area IN (56, 64, 76)) 
                    THEN 1 else 0 
                END AS is_airport,
                unique_key
            FROM
                `bigquery-public-data.chicago_taxi_trips.taxi_trips`
            WHERE
                trip_miles > 0
                AND fare BETWEEN 5 and 599
                AND trip_seconds > 0 
                AND pickup_longitude IS NOT NULL
                AND pickup_latitude IS NOT NULL
                AND dropoff_longitude IS NOT NULL
                AND dropoff_latitude IS NOT NULL
                """

    # Bigquery query to take a 4/2500 sample of the results of the base query for
    # the training dataset.
    bql_train = """
                SELECT fare_amount, dayofweek, hourofday, pickup_longitude , pickup_latitude, is_luxury, 
                    distance, is_airport, dropoff_longitude , dropoff_latitude FROM({0}) WHERE
                MOD(ABS(FARM_FINGERPRINT(unique_key)), 2500) >= 1
                AND MOD(ABS(FARM_FINGERPRINT(unique_key)), 2500) <= 4
                """.format(
        bql
    )

    # operation to prepare training data
    bq_train_data_op = BigQueryOperator(
        task_id="bq_training_data_prep",
        bql=bql_train,
        destination_dataset_table="{}.{}_training_data".format(
            DESTINATION_DATASET, MODEL
        ),
        write_disposition="WRITE_TRUNCATE",  # specify to truncate on writes
        use_legacy_sql=False,
        dag=dag,
    )

    # bigquery query to take a 1/2500 sample of the results of the base query for
    # the vaidation dataset.
    bql_valid = """
                SELECT fare_amount, dayofweek, hourofday, pickup_longitude , pickup_latitude, is_luxury,
                     distance, is_airport, dropoff_longitude , dropoff_latitude FROM({0}) WHERE
                MOD(ABS(FARM_FINGERPRINT(unique_key)), 2500) = 5
                """.format(
        bql
    )

    bq_valid_data_op = BigQueryOperator(
        task_id="bq_evaluation_data_prep",
        bql=bql_valid,
        destination_dataset_table="{}.{}_validation_data".format(
            DESTINATION_DATASET, MODEL
        ),
        write_disposition="WRITE_TRUNCATE",  # specify to truncate on writes
        use_legacy_sql=False,
        dag=dag,
    )

    train_files = BUCKET + "/chicago_taxi/data/train/"
    valid_files = BUCKET + "/chicago_taxi/data/valid/"

    # Export the results of the previous BigQueryOperators to
    # Google Cloud Storage to stage for AI Platform Training job.
    bq_export_train_csv_op = BigQueryToCloudStorageOperator(
        task_id="bq_export_to_gcs_training_data_csv",
        source_project_dataset_table="{}.{}_training_data".format(
            DESTINATION_DATASET, MODEL
        ),
        destination_cloud_storage_uris=[train_files + "{}/train-*.csv".format(MODEL)],
        export_format="CSV",
        print_header=False,
        dag=dag,
    )

    bq_export_valid_csv_op = BigQueryToCloudStorageOperator(
        task_id="bq_export_to_gcs_validation_data_csv",
        source_project_dataset_table="{}.{}_validation_data".format(
            DESTINATION_DATASET, MODEL
        ),
        destination_cloud_storage_uris=[valid_files + "{}/valid-*.csv".format(MODEL)],
        export_format="CSV",
        print_header=False,
        dag=dag,
    )

    # Python callable to set NEW_VERSION_NAME Airflow variable.
    def set_new_version_name(**kwargs):
        Variable.set(
            "NEW_VERSION_NAME",
            "v_{0}".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S")),
        )

    # SET the new python version
    py_new_version_name_op = PythonOperator(
        task_id="python_new_version_name_task",
        python_callable=set_new_version_name,
        provide_context=True,
        dag=dag,
    )

    # Required arguments for MLEngineTrainingOperator
    job_id = "taxi_{}_{}".format(
        MODEL, datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    )
    output_dir = BUCKET + "/taxi/trained_model/{}".format(MODEL)
    log_dir = BUCKET + "/taxi/training_logs/{}".format(MODEL)
    job_dir = JOB_DIR + "/" + job_id
    train_data_path = train_files + MODEL + "/*.csv"
    eval_data_path = valid_files + MODEL + "/*.csv"
    output_ds = f"{PROJECT_ID}.{DESTINATION_DATASET}"

    # BashOperator to remove the old SavedModel to ensure a fresh run.
    sh_rm_trained_model_op = BashOperator(
        task_id="bash_remove_old_trained_model_{}_task".format(MODEL),
        bash_command=(
            "if gsutil ls {0} 2> /dev/null;"
            "then gsutil -m rm -rf {0}/*; else true; fi".format(output_dir + MODEL)
        ),
        dag=dag,
    )

    # requried arguments expected by the trainer package.
    training_args = [
        "--job-dir",
        job_dir,
        "--output_dir",
        output_dir,
        "--log_dir",
        log_dir,
        "--train_data_path",
        train_data_path,
        "--eval_data_path",
        eval_data_path,
        "--output_ds",
        output_ds,
        "--version_name",
        Variable.get("NEW_VERSION_NAME"),
    ]

    # Task to submit AI Platform training job
    ml_engine_training_op = MLEngineTrainingOperator(
        task_id="ml_engine_training_{}".format(MODEL),
        project_id=PROJECT_ID,
        job_id=job_id,
        package_uris=[PACKAGE_URI],
        training_python_module="trainer.task",
        training_args=training_args,
        region=REGION,
        scale_tier="BASIC",
        runtime_version=TF_RUNTIME_VERSION,
        python_version=PYTHON_VERSION,
        dag=dag,
    )

    # BashOperator to copy SavedModel into staging location for AI Platform
    bash_copy_saved_model_op = BashOperator(
        task_id="bash_copy_new_saved_tf_model_{}".format(MODEL),
        bash_command=(
            "gsutil -m rsync -d -r {0} {1}".format(output_dir, MODEL_LOCATION + MODEL)
        ),
        dag=dag,
    )

    # Using an MLEngineModelOperator to create the new model.
    ml_engine_create_model_op = MLEngineModelOperator(
        task_id="ml_engine_create_tf_model_{}".format(MODEL),
        project_id=PROJECT_ID,
        model={"name": MODEL},
        operation="create",
        dag=dag,
    )

    # Set CURRENT_VERSION_NAME Airflow variable
    def set_current_version_name(**kwargs):
        Variable.set("CURRENT_VERSION_NAME", Variable.get("NEW_VERSION_NAME"))

    # PythonOperator to run the above Python callable
    py_current_version_name_op = PythonOperator(
        task_id="python_curent_version_name",
        python_callable=set_current_version_name,
        provide_context=True,
        trigger_rule="none_failed",
        dag=dag,
    )

    # MLEngineVersionOperator with operation set to "create" to create a new
    # version of our model
    ml_engine_create_version_op = MLEngineVersionOperator(
        task_id="ml_engine_create_version_{}".format(MODEL),
        project_id=PROJECT_ID,
        model_name=MODEL,
        version_name=Variable.get("CURRENT_VERSION_NAME"),
        version={
            "name": Variable.get("CURRENT_VERSION_NAME"),
            "deploymentUri": MODEL_LOCATION + MODEL,
            "runtimeVersion": TF_RUNTIME_VERSION,
            "framework": "TENSORFLOW",
            "pythonVersion": PYTHON_VERSION,
        },
        operation="create",
        trigger_rule="none_failed",
        dag=dag,
    )

    # MLEngineVersionOperator with operation set to "set_default" to sets
    # newly deployed version to be the default version.
    ml_engine_set_default_version_op = MLEngineVersionOperator(
        task_id="ml_engine_set_default_version_{}".format(MODEL),
        project_id=PROJECT_ID,
        model_name=MODEL,
        version_name=Variable.get("NEW_VERSION_NAME"),
        version={"name": Variable.get("NEW_VERSION_NAME")},
        operation="set_default",
        dag=dag,
    )

    # setting the up-stream and down-strean to set DAG
    bq_check_data_op >> py_new_version_name_op
    bq_check_data_op >> [bq_train_data_op, bq_valid_data_op]
    bq_check_data_op >> sh_rm_trained_model_op

    bq_train_data_op >> bq_export_train_csv_op
    bq_valid_data_op >> bq_export_valid_csv_op

    [bq_export_train_csv_op, bq_export_valid_csv_op] >> ml_engine_training_op
    py_new_version_name_op >> ml_engine_training_op
    sh_rm_trained_model_op >> ml_engine_training_op

    ml_engine_training_op >> py_current_version_name_op
    ml_engine_training_op >> bash_copy_saved_model_op
    ml_engine_training_op >> ml_engine_create_model_op

    [bash_copy_saved_model_op, ml_engine_create_model_op] >> ml_engine_create_version_op
    py_current_version_name_op >> ml_engine_create_version_op
    ml_engine_create_version_op >> ml_engine_set_default_version_op
