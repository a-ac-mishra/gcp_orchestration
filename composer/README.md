# Running in Cloud Shell

1. Enable google cloud servies
```
gcloud services enable ml.googleapis.com
gcloud services enable composer.googleapis.com
```
2. create environment variables
```
export REGION=us-central1
export BUCKET_NAME=${DEVSHELL_PROJECT_ID}
```
3. create GCS bucket and copy artifacts
```
gsutil mb -l ${REGION} gs://${BUCKET_NAME}
tar -cvf trainer.tar chicago_taxifare
gsutil cp ./trainer.tar gs://${BUCKET_NAME}/chicago_taxi/code/
```
4. create bigquery table.
```
bq mk -d chicago_taxi_fare
bq mk --table chicago_taxi_fare.model_metrics version_name:STRING,rmse:FLOAT
```
**NOTE**: Please change the project name/ id, bigquery dataset, bigquery tablename & GCS bucket name; if changed in above commands
5. spin up the cloud composer instance
```
gcloud composer environments create cc-environment \
  --location $REGION \
  --python-version 3
```

6. update any environment variables for cloud composer
```
gcloud composer environments storage data import \
  --source vars.json \
  --environment cc-environment \
  --location $REGION
```

```
gcloud composer environments run cc-environment\
  --location $REGION variables \
  -- \
  --i /home/airflow/gcs/data/vars.json
```
7. copy airflow dag to airflow GCS bucket
```
export DAGS_FOLDER=$(gcloud composer environments describe cc-environment \
   --location $REGION   --format="get(config.dagGcsPrefix)")
   
gsutil cp ./chicago_taxi_dag.py ${DAGS_FOLDER}/
```
8. Last step is to manually go to airflow UI and trigger the DAG.