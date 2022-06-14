import os
from absl import logging
from tfx import v1 as tfx
from pipeline import pipeline

# user defined pipeline name and user created GCP project
PROJECT_NAME = ""
if PROJECT_NAME == "":
    raise "Please provide PROJECT_NAME (GCP Project ID)"
PIPELINE_NAME = "kfp_tfx_pipeline"

# Kubeflow Pipelines will used the below image to run TFX pipeline components.
# Docker Image will be automatically built using TFX CLI command when used with --build-image flag.
PIPELINE_IMAGE_URI = f"gcr.io/{PROJECT_NAME}/{PIPELINE_NAME}"
# preprocessing function path used by Transform component
PREPROCESSING_FN = "models.preprocessing.preprocessing_fn"
# run function path used by Trainer component
RUN_FN = "models.keras_model.model.run_fn"
# model training and evaluation hyperparameters
TRAIN_NUM_STEPS = 1000
EVAL_NUM_STEPS = 150
EVAL_ACCURACY_THRESHOLD = 0.5
GCS_BUCKET_NAME = PROJECT_NAME + "-kubeflowpipelines-default"
# data used by the pipeline is in DATA_PATH (i.e. GCS bucket)
DATA_PATH = f"gs://{GCS_BUCKET_NAME}/tfx-template/data/taxi/"
# OUTPUT_DIR will store the files and metadata produced by TFX pipeline.
OUTPUT_DIR = os.path.join("gs://", GCS_BUCKET_NAME)
# PIPELINE_ROOT will hold TFX generated files and metadata.
PIPELINE_ROOT = os.path.join(OUTPUT_DIR, "tfx_pipeline_root", PIPELINE_NAME)
# Pusher component will generate serving model under SERVING_MODEL_DIR.
SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, "serving_model")


def run():
    metadata_config = (
        tfx.orchestration.experimental.get_default_kubeflow_metadata_config()
    )

    runner_config = tfx.orchestration.experimental.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=metadata_config, tfx_image=PIPELINE_IMAGE_URI
    )

    pod_labels = {
        "add-pod-env": "true",
        tfx.orchestration.experimental.LABEL_KFP_SDK_ENV: "tfx-template",
    }
    tfx.orchestration.experimental.KubeflowDagRunner(
        config=runner_config, pod_labels_to_attach=pod_labels
    ).run(
        pipeline.create_pipeline(
            pipeline_name=PIPELINE_NAME,
            pipeline_root=PIPELINE_ROOT,
            data_path=DATA_PATH,
            preprocessing_fn=PREPROCESSING_FN,
            run_fn=RUN_FN,
            training_args=tfx.proto.TrainArgs(num_steps=TRAIN_NUM_STEPS),
            evaluation_args=tfx.proto.EvalArgs(num_steps=EVAL_NUM_STEPS),
            eval_accuracy_threshold=EVAL_ACCURACY_THRESHOLD,
            serving_model_dir=SERVING_MODEL_DIR,
        )
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    run()
