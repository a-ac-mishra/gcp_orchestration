import tensorflow_model_analysis as tfma
from tfx import v1 as tfx
from ml_metadata.proto import metadata_store_pb2


def create_pipeline(
    pipeline_name,
    pipeline_root,
    data_path,
    preprocessing_fn,
    run_fn,
    training_args,
    evaluation_args,
    eval_accuracy_threshold,
    serving_model_dir,
):
    """create TFX pipeline

    Args:
        pipeline_name : Name of the pipeline
        pipeline_root : GCS path for pipeline root
        data_path : GCS path
        preprocessing_fn : fucntion for preprocessing
        run_fn : function for running
        training_args : training arguments for TF model
        evaluation_args : evaluation arguments for TF model
        eval_accuracy_threshold : evaluation accuracy threshold
        serving_model_dir : serving model directory

    Returns:
        tfx pipeline
    """

    tfx_components = []
    # ingesting data into the pipeline
    example_gen_component = tfx.components.CsvExampleGen(input_base=data_path)
    tfx_components.append(example_gen_component)
    # Statistics generation using data
    statistics_gen_component = tfx.components.StatisticsGen(
        examples=example_gen_component.outputs["examples"]
    )
    tfx_components.append(statistics_gen_component)
    # Schema generation using statistics
    schema_gen_component = tfx.components.SchemaGen(
        statistics=statistics_gen_component.outputs["statistics"]
    )
    tfx_components.append(schema_gen_component)
    # Anomaly detection by example validator using statistics and schema.
    example_validator_component = tfx.components.ExampleValidator(
        statistics=statistics_gen_component.outputs["statistics"],
        schema=schema_gen_component.outputs["schema"],
    )
    tfx_components.append(example_validator_component)
    # Performs transformations and feature engineering in training and serving.
    transform_component = tfx.components.Transform(
        examples=example_gen_component.outputs["examples"],
        schema=schema_gen_component.outputs["schema"],
        preprocessing_fn=preprocessing_fn,
    )
    tfx_components.append(transform_component)
    # user provided arguments for implementing dnn model.
    trainer_args = {
        "schema": schema_gen_component.outputs["schema"],
        "run_fn": run_fn,
        "examples": transform_component.outputs["transformed_examples"],
        "transform_graph": transform_component.outputs["transform_graph"],
        "train_args": training_args,
        "eval_args": evaluation_args,
    }
    trainer_component = tfx.components.Trainer(**trainer_args)
    tfx_components.append(trainer_component)
    # Get the latest blessed model for model validation.
    model_resolver_component = tfx.dsl.Resolver(
        strategy_class=tfx.dsl.experimental.LatestBlessedModelStrategy,
        model=tfx.dsl.Channel(type=tfx.types.standard_artifacts.Model),
        model_blessing=tfx.dsl.Channel(type=tfx.types.standard_artifacts.ModelBlessing),
    ).with_id("latest_blessed_model_resolver")
    tfx_components.append(model_resolver_component)
    # set the evaluation config
    evaluation_config = tfma.EvalConfig(
        model_specs=[
            tfma.ModelSpec(
                signature_name="serving_default",
                label_key="tips_xf",
                preprocessing_function_names=["transform_features"],
            )
        ],
        slicing_specs=[tfma.SlicingSpec()],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(
                        class_name="BinaryAccuracy",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": eval_accuracy_threshold}
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value": -1e-10},
                            ),
                        ),
                    )
                ]
            )
        ],
    )
    # evaluation component
    evaluator_component = tfx.components.Evaluator(
        examples=example_gen_component.outputs["examples"],
        model=trainer_component.outputs["model"],
        baseline_model=model_resolver_component.outputs["model"],
        eval_config=evaluation_config,
    )
    tfx_components.append(evaluator_component)
    # pusher arguments
    pusher_args = {
        "model": trainer_component.outputs["model"],
        "model_blessing": evaluator_component.outputs["blessing"],
    }
    pusher_args["push_destination"] = tfx.proto.PushDestination(
        filesystem=tfx.proto.PushDestination.Filesystem(
            base_directory=serving_model_dir
        )
    )
    # pusher components
    pusher_component = tfx.components.Pusher(**pusher_args)
    tfx_components.append(pusher_component)

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=tfx_components
        # enable_cache=True
    )
