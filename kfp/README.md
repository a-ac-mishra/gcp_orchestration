## Documentation
---------------------

### Initial Setup
-------------------

1. Set up and deploy an AI Platform Pipeline on a new Kubernetes cluste

  - Go to `AI Platform` -> `Pipeline`
  - Click on `New Instance` & on next page click `Configure`
  - Give cluster name, `Allow access to the following cloud APIs` and click on `CREATE NEW CLUSTER`
  - Click on `Deploy` at bottom of the page, wait for the deployment. Once deployed, you visit the kubeflow dashboard

2. Git clone the repository. You should see the following structure
  - kfp_tfx_pipeline
    - data
    - models
      - keras_model
        - model.py
      - features.py
      - preprocessing.py
    - pipeline
      - pipeline.py
    - kubeflow_runner.py
  - run.ipynb

3. Make the following changes
  - go to run.ipynb, change  `PIPELINE_NAME`, `ENDPOINT` (url of kubeflow pipeline)
  - go to kubeflow_runner.py , change `PIPELINE_NAME`, `PROJECT_NAME`

4. Clear all outputs and run all the cells one by one.
