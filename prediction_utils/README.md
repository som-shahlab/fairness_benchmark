## Prediction Utils

This library is composed of the following modules:
* `extraction_utils`: API for connecting to databases and extracting features
* `cohorts`: Definition for cohorts defined on the OMOP CDM
* `extraction_scripts`: Runtime scripts for defining cohorts and extracting features
* `pytorch_utils`: Pytorch models for supervised learning
* `vignettes`: Example workflows to help get started
* `experiments`: Computational experiments that leverage this library

### Installation
0. If you are Nero a pre-existing conda environment is available at `/share/pi/nigam/envs/anaconda/envs/prediction_utils`. Otherwise, continue with the following steps.
1. Clone the repository
2. `pip install -e .` from within the directory
3. If you plan to use OHDSI feature extractors, install the FeatureExtraction package to your R installation: https://github.com/OHDSI/FeatureExtraction

### Getting Started
* See the `vignettes` directory for examples of basic usage

### Modules

#### extraction_utils
* Connect to databases using the BigQuery client library or through the python DBAPI via SqlAlchemy
* Extract clinical features for machine learning using custom code or with the OHDSI feature extractors

#### cohorts
* The following cohorts are implemented
    * Inpatient admissions rolled up to continuous episodes of care
        * Labeling functions for this cohort
            * Hospital mortality
            * Length of stay
            * 30-day readmission
    * Implementations of MIMIC-Extract cohorts in MIMIC-OMOP

#### pytorch_utils
* Several pipelines are implemented
    * Dataloaders
        * Sparse data in scipy.sparse.csr_matrix format
        * Larger-than-memory sparse data via Arrow/Parquet and Dask Dataframe
    * Layers
        * Input layers that efficiently handle sparse inputs
        * Feedforward networks
    * Training pipelines
        * Supervised learning for binary outcomes
        * Regularized objectives for fair machine learning
