# fairness_benchmark

This repo provides the code to reproduce the experiments in "An Empirical Characterization of Fair Machine Learning For Clinical Risk Prediction" by Stephen R. Pfohl, Agata Foryziarz, and Nigam H. Shah. Paper: https://arxiv.org/abs/2007.10306

The repo is structured into two main modules:

* prediction_utils
    * This is a library that defines cohort definition, feature extraction, and modeling pipelines.
* fairness_benchmark
    * This codebase calls functions from prediction_utils to run the experiments in the paper.