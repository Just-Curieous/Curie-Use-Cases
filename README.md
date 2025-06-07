# Curie Use Cases

[![arXiv](https://img.shields.io/badge/arXiv-2502.16069-b31b1b.svg)](https://arxiv.org/abs/2502.16069)
[![Slack](https://img.shields.io/badge/Slack-Join%20Community-4A154B?logo=slack)](https://join.slack.com/t/just-curieous/shared_invite/zt-313elxhhy-hpEK5r9kX9Xv1Pfxzt9CJQ)
[![Demo](https://img.shields.io/badge/Demo-Live-green)](http://44.202.70.8:5000/)
[![Blog](https://img.shields.io/badge/Blog-Read%20More-orange)](https://www.just-curieous.com/)

This repository provides a collection of use cases and examples demonstrating the application of the [Curie](https://github.com/Just-Curieous/Curie) for automated and rigorous scientific experimentation with AI agents.

## Overview

The Curie framework is designed to facilitate scientific discovery across various domains by automating the experimentation process. This repository contains specific implementations and experiments that showcase how Curie can be applied to different problems, particularly in machine learning and stock prediction.

## Directory Structure

- **🤖 General Machine Learning Experiments**: Contains use cases related to machine learning experiments.
  - [`q1_activation_func`](./machine_learning/q1_activation_func): Experiments with different activation functions in neural networks.
  - [`q2_dog-breed-identification`](./machine_learning/q2_dog-breed-identification): Image classification task for identifying dog breeds.
  - [`q3_siim-isic-melanoma-classification`](./machine_learning/q3_siim-isic-melanoma-classification): Classification task for melanoma detection in medical images.
  - [`q4_aptos2019-blindness-detection`](./machine_learning/q4_aptos2019-blindness-detection): Classification task for detecting diabetic retinopathy in eye images.
  - [`q5_histopathologic-cancer-detection`](./machine_learning/q5_histopathologic-cancer-detection): Classification task for detecting metastatic cancer.
- **📈 Stock Price Prediction Use Case**: Put machine learning to work on real-world financial data (note: raw data not published due to privacy constraints):
  - [`q0_general_optimize`](./stock_prediction/q0_general_optimize): Explore general optimization strategies to tune your models for peak performance.
  - [`q1_optimize_hyperparameter`](./stock_prediction/q1_optimize_hyperparameter):  Test out different hyperparameter optimizations for stock prediction performance.
  - [`q2_feature_selection`](./stock_prediction/q2_feature_selection):  Test out different feature selection parameters to improve stock prediction accuracy.
  - [`q3_feature_engineering`](./stock_prediction/q3_feature_engineering):  Test out different feature engineering techniques for stock prediction.
  - [`q4_ensemble`](./stock_prediction/q4_ensemble): Test out different ensemble methods for improving stock prediction accuracy.
  - [`starter_code`](./stock_prediction/starter_code): Starter code and templates for stock prediction experiments.


  
<!-- - **template**:Provides a template for creating new use cases with the Curie framework. -->
<!-- 
## Demo
Curie can automatically generate experiment report for you research question, here is a sample report for `Histopathologic Cancer Detection` task:
 -->
