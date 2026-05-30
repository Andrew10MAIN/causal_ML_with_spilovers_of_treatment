# Spatial Causal Machine Learning Simulation Framework

## Purpose

This repository contains a reproducible simulation framework for evaluating causal machine learning methods in a spatial setting. The experiment generates 1,000 spatial samples and compares three causal inference approaches under treatment spillover effects modeled using distance-based structures.

The evaluated models are:

Causal Forest with distance rings (discrete spillovers)
Causal Forest with continuous treatment (distance-decay spillovers)
S-learner with distance rings (discrete spillovers)

The framework also includes a Difference-in-Differences-style structure extended to spatial spillovers via distance rings.

## Reproducibility

To reproduce the full experiment:

git clone <repo>
cd repo

python -m venv .venv
.\.venv\Scripts\activate   # Windows

pip install -r requirements.txt

python pipelines/run_all.py

After execution, all simulation results and model outputs will be generated. The results can be further explored and visualized using the Jupyter notebooks located in the results/ directory.

## Configuration

To customize the experimental setup, modify:

configs/experiment.yaml – main experiment parameters
src/utils/config.py – configuration handling and parsing logic
src/models/modelling.py – model definitions and training logic
pipelines/run_models.py – model execution pipeline

## Project Structure

configs/        Experiment configuration files
data/           Simulated spatial datasets (generated samples)
pipelines/      End-to-end workflows (simulation + modeling)
src/            Core logic, utilities, and model implementations
results/        Model outputs, figures, and Jupyter notebooks for analysis

## Pipeline Overview

The main entry point is:

python pipelines/run_all.py

This script executes two sequential pipelines:

Data Generation Pipeline
Simulates 1,000 spatial datasets under predefined causal structures
Model Estimation Pipeline
Fits and evaluates the three causal ML models
Stores results for downstream analysis