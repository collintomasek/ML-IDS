# Intrusion Detection System (UNSW-NB15)

## Overview
Machine-learning-based IDS for detecting malicious network traffic.

## Dataset
UNSW-NB15 dataset with flow-level network features.

## Approach
- Data preprocessing and feature selection
- Supervised learning (Random Forest)
- Evaluation via precision, recall, F1-score

## How to Run
pip install -r requirements.txt
python src/ids_model.py

## Project Structure
src/ - model logic
data/ - dataset instructions
reports/ - evaluation outputs

## Future Work
- Real-time packet ingestion
- Model explainability
- SOC-style alerting
