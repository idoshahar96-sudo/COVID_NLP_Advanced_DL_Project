# Advanced Topics in DL - Project - Sentiment Analysis of COVID-19 Tweets using Fine-Tuned NLP Models
This project applies advanced deep learning techniques for sentiment analysis of COVID-19 tweets. It has been a major part of a course we took - "Advanced Topics in Deep Learning" at TAU's faculty of engineering (2025), during the 3rd year of our studies. As a group of 2, we fine-tuned two pre-trained HuggingFace transformer models (BERTweet and RoBERTa variants) for sentiment analysis on the COVID-19 Twitter Sentiment Dataset from Kaggle. Building on these, we explored model compression techniques (pruning, quantization, and knowledge distillation) to evaluate the trade-off between efficiency and performance, under a strict compute budget.

The first, fine-tuning stage included:
- *Preprocessing: tweet cleaning, tokenization and more.*
- *Data splitting - training & validation*.
- *Transfer Learning of pre-trained HuggingFace Transformers*.
- *Hyperparameter Tuning with [Optuna](https://optuna.org/)*.
- *Experiment Tracking with [Weights & Biases (W & B)](https://wandb.ai/) API*.

Afterwards, the second, model compression stage included three complementary post-training techniques:

- **Pruning**: structured pruning of attention and dense layers.

- **Quantization**: dynamic post-training quantization to reduce model size and accelerate inference.

- **Knowledge Distillation**: training smaller student models (e.g., HunggingFace's arampacha/roberta-tiny) with guidance from fine-tuned teachers (BERTweet, RoBERTa).

3. Evaluation

All models were compared on:

Accuracy, F1, and inference speed.

Parameter count & model size trade-offs.

Generalization performance under constrained compute.

Key Assumptions & Workflow

Google Drive Storage

Paths in notebooks assume a base folder:

BASIC_DRIVE_PATH = "/content/drive/MyDrive"
DATA_PATH = f"{BASIC_DRIVE_PATH}/COVID_NLP_Advanced_DL_Project"


All trained weights, results CSVs, and artifacts are stored under this hierarchy.

Weights & Artifacts

Pruned and quantized models are saved in dedicated subfolders (/Pruned_Model_Weights, /Quantized_Model_Weights).

KD results are exported as CSVs named after the student model for clarity, e.g.:

KD_results_arampacha_roberta_tiny.csv


Experiment Tracking

All training runs were logged with W&B.

Due to project restrictions, projects are private by default; sharing requires collaborator access.

How to Run

Clone the repo:

git clone https://github.com/IdanKanat/COVID_NLP_Advanced_DL_Project.git
cd COVID_NLP_Advanced_DL_Project


Set up environment:

Python ≥ 3.9

Install requirements:

pip install -r requirements.txt


Prepare Data:

Download the Corona_NLP dataset (Kaggle).

Place it under DATA_PATH/datasets.

Run notebooks:

[Fine_Tuning.ipynb] → baseline training

[Pruning.ipynb] → pruning + eval

[Quantization.ipynb] → quantization + eval

[KD.ipynb] → knowledge distillation + eval

Using trained models:

Pretrained fine-tuned weights are stored in Drive.

If you wish to re-use them, adjust paths in the notebooks to your own Drive mount.
