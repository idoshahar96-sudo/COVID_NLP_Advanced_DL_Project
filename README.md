# Advanced Topics in Deep Learning - Project - Sentiment Analysis of COVID-19 Tweets using Fine-Tuned NLP Models
This project applies advanced deep learning (DL) techniques for sentiment analysis of COVID-19 tweets. It has been a major part of a course we took - "Advanced Topics in Deep Learning" at TAU's faculty of engineering (2025), during the 3rd year of our studies. As a group of 2, we fine-tuned two pre-trained HuggingFace transformer models ([BERTweet](https://huggingface.co/vinai/bertweet-base) and [RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base) variants) for sentiment analysis on the [COVID-19 Twitter Sentiment Dataset from Kaggle](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification/data). Building on these, we explored model compression techniques (pruning, quantization, and knowledge distillation) to evaluate the trade-off between efficiency and performance, under a strict compute budget.

![_](https://github.com/IdanKanat/COVID_NLP_Advanced_DL_Project/blob/f10660f8b73eda11dc920446c6db5804c0e43fcd/AdvancedTopicsInDL_Project_COVID_NLP_ThemePic%20-%20FINAL.png)
The first, fine-tuning stage included:

- *Preprocessing: tweet cleaning, tokenization and more.*
- *Data splitting - training & validation*.
- *Transfer Learning of pre-trained HuggingFace Transformers for sentiment analysis of COVID-19 tweets*.
- *Hyperparameter Tuning with [Optuna](https://optuna.org/), with final training using a larger dataset*.
- *Experiment Tracking with [Weights & Biases (W & B)](https://wandb.ai/) API*.

After final training of the models using various modeling methodologies, the second, model compression stage included three complementary post-training techniques:

- ***Pruning***: structured pruning of attention and dense layers.
- ***Quantization***: dynamic post-training quantization to reduce model size and accelerate inference.
- ***Knowledge Distillation (KD)***: training smaller student models (e.g., HunggingFace's [arampacha/roberta-tiny[(https://huggingface.co/arampacha/roberta-tiny)) with guidance from fine-tuned teachers (BERTweet, RoBERTa).

All models were later compared using train & test metrics (accuracy, F1-score, precision and recall) & parameter count. The comparison was formatted in a neat CSV file and model weights are kept in the accessible through drive.

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
