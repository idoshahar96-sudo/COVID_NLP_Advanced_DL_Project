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
- ***Knowledge Distillation (KD)***: training smaller student models (e.g., HunggingFace's [arampacha/roberta-tiny](https://huggingface.co/arampacha/roberta-tiny)) with guidance from fine-tuned teachers (the aforementioned [BERTweet](https://huggingface.co/vinai/bertweet-base) and [RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base) variants above).

All models were later compared using train & test metrics (accuracy, F1-score, precision and recall) & parameter count. The comparison was formatted in a neat CSV file and model weights are kept in the accessible through drive.

## How to reproduce using Git Clone?
The notebook has been run using Google Colab, while all model weights & relevant folders have been saved to my Google Drive while running the notebook (this includes model weights, best hyperparameters, datasets and more). In a nutshell:
- Access the publicly available [project folder](https://drive.google.com/drive/folders/1egGGJ6F878xIk_bKUfjhyZStESiliwRC?usp=sharing), called "Project_COVID_NLP", which is located in Google Drive.
- The data could be obtained from Kaggle, but also from the project folder - in the [data subfolder](https://drive.google.com/drive/folders/1S1jxDoTxNXFIZKACrRp9nj2jhgatr_Au?usp=sharing). More specifically:
- - [Original Training Data - CORONA_NLP_Train](https://drive.google.com/file/d/1dCbfsXJuU_Ers3k1JmtjLrfqVMUgEO6m/view?usp=drive_link)
- - [Original Test Data - CORONA_NLP_Test](https://drive.google.com/file/d/1fmODEknrlX9MkB7VCr7EaZOPCpZy6QUx/view?usp=sharing)
- - []()
- - []()
