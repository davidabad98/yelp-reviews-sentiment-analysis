# Comparative Analysis of Deep Learning Models for Sentiment Analysis on Yelp Reviews

This project implements, trains, and evaluates two deep learning architectures (LSTM and DistilBERT) for sentiment analysis on Yelp reviews in the hospitality industry. The models classify reviews as positive, negative, or neutral and are compared for performance across different text characteristics.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Results](#results)
- [License](#license)

## 🔍 Project Overview

This project performs a systematic comparison of traditional sequence models (LSTM) against transformer-based architectures (DistilBERT) for sentiment analysis in the domain of restaurant and hotel reviews. Key aspects include:

- Data preprocessing specific to review text
- Implementation of both LSTM and DistilBERT architectures
- Robust training with cross-validation
- Comprehensive evaluation metrics
- Model explainability analysis

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip
- CUDA-compatible GPU (recommended for training)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/your-username/yelp-reviews-sentiment-analysis.git
cd yelp-reviews-sentiment-analysis
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .  # Install the project in development mode
```

## 📊 Data Preparation

1. Download the Yelp review dataset:
```bash
# Run the data download script
python scripts/download_data.py
```

2. Preprocess the data:
```bash
# Process raw data into training/validation/test splits
python scripts/preprocess_data.py
```

Alternatively, you can run the Jupyter notebooks in the `notebooks` directory sequentially:
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
jupyter notebook notebooks/02_data_preprocessing.ipynb
```

## 📁 Project Structure

```
yelp-reviews-sentiment-analysis/
├── README.md                       # Project overview and instructions 
├── requirements.txt                # Dependencies
├── setup.py                        # Package installation
├── .gitignore                      # Git ignore file
├── data/                           # Data directory
│   ├── raw/                        # Raw downloaded data
│   ├── processed/                  # Processed datasets
│   └── embeddings/                 # Word embeddings for LSTM
├── logs/                           # Training logs
├── notebooks/                      # Jupyter notebooks for exploration and development
├── results/                        # Model evaluation results
├── src/                            # Source code
│   ├── data/                       # Data handling
│   ├── models/                     # Model implementations
│   ├── training/                   # Training utilities
│   ├── utils/                      # Helper functions
│   └── inference/                  # Inference utilities
└── scripts/                        # Training and evaluation scripts
```

## 🚂 Model Training

### LSTM Model
```bash
python scripts/train_lstm.py --epochs 20 --batch_size 64 --learning_rate 0.001
```

### DistilBERT Model
```bash
python scripts/train_distilbert.py --epochs 5 --batch_size 16 --learning_rate 2e-5
```

Both scripts support additional arguments:
- `--save_dir`: Directory to save model checkpoints
- `--log_dir`: Directory for TensorBoard logs
- `--device`: Computation device (cuda/cpu)

## 🔧 Hyperparameter Tuning

Run hyperparameter optimization for each model:

```bash
# LSTM hyperparameter tuning
python scripts/tune_lstm.py --trials 50 --max_epochs 15

# DistilBERT hyperparameter tuning
python scripts/tune_distilbert.py --trials 20 --max_epochs 5
```

## 📏 Evaluation

Evaluate and compare both models:

```bash
python scripts/evaluate_models.py --lstm_checkpoint path/to/lstm/checkpoint --distilbert_checkpoint path/to/distilbert/checkpoint
```

This will generate comprehensive evaluation metrics and comparison visualizations in the `results/` directory.

## 🔮 Inference

For making predictions with trained models:

```python
from src.inference.predictor import SentimentPredictor

# Initialize predictor with trained model
predictor = SentimentPredictor(model_type="distilbert", model_path="path/to/model/checkpoint")

# Predict sentiment for a review
sentiment = predictor.predict("The food was amazing and the staff was very friendly!")
print(f"Predicted sentiment: {sentiment}")
```

## 📈 Results

Detailed results, including performance metrics, comparative analysis, and visualization of model outputs can be found in the `results/` directory after running the evaluation script.

Key metrics compared:
- Accuracy, Precision, Recall, F1-score
- Confusion matrices
- ROC curves
- Inference time
- Model interpretability analysis

## Contributors
- **[David Abad](https://github.com/davidabad98)**
- **[Rizvan Nahif](https://github.com/joyrizvan)**
- **[Darshil Shah](https://github.com/darshil0811)**
- **[Navpreet Kaur Dusanje](https://github.com/Navpreet-Kaur-Dusanje)**

## 📄 License

[MIT License](LICENSE)
