# yelp-reviews-sentiment-analysis
Comparative Analysis of Deep Learning Models  for Sentiment Analysis on Yelp Reviews


References:
https://business.yelp.com/data/resources/open-dataset/
https://medium.com/analytics-vidhya/performing-sentiment-analysis-on-yelp-restaurant-reviews-962334d6336d
https://github.com/wutianqidx/Yelp-Review-Sentiment-Analysis/tree/main
https://www.kaggle.com/datasets/irustandi/yelp-review-polarity
https://github.com/dailyLi/yelp_da/tree/main
https://www.kaggle.com/code/omkarsabnis/sentiment-analysis-on-the-yelp-reviews-dataset/notebook
https://www.linkedin.com/pulse/project-yelp-reviews-sentiment-analysis-tanja-ad%C5%BEi%C4%87/
https://github.com/adzict/yelp_sentiment_analysis
https://huggingface.co/datasets/Yelp/yelp_review_full


# yelp-reviews-sentiment-analysis
Comparative Analysis of Deep Learning Models  for Sentiment Analysis on Yelp Reviews

yelp-reviews-sentiment-analysis/
├── README.md                       # Project overview and instructions
├── requirements.txt                # Dependencies
├── setup.py                        # Package installation
├── .gitignore                      # Git ignore file
├── data/
│   ├── raw/                        # Raw downloaded data
│   ├── processed/                  # Processed datasets
│   └── embeddings/                 # Word embeddings for LSTM
├── notebooks/
│   ├── 01_data_exploration.ipynb   # EDA notebook
│   ├── 02_data_preprocessing.ipynb # Preprocessing experiments
│   ├── 03_lstm_experiments.ipynb   # LSTM model experiments
│   ├── 04_distilbert_experiments.ipynb # DistilBERT experiments
│   └── 05_model_comparison.ipynb   # Comparative analysis
├── src/
│   ├── __init__.py
│   ├── config.py                   # Configuration parameters
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py          # Data loading utilities
│   │   ├── preprocessor.py         # Text preprocessing
│   │   └── dataset.py              # PyTorch dataset classes
│   ├── models/
│   │   ├── __init__.py
│   │   ├── lstm_model.py           # LSTM architecture
│   │   └── distilbert_model.py     # DistilBERT implementation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py              # Training loop implementations
│   │   └── metrics.py              # Evaluation metrics
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py               # Logging utilities
│   │   ├── visualization.py        # Visualization helpers
│   │   └── interpretability.py     # Model interpretability tools
│   └── inference/
│       ├── __init__.py
│       └── predictor.py            # Inference utilities
├── scripts/
│   ├── download_data.py            # Data download script
│   ├── train_lstm.py               # LSTM training script
│   ├── train_distilbert.py         # DistilBERT training script
│   └── evaluate_models.py          # Evaluation script
└── tests/                          # Unit tests
    ├── __init__.py
    ├── test_data.py
    ├── test_models.py
    └── test_training.py