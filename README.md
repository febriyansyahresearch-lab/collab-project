# Collab Project — ML for Cloud & Collaboration

[![CI](https://github.com/febriyansyahresearch-lab/collab-project/actions/workflows/test.yml/badge.svg)](https://github.com/febriyansyahresearch-lab/collab-project/actions)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-20%20passed-brightgreen)](projects/)

**Febriyansyah** — MTI, IT Security Leader (15+ yrs, Banking)

Collaborative ML workspace designed for Google Colab, VS Code, and GitHub Codespaces.

## Projects

| Project | Type | Models | Tests |
|---|---|---|---|
| `projects/nlp/` | Text Classification | Naive Bayes, Logistic, SVM | 5 ✅ |
| `projects/image/` | Image Classification | RandomForest, MLP | 5 ✅ |
| `projects/timeseries/` | Time Series Forecasting | Linear, RF, GBR | 5 ✅ |
| `projects/recommender/` | Recommendation System | Collaborative Filtering | 5 ✅ |

## Setup

```bash
pip install -r requirements.txt
```

## Test

```bash
python -m pytest projects/ -v
```

## Usage

```bash
# NLP: train text classifier (TF-IDF + Naive Bayes, Logistic, SVM)
python -m projects.nlp.src.train

# Image: train shape classifier (RF, MLP)
python -m projects.image.src.train

# Time Series: forecast with trend + seasonality
python -m projects.timeseries.src.train

# Recommender: user-based collaborative filtering
python -m projects.recommender.src.recommend
```

## Colab Ready

Each project has a notebook in `notebooks/` for interactive exploration in Google Colab.
