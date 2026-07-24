# Collab Project — ML for Cloud & Collaboration

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

## Colab Ready

Each project has a notebook in `notebooks/` for interactive exploration in Google Colab.
