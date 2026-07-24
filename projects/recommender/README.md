# Recommender — Collaborative Filtering

**Domain:** Recommendation Systems  
**Algorithm:** User-based collaborative filtering (cosine similarity)  

## Methodology

1. **Data**: Synthetic user-item ratings (50 users × 20 items)
2. **Similarity**: Cosine similarity between users
3. **Prediction**: Weighted average of similar users' ratings
4. **Output**: Top-N item recommendations

## Usage

```bash
python -m projects.recommender.src.recommend
```
