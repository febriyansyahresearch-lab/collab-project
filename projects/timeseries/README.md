# Time Series — Forecasting

**Domain:** Time Series Analysis  
**Models:** Linear Regression, RandomForest, Gradient Boosting  

## Methodology

1. **Data**: Synthetic daily series with trend + seasonality + noise
2. **Feature Engineering**: Day index, month sin/cos, day of week
3. **Models**: Compare 3 regressors
4. **Evaluation**: RMSE, MAE

## Usage

```bash
python -m projects.timeseries.src.train
```
