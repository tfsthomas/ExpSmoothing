import numpy as np
import pandas as pd
import scipy.special as spsp 
import matplotlib.pyplot as plt


def simple_exp_smooth(d,extra_periods=1,alpha=0.4):  
  d = np.array(d)  # Transform the input into a numpy array  
  cols = len(d)  # Historical period length  
  d = np.append(d,[np.nan]*extra_periods)  # Append np.nan into the demand array to cover future periods  
  f = np.full(cols+extra_periods,np.nan)  # Forecast array  
  f[1] = d[0]  # initialization of first forecast  
  # Create all the t+1 forecasts until end of historical period  
  for t in range(2,cols+1):  
    f[t] = alpha*d[t-1]+(1-alpha)*f[t-1]  
  f[cols+1:] = f[t]  # Forecast for all extra periods  
  df = pd.DataFrame.from_dict({"Demand":d,"Forecast":f,"Error":d-f})
  print(df)
  MAE = df["Error"].abs().mean()  
  print("MAE:",round(MAE,2)) 
  RMSE = np.sqrt((df["Error"]**2).mean())
  print("RMSE:",round(RMSE,2))
  df.index.name = "Periods"
  df[["Demand","Forecast"]].plot(figsize=(8,3),title="Simple Smoothing",ylim=(0,30),style=["-","--"])
  

d=[28,19,18,13,19,16,19,18,13,16,16,11,18,15,13,15,13,11,13,10,12]
simple_exp_smooth(d,extra_periods=4)


def double_exponential_smoothing(series, alpha, beta, n_preds=2):
    """
    Given a series, alpha, beta and n_preds (number of
    forecast/prediction steps), perform the prediction.
    """
    n_record = series.shape[0]
    results = np.zeros(n_record + n_preds)

    # first value remains the same as series,
    # as there is no history to learn from;
    # and the initial trend is the slope/difference
    # between the first two value of the series
    level = series[0]
    results[0] = series[0]
    trend = series[1] - series[0]
    for t in range(1, n_record + 1):
        if t >= n_record:
            # forecasting new points
            value = results[t - 1]
        else:
            value = series[t]

        previous_level = level
        level = alpha * value + (1 - alpha) * (level + trend)
        trend = beta * (level - previous_level) + (1 - beta) * trend 
        results[t] = level + trend


    if n_preds > 1:
        results[n_record + 1:] = level + np.arange(2, n_preds + 1) * trend

    return results

d=[28,19,18,13,19,16,19,18,13,16,16,11,18,15,13,15,13,11,13,10,12]
d=np.array(d)

double_exponential_smoothing(d, 0.8, 0.9)
