# Flight Delay Predictions

By: Victor Cheung, Cole Hynes, Yash Jagirdar, Josiah Lam, Rohan Nair, Colin Pierce
 
## Project Overview

The primary objective of this project is to develop a predictive model for flight delays using machine learning techniques. By utilizing historical flight data, we aim to:
- Obtain insights into the key factors contributing to flight delays, such as weather conditions, time of flight etc.
- Compare different regression models to determine the most effective approach for delay predictions
- Optimize model performance through parameter tuning and feature selection to achieve the best predictions

Flight delays are a significant issue in the aviation industry, affecting passengers, airlines, and airport operations. Delay predictions can help mitigate their impact by allowing airlines to optimize schedules, improve customer experience, and reduce operational costs. This project aims to develop a predictive model that can identify patterns contributing to delays based on historical data.

## Problem Statement

Flight delays are a significant issue in the aviation industry, affecting passengers, airlines, and airport operations. Delay predictions can help mitigate their impact by allowing airlines to optimize schedules, improve customer experience, and reduce operational costs. This project aims to develop a predictive model that can identify patterns contributing to delays based on historical data.

A comprehensive study done by UC Berkeley estimated that in 2007, the cost of U.S. flight delays was $32.9 billion. This included $8.3 billion in increased airline expenses for crew, fuel, and maintenance, $16.7 billion in passengers' lost time due to delayed flights, cancellations, and missed connections, and $3.9 billion from lost demand, representing the welfare loss of passengers who avoided air travel due to delays. (Ball et al., 2010)

The following models will be used for this project:

- Linear Regression
- KNN
- Support Vector Regression (SVR)
- Random Forest
- Gradient Boosting

## Data Collection

We will use the **Flight Take-Off data** from JFK Airport available on Kaggle.

https://www.kaggle.com/datasets/deepankurk/flight-take-off-data-jfk-airport

## Dataset Features

| Feature              | Description                   | Feature            | Description                |
|----------------------|-------------------------------|--------------------|----------------------------|
| MONTH (int)          | Month                         | Temperature (int)  | Temp.                      |
| DAY_OF_MONTH (int)   | Date of flight                | Dew Point (int)    | Dew                        |
| DAY_OF_WEEK (int)    | Day of the week               | Humidity (int)     | Hum                        |
| OP_UNIQUE_CARRIER (string) | Carrier Code             | Wind (string)      | Wind                       |
| TAIL_NUM (string)    | Airflight Number              | Wind Speed (int)   | Wind speed                 |
| DEST (string)        | Destination                   | Wind Gust (int)    | Wind Gust                  |
| CRS_ELAPSED_TIME (int) | Scheduled journey time       | Pressure (float)   | Pressure                   |
| DISTANCE (int)       | Distance of the flight        | Condition (string) | Condition of the climate   |
| CRS_DEP_M (int)      | Scheduled Departure Time      | Sch_dep (int)      | No. of flights scheduled for arrival |
| CRS_ARR_M (int)      | Scheduled Arrival Time        | Sch_arr (int)      | No. of flights scheduled for departure |

The target variable will be **DEP_DELAY** (Departure delay of the flight).

## Performance Metrics

Model performance will be compared using the following metrics:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)

## References

Ball, M., Barnhart, C., Dresner, M., Hansen, M., Neels, K., Odoni, A. R., Peterson, E., Sherry, L., Trani, A., & Zou, B. (2010, October 1). Total delay impact study : A comprehensive assessment of the costs and impacts of flight delay in the United States. National Transportation Library. https://rosap.ntl.bts.gov/view/dot/6234

Kansal, D. (2021, June 11). Flight take off data - JFK airport. Kaggle. https://www.kaggle.com/datasets/deepankurk/flight-take-off-data-jfk-airport/data

