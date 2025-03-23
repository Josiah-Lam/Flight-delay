# Flight Delay Classification Model
 
## Project Overview

We chose to build an application in the transportation industry, specifically focusing on predicting flight delays. The goal of this project is to develop a flight delay prediction model using machine learning algorithms, utilizing the **Flight Take-off dataset** from JFK Airport and the **FAA Aircraft Registration dataset**. 

Our aim is to use multiple machine learning models to find the best-suited model for predicting whether a flight will be delayed by more than 15 minutes. This will help airports predict and plan for aircraft delays, ensuring a smoother flying experience. 

The following models will be used for this project:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machines (SVM)
- Neural Networks

## Data Collection

We will use the **Flight Take-Off data** from JFK Airport available on Kaggle. This dataset was scraped from an Academic Paper under Review by IEEE Transportation. Additionally, we will need **aircraft registration data** from the FAA to include additional features like the model of the aircraft, weight, and seat count, which are not present in the Kaggle dataset.

https://www.kaggle.com/datasets/deepankurk/flight-take-off-data-jfk-airport

### Datasets:
- **Flight Take-off data** from JFK Airport
- **FAA Aircraft Registration dataset** to get aircraft details

## Project Needs

This project requires relevant datasets with descriptive features to effectively predict flight delays. The current dataset contains 2,092 rows, but more data would improve model performance. Additionally, sufficient computational resources are needed for model training and evaluation.

## Features Included in Datasets

### Dataset 1: Flight Take-off Data from JFK Airport

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

### Dataset 2: FAA Aircraft Master File

| Feature                | Description                             | Feature                | Description                  |
|------------------------|-----------------------------------------|------------------------|------------------------------|
| N-Number (str)         | Identification number assigned to aircraft | Year Mfr (int)         | Year manufactured            |
| Aircraft Mfr Model Code (str) | Aircraft manufacturer, model, and series |                        |                              |

### Dataset 3: FAA Aircraft Reference File

| Feature                    | Description                                        | Feature              | Description                 |
|----------------------------|----------------------------------------------------|----------------------|-----------------------------|
| Aircraft Mfr Model Code (str) | Aircraft manufacturer, model, and series        | Aircraft Weight (int) | Maximum takeoff weight in pounds |
| Model Name (str)           | Aircraft model and series                         | Number of Seats (int) | Maximum number of seats in the aircraft |

### Data Merging Strategy

- **Join Dataset 1 and Dataset 2** on `TAIL_NUM` and `N-Number`.
- **Join Dataset 2 and Dataset 3** on `Aircraft Mfr Model Code`.

The target variable will be **DEP_DELAY** (Departure delay of the flight).

## Project Plan

| Task                       | Start Date   | End Date     |
|----------------------------|--------------|--------------|
| Data Pre-processing         | Feb 3rd      | Feb 14th     |
| Logistic Regression         | Feb 15th     | Feb 19th     |
| Random Forest               | Feb 20th     | Feb 24th     |
| Gradient Boosting Tree      | Feb 25th     | March 1st    |
| SVM                         | March 2nd    | March 6th    |
| K-Means Clustering          | March 7th    | March 11th   |
| Neural Networks             | March 12th   | March 16th   |
| Final Report                | March 17th   | March 25th   |

## Validations and Comparison

We will evaluate the models using an **80-20 train-test split** and **time series cross-validation**. Model performance will be compared using the following metrics:

- **F1-score**
- **ROC-AUC**
- **RMSE** (Root Mean Squared Error)

## Potential Risks

- **Overfitting:** The dataset is limited, which might lead to overfitting. If delayed flights are underrepresented, the model may become biased.
- **Computational Resources:** Complex models like neural networks require significant computational power, which may be a constraint.
- **Generalization:** The data comes from JFK Airport, so the model may not generalize well to other airports. Expanding the dataset would mitigate this risk.

## Conclusion

This project aims to develop a robust flight delay prediction system using multiple machine learning models. With the right datasets and evaluation techniques, the project will offer valuable insights into predicting flight delays, improving airport management, and enhancing the passenger experience.

