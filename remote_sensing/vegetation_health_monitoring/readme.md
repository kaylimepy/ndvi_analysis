# Vegetation Health Monitoring and Prediction

## Overview
This Python project aims to evaluate and predict the health of vegetation in a specific region, utilizing satellite data and machine learning techniques. The project employs NASA's EarthExplorer's Landsat 8 satellite imagery for the city of Tempe, Arizona, focusing on data from the year 2020.

## Methodology

1. **Data Acquisition:** Using the `landsatxplore` Python library, we interface with NASA's EarthExplorer API to retrieve Landsat 8 scenes that cover the specified location and date range.

2. **Data Processing:** Each scene is processed to calculate the Normalized Difference Vegetation Index (NDVI), a common indicator of plant health. The red and near-infrared (NIR) bands from each scene are leveraged for this purpose.

3. **Data Analysis:** After processing all scenes, we compute the average NDVI value for each scene and generate a time-series plot of these values to visualize changes in vegetation health over the year.

4. **Machine Learning Prediction:** We utilize the Scikit-learn library to develop a simple linear regression model that predicts future NDVI values based on previous ones. We assess the model's performance using metrics like Mean Squared Error and Coefficient of Determination (R² Score).

## Output
The output is a graphical visualization of the NDVI values over time, combined with the predictive regression line generated by the model. This time-series analysis allows us to track and predict changes in the health of vegetation over a certain period.

## Future Work
This project serves as a strong starting point for more comprehensive analyses and sophisticated methodologies:

1. **Land Cover Type Filtering:** While this project analyzes vegetation health for an entire region, future analyses could be more granular by specifically filtering for certain land cover types (like forests, croplands, etc.), providing more targeted insights into different types of vegetation health.

2. **Advanced Machine Learning Models:** While our project employs a simple linear regression model, more accurate predictions could be achieved by implementing more complex time-series forecasting models. For instance:

    - **ARIMA (Autoregressive Integrated Moving Average):** This is a class of models that 'explains' a given time series based on its own past values, i.e., its own lags and the lagged forecast errors.

    - **SARIMA (Seasonal Autoregressive Integrated Moving Average):** This model is an extension of ARIMA that supports univariate time series data with a seasonal component. It adds three new hyperparameters to specify the autoregression (AR), differencing (I), and moving average (MA) for the seasonal component of the series, as well as an additional parameter for the period of the seasonality.

    - **Prophet:** Prophet is a procedure for forecasting time series data. It is based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, plus holiday effects. Prophet is robust to missing data and shifts in the trend, and typically handles outliers well.

    - **LSTM (Long Short Term Memory):** LSTMs are a type of recurrent neural network that can learn and remember over long sequences and can model complex multivariate sequences of observations. This would involve transforming the problem into a supervised learning problem, scaling the input data, and framing the forecasting problem such that multiple time steps can be predicted.

3. **Inclusion of Ancillary Data:** Additional environmental data such as climate (temperature, precipitation) or human activity (land use change, urban development) could be integrated into the model to help explain and predict changes in vegetation health.

4. **Spatial Analysis:** Investigate spatial patterns and changes in NDVI over the study area. This could include identifying areas of greatest change or areas of consistently high or low NDVI.

These enhancements would provide more nuanced understanding and predictive capabilities for vegetation health in a given region, enhancing its value for various ecological and agricultural applications.

## Usage
Remove the extension `.example` from config.yml.example and update with usernane and password. 