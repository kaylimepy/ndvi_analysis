# Vegetation Health Monitoring and Prediction

## Overview
This Python project aims to evaluate and predict the health of vegetation in a specific region, utilizing satellite data and machine learning techniques. The project employs NASA's EarthExplorer's Landsat 8 satellite imagery for the city of Tempe, Arizona, focusing on data from the year 2020.

## Methodology

1. **Data Acquisition:** Using the `landsatxplore` Python library, we interface with NASA's EarthExplorer API to retrieve Landsat 8 scenes that cover the specified location and date range. We then download the corresponding `.tar` files and extract them.

2. **Data Processing:** Each scene is processed to calculate the Normalized Difference Vegetation Index (NDVI), a common indicator of plant health. The red and near-infrared (NIR) bands from each scene are leveraged for this purpose.

3. **Data Analysis:** After processing all scenes, we compute the average NDVI value for each scene and generate a time-series plot of these values to visualize changes in vegetation health over the year.

4. **Machine Learning Prediction:** We utilize LSTM model to forecast future NDVI values. LSTM, a type of recurrent neural network, works particularly well with sequential data. We assess the model's performance using metrics such as Root Mean Squared Error (RMSE).

## Output
The output is a graphical visualization of the NDVI values over time, combined with the predictive models' forecasts. This time-series analysis allows us to track and predict changes in the health of vegetation over a certain period.

## Future Work
This project serves as a strong starting point for more comprehensive analyses and sophisticated methodologies:

1. **Land Cover Type Filtering:** Future analyses could be more granular by specifically filtering for certain land cover types (like forests, croplands, etc.), providing more targeted insights into different types of vegetation health.

2. **Advanced Machine Learning Models:** More accurate predictions could be achieved by implementing more complex time-series forecasting models such as ARIMA (Autoregressive Integrated Moving Average), SARIMA (Seasonal Autoregressive Integrated Moving Average), Prophet, and LSTM (Long Short Term Memory).

3. **Inclusion of Ancillary Data:** Additional environmental data such as climate (temperature, precipitation) or human activity (land use change, urban development) could be integrated into the model to help explain and predict changes in vegetation health.

4. **Spatial Analysis:** Investigate spatial patterns and changes in NDVI over the study area. This could include identifying areas of greatest change or areas of consistently high or low NDVI.

## Usage
Remove the extension `.example` from `config.yml.example` and update with your username and password. Run `download.py` to retrieve and extract the necessary Landsat 8 scenes. Next, execute the Jupyter notebook for data processing, analysis, and machine learning predictions.
