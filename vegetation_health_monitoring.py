import rasterio
import numpy as np
import matplotlib.pyplot as plt
from landsatxplore.api import API
from landsatxplore.earthexplorer import EarthExplorer
from glob import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import yaml 

config = yaml.load(open('config.yml'), Loader=yaml.FullLoader)


def download_scenes(api, earth_explorer, latitude, longitude, start_date, end_date, max_cloud_cover, output_dir):
    scenes = api.search(
        dataset='LANDSAT_8_C1',
        latitude=latitude,
        longitude=longitude,
        start_date=start_date,
        end_date=end_date,
        max_cloud_cover=max_cloud_cover
    )

    print(f"{len(scenes)} scenes found.")

    for scene in scenes:
        earth_explorer.download(scene_id=scene['entity_id'], output_dir=output_dir)

    return scenes


def calculate_ndvi(scenes, output_dir):
    ndvi_values = []
    for scene in scenes:
        with rasterio.open(glob(f"{output_dir}/{scene['entity_id']}/*_B4.TIF")[0]) as red_band, \
             rasterio.open(glob(f"{output_dir}/{scene['entity_id']}/*_B5.TIF")[0]) as nir_band:
            red = red_band.read(1)
            nir = nir_band.read(1)

        ndvi = (nir.astype(float) - red) / (nir + red)
        ndvi_values.append(np.nanmean(ndvi))

    return ndvi_values


def train_regression_model(x, y):
    model = LinearRegression()
    model.fit(x, y)
    return model


def evaluate_regression_model(model, x, y):
    y_pred = model.predict(x)
    mean_squared_error = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mean_squared_error, r2

def plot_results(x, y, model):
    plt.scatter(x, y, color='black')
    plt.plot(x, model.predict(x), color='blue', linewidth=3)

    plt.xlabel('Time (in arbitrary units)')
    plt.ylabel('NDVI')
    plt.title('NDVI over Time')
    plt.grid(True)
    plt.show()


def main():
    # Initialize a new API instance and get an access key
    api = API(config['usgs']['username'], config['usgs']['password'])

    # Define the region (Tempe, Arizona)
    latitude, longitude = 33.4255, -111.94

    # Define the date interval
    start_date = '2020-01-01'
    end_date = '2020-12-31'

    # Output path for the downloaded scenes
    output_dir = './temp/vegetation_health_monitoring/landsat_scenes/'

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the EarthExplorer instance
    earth_explorer = EarthExplorer(config['earth_explorer']['username'], config['earth_explorer']['password'])

    # Download the scenes
    scenes = download_scenes(api, earth_explorer, latitude, longitude, start_date, end_date, 10, output_dir)

    # Calculate NDVI values
    ndvi_values = calculate_ndvi(scenes, output_dir)

    # Convert time points and NDVI values to numpy arrays
    time_points = np.array(range(1, len(ndvi_values) + 1)).reshape(-1, 1)
    ndvi_values = np.array(ndvi_values).reshape(-1, 1)

   
    # Split the data into training and testing datasets
    x_train, x_test, y_train, y_test = train_test_split(time_points, ndvi_values, test_size=0.2, random_state=42)

    # Train the regression model
    model = train_regression_model(x_train, y_train)

    # Evaluate the regression model
    mse, r2 = evaluate_regression_model(model, x_test, y_test)

    # Print the coefficients, mean squared error, and coefficient of determination
    print('Coefficients: \n', model.coef_)
    print('Mean squared error: %.2f' % mse)
    print('Coefficient of determination: %.2f' % r2)

    # Plot the results
    plot_results(time_points, ndvi_values, model)

    # Close the API and EarthExplorer instances
    api.logout()
    earth_explorer.logout()


if __name__ == "__main__":
    main()
