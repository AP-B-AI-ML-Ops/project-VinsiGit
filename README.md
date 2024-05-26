# how do run:
1. Install the required dependencies by running the command: `pip install -r requirements.txt`
2. Start the MLflow UI with the command: `mlflow ui --backend-store-uri sqlite:///mlflow.db`
3. Start the Prefect server with the command: `prefect server start`
4. Run the main flow with the command `python main.py`


# Dataset:

Earthquake dataset from https://earthquake.usgs.gov/earthquakes/search/ \
The data will be called by this link: https://earthquake.usgs.gov/fdsnws/event/1/query.csv?starttime=2023-04-17&endtime=2024-04-17&minmagnitude=2.5&maxmagnitude=5&orderby=time&limit=20000

The minimum magnitude will be 2.5 and the maximum will be 5.0, like this. \
The end time will be the date when the data gets pulled, and the start time will be 1 year before. \
As only, 20000 rows can be pulled from the API, the limit will be on 20000. This is around 8 months of data.

The training data will be from the start time when the data gets called to 1 year before, the test data will be from 1 year before the training data and the validation data will be 2 years before the training data.

 training data:   2023-01-01 -- 2023-12-31 \
 test data:       2022-01-01 -- 2022-12-31 \
 validation data: 2021-01-01 -- 2021-12-31


API Documentation: https://earthquake.usgs.gov/fdsnws/event/1/

### Example:

starttime:Date = 2023-01-01 \
endtime:Date = 2023-12-31 \
minmagnitude:Float = 2.5 \
maxmagnitude:Float = 5 \
limit:Int = 20000

# Explanation:

We will make a XGBRegressor model to predict the magnitude based on the latitude and longitude. The user will be able to give a location and see what magnitude earthquake would happen at this location.

# Flows & Actions:

### Collection flow:

Will collect 20000 rows from the day when the API call gets made. This data will be stored and used to train and validated the model

### Preprocessing flow:

Drop unnecessary columns from the database and normalize the data.

### Training flow:

Building the basic model for quick testing.

### Register flow:

Building the model with a grid search and saving the best model.
