# Dataset:

Earthquake dataset from https://earthquake.usgs.gov/earthquakes/search/ \
The data will be called by this link: https://earthquake.usgs.gov/fdsnws/event/1/query.csv?starttime=2023-04-17&endtime=2024-04-17&minmagnitude=2.5&maxmagnitude=5&orderby=time&limit=20000

The minimum magnitude will be 2.5 and the maximum will be 5.0, like this. \
The end time will be the date when the data gets pulled, and the start time will be 1 year before. \
As only, 20000 rows can be pulled from the API, the limit will be on 20000. This is around 8 months of data.

The training data will be from the start time when the data gets called to 1 year before, the test data will be from 1 year before the training data and the validation data will be 2 years before the training data.

 training data:   2023-04-17 -- 2024-04-17 \
 test data:       2022-04-17 -- 2023-04-17 \
 validation data: 2021-04-17 -- 2022-04-17


API Documentation: https://earthquake.usgs.gov/fdsnws/event/1/

### Example:

starttime:Date = 2023-04-17 \
endtime:Date = 2024-04-17 \
minmagnitude:Float = 2.5 \
maxmagnitude:Float = 5 \
limit:Int = 20000

# Explanation:

We will make a XGBClassifier model to predict the magnitude based on the latitude and longitude. The user will be able to give a location and see what magnitude earthquake would happen at this location.

# Flows & Actions:

### Collection flow:

Will collect 20000 rows from the day when the API call gets made. This data will be stored and used to train and validated the model

### Preprocessing flow:

Drop unnecessary columns from the database and normalize the data.

### Training flow:

Building the basic model for quick testing.

### Register flow:

Building the model with a grid search and saving the best model.
