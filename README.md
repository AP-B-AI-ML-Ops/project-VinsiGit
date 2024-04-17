# Dataset:

Earthquake dataset from https://earthquake.usgs.gov/earthquakes/search/ \
The data will be called by this link: https://earthquake.usgs.gov/fdsnws/event/1/query.csv?starttime=2023-04-17&endtime=2024-04-17&minmagnitude=2.5&maxmagnitude=5&orderby=time&limit=20000

The minimum magnitude will be 2.5 and the maximum will be 5.0, like this. \
The end time will be the date when the data gets pulled and the start time will be 1 year before. \
As only 20000 rows can be pulled from the the limit will be on 20000. This is around 8 months of data.

The training data will be from the starttime when the data gets called, the test data will be one year before the starttime and the validation data will be 2 years before the starttime.

API Documentation: https://earthquake.usgs.gov/fdsnws/event/1/

### example:

starttime:Date = 2023-04-17 \
endtime:Date = 2024-04-17 \
minmagnitude:Float = 2.5 \
maxmagnitude:Float = 5 \
limit:Int = 20000

# Explanation:

We will make a XGBClassifier model to predict the magitute based on the latitude and longitude. The user will able to give a location and see what magnitute earthquake would happen at this location.

# Flows & Actions:

### Collection flow:

Will collect 20000 rows from the day when the API call gets made. This data will be stored and used to train and validated the model

### Preproccesing flow:

Drop unnecessary columns from the database and normalize the data.

### Training flow:

Building the basic model for quick testing.

### register flow:

Building the model with a gridseach and saving the best model.
