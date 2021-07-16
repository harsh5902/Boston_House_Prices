from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

# Gather Data
boston_dataset = load_boston()
data = pd.DataFrame(data=boston_dataset.data, columns=boston_dataset.feature_names)
features = data.drop(['INDUS', 'AGE'], axis=1)

log_prices = np.log(boston_dataset.target)
target = pd.DataFrame(log_prices, columns=['PRICE'])

CRIM_IDX = 0
ZN_IDX = 1
CHAS_IDX = 2
RM_IDX = 4
PTRATIO_IDX = 8
ZILLOW_MEDIAN_PRICE = 583.3
SCALE_FACTOR = ZILLOW_MEDIAN_PRICE/np.median(boston_dataset.target)

property_stats = np.ndarray(shape=(1, 11))
property_stats = features.mean().values.reshape(1, 11)

regr = LinearRegression().fit(features, target)
fitted_vals = regr.predict(features)

MSE = mean_squared_error(target, fitted_vals)
RMSE = np.sqrt(MSE)

############################################################################################################################################

def get_log_estimate(nr_rooms,
                    students_per_classroom,
                    next_to_river=False,
                    high_confidence=True):
    
    #Configure Property
    property_stats[0][RM_IDX] = nr_rooms
    property_stats[0][PTRATIO_IDX] = students_per_classroom
    
    if next_to_river:
        property_stats[0][CHAS_IDX] = 1
    else:
        property_stats[0][CHAS_IDX] = 0
    
    
    #Make Prediction
    log_estimate = regr.predict(property_stats)[0][0]
    
    #Calc Range
    if high_confidence:
        upper_bound = log_estimate + 2*RMSE
        lower_bound = log_estimate - 2*RMSE
        interval= 95
    else:
        upper_bound = log_estimate + RMSE
        lower_bound = log_estimate - RMSE
        interval = 68
    
    return log_estimate, upper_bound, lower_bound, interval

############################################################################################################################################

def get_dollar_estimate(rm, ptratio, chas=False, large_range=True):
    
    """
    Keyword Arguments:
    rm -- number of rooms in a property
    ptratio -- pupil per teacher in the locality of the property
    chas -- True if property is near Charles river, False otherwise
    large_range -- True for 95% prediction interval, False for 68% prediction interval
    
    
    """
    
    
    
    if rm<1 or ptratio<1:
        print('That is unrealistic, please try again')
        return
    
    estimated_log_price, log_upper_bound, log_lower_bound, interval = get_log_estimate(rm, ptratio, next_to_river=chas,
                                                                                   high_confidence=large_range)
    
    estimated_price = round((np.e**estimated_log_price)*1000)                           #In 1970
    estimated_price_latest = np.around(estimated_price * SCALE_FACTOR, -3)                             #In 2020

    upper_bound = round((np.e**log_upper_bound)*1000)                                   #In 1970
    upper_bound_latest = np.around(upper_bound * SCALE_FACTOR, -3)                                     #In 2020

    lower_bound = round((np.e**log_lower_bound)*1000)                                   #In 1970
    lower_bound_latest = np.around(lower_bound*SCALE_FACTOR, -3)                                       #In 2020

    print('Price in 2020 for house esimate: $',estimated_price_latest)
    print('Price in 2020 for maximum house price: $',upper_bound_latest)
    print('Price in 2020 for minimum house price: $',lower_bound_latest)


