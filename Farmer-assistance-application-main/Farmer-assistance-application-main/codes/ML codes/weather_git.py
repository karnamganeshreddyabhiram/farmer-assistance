frequency=3
start_date = '11-Dec-2019'
end_date = '11-Mar-2020'
api_key = 'xxxxxxxxxxxxxxxxxxxxx'
location_list = ['singapore']

from wwo_hist import retrieve_hist_data
hist_weather_data = retrieve_hist_data(api_key,
                                location_list,
                                start_date,
                                end_date,
                                frequency,
                                location_label = False,
                                export_csv = True,
                                store_df = True)

import pandas as pd
df=pd.read_csv('nellore.csv')
