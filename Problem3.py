#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Tue Feb  4 20:59:11 2019

@author: yichuanniu
'''

import pandas as pd
import googlemaps
from multiprocessing import Pool
import numpy as np
import folium
import geopandas as gpd
from geopandas.geoseries import Point
from us_state_abbrev import us_state_abbrev

working_dir = "/Users/yichuanniu/Downloads/uber_exercise_yichuan_niu"
state_map = gpd.read_file(working_dir + "/cb_2017_us_state_500k/cb_2017_us_state_500k.shp")


def parallelize_dataframe(df, func, num_partitions, num_threads):
    """ Parallel processing for State count """
    df_split = np.array_split(df, num_partitions)
    pool = Pool(num_threads)
    result = pool.map(func, df_split)
    pool.close()
    pool.join()
    return result


def get_state_count(data):
    ''' Get state widgets from a single process '''
    state_count = {}
    for index in list(data.index):
        if index % 1000 == 0:
            print("processing at:", index)
        try:
            point = Point(data.loc[index, "lng"], data.loc[index, "lat"])
            for index1 in list(state_map.index):
                state_polygon = state_map.loc[index1, "geometry"]
                state_name = state_map.loc[index1, "NAME"]
                if point.within(state_polygon):
                    if state_count.get(state_name) != None:
                        state_count[state_name] += data.loc[index, "widgets"]
                    else:
                        state_count[state_name] = data.loc[index, "widgets"]
                    break
        except:
            print("ERROR in index: ", index, data.loc[index, "lng"])
    return state_count


def get_state_count_google(data):
    ''' This is an alternative solution to get state info from Google geocoding services '''
    google_apikey = "Your_Google_Api_Key_Here"
    gmaps = googlemaps.Client(key = google_apikey)
    state_count = {}
    for index in list(data.index):
#        if index % 1000 == 0:
#            print("processing at:", index)
        try:
            reverse_geocode_result = gmaps.reverse_geocode((data.loc[index, "lat"],
                                                            data.loc[index, "lng"]))
            for rgc in reverse_geocode_result:
                if rgc.get("types") != None and len(rgc.get("types") ) > 1 and rgc.get("types")[0] == "administrative_area_level_1":
                    state_name = rgc.get("address_components")[0].get("long_name")
                    if state_count.get(state_name) != None:
                        state_count[state_name] += data.loc[index, "widgets"]
                    else:
                        state_count[state_name] = data.loc[index, "widgets"]
        except:
            print("ERROR in index: ", index)
    return state_count


def state_count_merger(count_list):
    ''' Merging result from multi-threaded results '''
    state_count = {}
    for element in count_list:
        for key in element:
            if state_count.get(key) != None:
                state_count[key] += element[key]
            else:
                state_count[key] = element[key]

    return state_count




if __name__ == "__main__":

    # Read in data
    raw_data = pd.read_csv(working_dir + "/exercise_final.csv")
    widgets = raw_data["widgets"]

    # Caldulate mean, median, and 75th - 25th difference from raw data    
    print("widgets raw data mean:", widgets.mean())
    print("widgets raw data median:", widgets.median())
    print("widgets raw data 75th to 25th diff:", widgets.quantile(0.75) - widgets.quantile(0.25), "\n")
    
    # Filtering data
    print("Minimum widgets:", min(widgets))
    print("Maximum widgets:", max(widgets))
    quantile_0_005, quantile_0_995 = widgets.quantile(0.005), widgets.quantile(0.995)
    print("widgets 0.005 quantile:", quantile_0_005)
    print("widgets 0.995 quantile:", quantile_0_995, "\n")
    
    '''
    Based on teh 0.5% and 99.5% quantile, decide to drop widgets 
    below value 0.5%(0.10537) and larger than 99.5%(9288.83183)
    '''
    filtered_data = raw_data[raw_data["widgets"] >= quantile_0_005]
    filtered_data = raw_data[raw_data["widgets"] <= quantile_0_995]
    filtered_widgets = filtered_data["widgets"]
    
    # Re-caldulate mean, median, and 75th - 25th difference from filtered data    
    print("widgets filtered data mean:", filtered_widgets.mean())
    print("widgets filtered data median:", filtered_widgets.median())
    print("widgets filtered data 75th to 25th diff:", filtered_widgets.quantile(0.75) - filtered_widgets.quantile(0.25))
    

    # Perform multiprocess on data to calculated total widgets based on state
    num_partitions, num_threads = 16, 16
    
    # Using Polygon
    count_list = parallelize_dataframe(filtered_data, get_state_count, num_partitions, num_threads)
    
    # Using Google Reverse Geocoding for State Info
#    count_list = parallelize_dataframe(filtered_data, get_state_count_google, num_partitions, num_cores)
    
    # Merging result from multiprocess
    state_count = state_count_merger(count_list)
    
    # Export and save result
    state_count_df = pd.DataFrame(columns = ["state", "total_widgets"])
    index = 0
    for key in state_count:
        state_count_df.loc[index] = [key, state_count[key]]
        index += 1
    state_count_df.sort_values(by = ["state"], inplace = True)
    state_count_df.to_csv(working_dir + "/state_widgets_count.csv", index = False)
    

    '''
    Plotting choropleth graph
    '''
    # Load the shape of the zone (US states)
    state_geo = working_dir + "/folium/us-states.json"
     
    # Load the unemployment value of each state
    state_data = pd.read_csv(working_dir + "/state_widgets_count.csv")
    for index in list(state_data.index):
        state_data.loc[index, "state"] = us_state_abbrev.get(state_data.loc[index, "state"])
        state_data.loc[index, "total_widgets"] /= 1000000.0
        
    
    # Initialize the map:
    map = folium.Map(location=[37, -102], zoom_start=5)
     
    # Add the color for the chloropleth:
    map.choropleth(
     geo_data=state_geo,
     name='choropleth',
     data=state_data,
     columns=['state', 'total_widgets'],
     key_on='feature.id',
     fill_color='YlOrRd',
     fill_opacity=0.7,
     line_opacity=0.2,
     legend_name='State Total Widgets (Million)',
     bins = list(state_data['total_widgets'].quantile([0, 0.3, 0.6, 0.75, 0.9, 0.95, 0.99, 1]))
    )
    folium.LayerControl().add_to(map)
     
    # Save to html
    map.save(working_dir + "/State_Total_Widgets.html")
