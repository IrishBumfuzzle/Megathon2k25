from keras.models import load_model
import os 
import cv2
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
from urllib.request import urlopen
import json
from geopy.geocoders import Nominatim

class_weather = ['Lightning', 'Rainy', 'Snow', 'Sunny']
class_ELA = ['Real', 'Tampered']

def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    ela_filename = 'temp_ela.png'
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)

    ela_image = ImageChops.difference(image, temp_image)

    extrema = ela_image.getextrema()
    max_diff = sum([ex[1] for ex in extrema])/3
    if max_diff == 0:
        max_diff = 1

    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    
    return ela_image

def prepare_image_for_ela(image_path):
    ela_img = convert_to_ela_image(image_path, 90)
    img = np.array(ela_img.resize((128,128))).flatten() / 255.0
    img = img.reshape(128,128,3)
    return np.expand_dims(img, axis=0), ela_img

# Functions
def check_img(image_name):
    img = cv2.resize(cv2.imread(image_name), (750, 750))
    return img

def detect_ELA(img_name):
    global ela_result
    np_img_input, ela_result = prepare_image_for_ela(img_name)
    ELA_model = load_model('models/model_ela.h5')
    Y_predicted = ELA_model.predict(np_img_input, verbose=0)
    return class_ELA[np.argmax(Y_predicted[0])], round(np.max(Y_predicted[0]) * 100)

weatherDict = {}
weatherDict[0] = 'Sunny'
for key in [1, 2]:
    weatherDict[key] = 'Clear or partly cloudy'
weatherDict[3] = 'Slight Rain'
for key in [45, 48]:
    weatherDict[key] = 'Fog'
for key in [51, 53, 55,56, 57]:
    weatherDict[key] = 'Slight Raining'
for key in [61, 63, 65,66, 67,80, 81, 82]:
    weatherDict[key] = 'Raining Raining'
for key in [45, 48,71, 73, 75,77,85, 86]:
    weatherDict[key] = 'Snow'
for key in [95,96,99]:
    weatherDict[key] = 'Lightning'

params = {}
base_url = 'https://archive-api.open-meteo.com/v1/era5?'

def get_weather(date_time,lat,long):
    #2022:12:01 12:09:65
    date = date_time[:10]
    time = date_time[11:]

    date = date.replace(':','-')
    params['Date'] = date
    params['Time'] = time
    params['Longitude'] = long
    params['Latitude'] = lat

    geoLoc = Nominatim(user_agent="GetLoc")
 
    # passing the coordinates
    locname = geoLoc.reverse("{},{}".format(lat,long))
    url = base_url + 'latitude={}&longitude={}&start_date={}&end_date={}&hourly=weathercode&timezone=Asia%2FBangkok'.format(params['Latitude'],params['Longitude'],params['Date'],params['Date'])

    response = urlopen(url)
    data_json = json.loads(response.read())

    weather_code = data_json['hourly']['weathercode']
    
    hour = int(time[:2])
    if weather_code[hour-1] is None:
        return locname, date, "NA"

    return locname, date, weatherDict[weather_code[hour-1]]

def prerpare_img_for_weather(image_path):
    img = np.array(Image.open(image_path).convert('RGB').resize((128,128)))/255.0
    img = img.reshape(128,128,3)
    return np.expand_dims(img, axis=0)

def detect_weather(img_name):
    np_img_input = prerpare_img_for_weather(img_name)
    model_Weather = load_model('models/Weather_Model.h5')
    Y_predicted = model_Weather.predict(np_img_input, verbose=0)
    return class_weather[np.argmax(Y_predicted[0])], round(np.max(Y_predicted[0]) * 100)

def org_weather(img_name, image_coords):
    global outdoor
    date_time, lat, long = image_coords
    # print(lat)
    location, date, weather = get_weather(date_time, lat, long)
    if lat == 0.0:
        return "", ""
    if weather == "NA":
        return location, ""
    return (location, weather)

def main_function(file_name, image_time, lat, long):
    # im = check_img(file_name)
    pred_t, acc_t = detect_ELA(file_name)

    image_coords = (image_time, lat, long)

    r_weath = org_weather(file_name, image_coords)
    pred_weath, acc_w = detect_weather(file_name)

    return (pred_t, acc_t), (pred_weath, acc_w), r_weath


