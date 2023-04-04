from flask import Flask, render_template, request, Markup
import pickle
import requests , json
import config
import numpy as np


crop_model = pickle.load(open('models/model.pkl', 'rb'))


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    complete_url= "https://api.openweathermap.org/data/2.5/weather?q="+ city_name + "&appid="+api_key
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None
    
app = Flask(__name__ ,template_folder= 'templates')


# index route
@ app.route('/')
def home():
    title = 'Harvestify - Home'
    return render_template('index.html', title=title)


@ app.route('/sdg')
def home():
    title = 'Harvestify - sdg'
    return render_template('sdg.html', title=title)




# render crop recommendation form page
@ app.route('/crop-recommend')
def crop_recommend():
    title = 'Harvestify - Crop Recommendation'
    return render_template('crop.html', title=title)

# render crop recommendation result page

@app.route('/crop-predict', methods=['GET', 'POST'])
def crop_prediction():
    title = 'CROP- Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        city = request.form.get("city")
        state = request.form.get("stt")
        result = weather_fetch(city)

        if result is not None:
            temperature, humidity = result
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_model.predict(data)
            final_prediction = my_prediction[0]
            return render_template('cropresult.html', prediction=final_prediction, title=title)
        else:
            return render_template('tryagain.html', title=title)


if __name__ == '__main__':
    app.run(debug=True)

