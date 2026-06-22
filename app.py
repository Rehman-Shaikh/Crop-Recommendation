from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load models and data
model = pickle.load(open('models/model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
df = pickle.load(open('models/df.pkl', 'rb'))

crop_labels = df['label'].unique()
crop_descriptions = {
    'rice': 'Rice is a staple food for over half the world population. It grows best in warm, humid climates.',
    'maize': 'Maize (corn) is a versatile crop used for food, feed, and industrial products.',
    'chickpea': 'Chickpea is a nutrient-rich legume that improves soil fertility through nitrogen fixation.',
    'kidneybeans': 'Kidney beans are protein-rich legumes that grow well in moderate climates.',
    'pigeonpeas': 'Pigeon peas are drought-resistant legumes popular in tropical regions.',
    'mothbeans': 'Moth beans are drought-tolerant legumes grown in arid regions.',
    'mungbean': 'Mung beans are fast-growing legumes used for sprouts and dhal.',
    'blackgram': 'Black gram is a highly nutritious pulse crop grown in South Asia.',
    'lentil': 'Lentils are protein-rich legumes that grow well in cooler climates.',
    'pomegranate': 'Pomegranates are antioxidant-rich fruits grown in subtropical regions.',
    'banana': 'Bananas are tropical fruits rich in potassium and other nutrients.',
    'mango': 'Mangoes are tropical stone fruits known as the "king of fruits".',
    'grapes': 'Grapes are versatile fruits used for fresh consumption, raisins, and wine.',
    'watermelon': 'Watermelons are refreshing fruits with high water content.',
    'muskmelon': 'Muskmelons are sweet, aromatic fruits rich in vitamins A and C.',
    'apple': 'Apples are temperate fruits known for their health benefits and long storage life.',
    'orange': 'Oranges are citrus fruits rich in vitamin C and antioxidants.',
    'papaya': 'Papayas are tropical fruits with digestive enzymes and high vitamin C content.',
    'coconut': 'Coconuts are versatile tropical palms providing food, oil, and fiber.',
    'cotton': 'Cotton is a fiber crop grown for textile production.',
    'jute': 'Jute is a long, soft fiber used for making burlap and hessian.',
    'coffee': 'Coffee is a tropical crop grown for its beans used to make beverages.'
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])
        
        # Scale the input
        input_data = scaler.transform([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        
        # Get probabilities for all crops
        probabilities = model.predict_proba(input_data)[0]
        
        # Pair crop names with their probabilities
        crop_probs = list(zip(crop_labels, probabilities))
        
        # Sort by probability (descending)
        crop_probs_sorted = sorted(crop_probs, key=lambda x: x[1], reverse=True)
        
        # Get the top recommended crop
        predicted_crop = crop_probs_sorted[0][0]
        crop_desc = crop_descriptions.get(predicted_crop.lower(), 'No description available.')
        
        return render_template('result.html', 
                             prediction=predicted_crop,
                             description=crop_desc,
                             crop_probs=crop_probs_sorted,  # Pass all probabilities to template
                             nitrogen=nitrogen,
                             phosphorus=phosphorus,
                             potassium=potassium,
                             temperature=temperature,
                             humidity=humidity,
                             ph=ph,
                             rainfall=rainfall)

if __name__ == '__main__':
    app.run(debug=True)