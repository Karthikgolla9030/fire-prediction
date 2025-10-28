from flask import Flask, render_template, request
import numpy as np
import pickle

# Load models
ridge_model = pickle.load(open('models/ridgemodel.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scmodel.pkl', 'rb'))

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    # main page
    return render_template('home.html')

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    try:
        # Collect input features from the form
        features = [float(request.form.get(col)) for col in [
            'Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI'
        ]]

        # Transform using scaler
        scaled_features = standard_scaler.transform([features])

        # Predict using Ridge model
        prediction = ridge_model.predict(scaled_features)[0]

        # Interpret prediction (adjust logic as needed)
        final_pred = '🔥 High Fire Risk' if prediction > 0.5 else '🟢 Low Fire Risk'

        # Show result in same page
        return render_template('home.html', prediction=final_pred)

    except Exception as e:
        return render_template('home.html', prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)