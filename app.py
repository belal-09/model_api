import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

# 1. Flask App Initialization
app = Flask(__name__)
# Model loading setup
MODEL_PATH = 'job_posting_model.h5'
model = None

# Load the model once at startup
def load_saved_model():
    global model
    try:
        model = load_model(MODEL_PATH)
        # Warm-up the model to prevent issues during first prediction
        model._make_predict_function() 
        print(f"--- Model '{MODEL_PATH}' loaded successfully! ---")
    except Exception as e:
        print(f"!!! Error loading model: {e} !!!")
        
load_saved_model()


# 2. Prediction Endpoint 
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is currently unavailable on the server.'}), 500

    # Get JSON data from the request
    data = request.json
    input_data = data.get('input_data')
    
    if input_data is None:
        return jsonify({'error': 'Missing "input_data" key in JSON body.'}), 400

    try:
        # --- CRITICAL: Customize input processing here ---
        
        # 1. Convert to NumPy array
        input_array = np.array(input_data)
        
        # 2. Reshape the input for the model
        # You MUST adjust this reshape based on what your specific .h5 model expects.
        # Example: Reshape for one input sample, allowing NumPy to figure out feature count (-1)
        final_input_shape = input_array.reshape(1, -1) 
        
        # 3. Make the prediction
        prediction_result = model.predict(final_input_shape)
        
        # 4. Convert result to a JSON-compatible list
        output = prediction_result.tolist()
        
        # 5. Return the final prediction
        return jsonify({'prediction': output})

    except Exception as e:
        return jsonify({'error': f'Prediction or processing failed. Details: {e}'}), 400

# Local test run (Optional)
if __name__ == '__main__':
    app.run(debug=True, port=5000)