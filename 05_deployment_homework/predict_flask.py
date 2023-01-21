# load model and predict

import pickle

from flask import Flask
from flask import request
from flask import jsonify

# Parameters
model_name = 'model1.bin'
dv_name = 'dv.bin'

# load model
with open(model_name, 'rb') as f_in: 
    model = pickle.load(f_in)

# load  DictVectorizer
with open(dv_name, 'rb') as f_in: 
    dv = pickle.load(f_in)


# flask app
app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

	