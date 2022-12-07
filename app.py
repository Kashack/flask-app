from flask import Flask, jsonify,request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/', methods=['POST'])
def predict():
    # input_data = (1,79.0,0,1,2,0,174.12,74,2,1)
    input_data = request.get_json(force=True)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = model.predict(input_data_reshaped)
    # data = request.get_json(force=True)
    # prediction = model.predict([[np.array(data['exp'])]])
    # output = prediction[0]

    if(prediction[0] == 0):
        output = "Low blood preassure"
    else:
        output = "High blood preassure"

    return jsonify(output)

if __name__ == '__main__':
    app.run(debug = True)