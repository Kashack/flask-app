from flask import Flask, jsonify,request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/predict',methods=['POST'])

def predict():
    # input_data = (60182,0,49,0,1,2,1,171.23,34.4,3,1)
    input_data = request.get_json(force=True)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = model.predict(input_data_reshaped)

    # data = request.get_json(force=True)
    # prediction = model.predict([[np.array(data['exp'])]])
    output = prediction[0]

    if(output == 0):
        print("Low blood preassure")
    else:
        print("High blood preassure")

    return jsonify(output)

def index():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug = True)