from flask import Flask , request, url_for, render_template
import numpy as np
import pickle



model = pickle.load(open('model1.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index1.html')

@app.route('/predict', methods=['POST'])
def predict ():
    inputs = [float(x) for x in request.form.values()]
    inputs = np.array([inputs])
    inputs = (inputs)
    output = model.predict(inputs)
    if output < 0.5:
        output = 0 
    else:
        output = 1
    return render_template('results.html' , prediction = output)

if __name__ =='__main__':
    app.run(debug=True)