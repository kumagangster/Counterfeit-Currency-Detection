from flask import Flask,render_template,request,redirect,url_for
from tensorflow.keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import os

UPLOAD_FOLDER = 'static/file/'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload',methods=['POST','GET'])
def upload():
    if request.method == 'POST':

        classes = ['Fake', 'Real']
        

        file1 = request.files['filename']
        imgfile = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file1.save(imgfile)
        model = load_model('model_inception.h5')
        #model = load_model('model.hdf5')
        img_ = image.load_img(imgfile, target_size=(224, 224, 3))
        img_array = image.img_to_array(img_)
        img_processed = np.expand_dims(img_array, axis=0)
        img_processed /= 255.
        prediction = model.predict(img_processed)
        index = np.argmax(prediction)
        result = str(classes[index]).title()
        percentage = round(float(prediction[0][index] * 100), 2)
        
        return render_template('index.html', msg = result, src = imgfile, view = 'style=display:block', view1 = 'style=display:none')
    


if __name__ == '__main__':
    app.run(debug=True)
