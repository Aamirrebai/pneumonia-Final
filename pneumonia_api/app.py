from flask import Flask, request, render_template
from PIL import Image
import json
import numpy as np
from keras_preprocessing.image.utils import img_to_array
import pandas 
from keras.models import load_model
import os 

app = Flask(__name__)
model = load_model("model.h5")


@app.route("/", methods = ['GET', 'POST'])
def index():
     return render_template('home.html')

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
     #imagefile = request.files["imagefille"]
     print(os.getcwd())
     print(os.listdir("resize_normal"))
     path_image = "resize_normal/_7_1509590.png"



     image = Image.open(path_image)
     size = (256, 256)
     image = image.resize(size)
     image = img_to_array(image) / 256
     image = np.expand_dims(image, 0)
     print(image.shape)
     prediction = model.predict(image)

     print(type(image))
     print(image)


     print(prediction)



     return 0
     



 
if __name__ == "__main__":
    app.run(debug=True)