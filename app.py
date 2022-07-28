from flask import Flask, render_template, request
from keras.models import load_model
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

dic = {0:'T-shirt/top',1:'Trouser',2:'Pullover',3:'Dress',4:'Coat',5:'Sandal',6:'Shirt',7:'Sneaker',8:'Bag',9:'Ankle boot'}

model = load_model('fashion_cnn_model.h5')


def predict_label(img_path):
	i = tf.keras.preprocessing.image.load_img(img_path, target_size=(28,28,1))
	i = tf.keras.preprocessing.image.img_to_array(i)/255.0
	i = i.reshape(-1,28,28,1)
	p = np.argmax(model.predict(i), axis=-1)
	return dic[p[0]]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

# @app.route("/about")
# def about_page():
# 	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/uploads/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)