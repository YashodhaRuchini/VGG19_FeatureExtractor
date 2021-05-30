from keras.applications.vgg19 import VGG19
from keras.models import Model
from pickle import dump
from keras.applications.vgg19 import decode_predictions
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

source_img = load_img('Tiger.jpg', target_size=(224, 224))
source_img = img_to_array(source_img)
source_img = source_img.reshape((1, source_img.shape[0], source_img.shape[1], source_img.shape[2]))
source_img = preprocess_input(source_img)

model = VGG19()
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

imageFeatures = model.predict(source_img)

dump(imageFeatures, open('Tiger_ImageFeatures.pkl', 'wb'))