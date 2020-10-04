import DCIS_grading_model as DCIS
import numpy as np

# PATH to the model weights file
MODEL_WEIGHTS_PATH = "...\\dcis_densenet_uncertainty_1.hdf5"

#Import the model and load the weights
model = DCIS.DCIS_model(IMAGE_SIZE=512)
model.load_weights(MODEL_WEIGHTS_PATH)

#Predict on a new image
YOUR_IMAGE_HERE = #Import your image here
YOUR_IMAGE_HERE = np.reshape(YOUR_IMAGE_HERE,(1,512,512,3)) #Make sure the size is correct
result = model.predict(YOUR_IMAGE_HERE/255, batch_size=1)

#Get the DCIS grade
DCIS_grade = np.argmax(result[0])+1
