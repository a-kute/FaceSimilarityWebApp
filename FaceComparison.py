from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
import urllib.request

from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from numpy import asarray

def compute_facial_datapoints(faces):
    samples = asarray(faces, 'float32')

    # prepare the data for the model
    samples = preprocess_input(samples, version=2)

    # create a vggface model object
    model = VGGFace(model='resnet50',
                    include_top=False,
                    input_shape=(224, 224, 3),
                    pooling='avg')

    # perform prediction
    return model.predict(samples)
