from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import urllib.request
from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle
import util
from FaceComparison import compute_facial_datapoints
from scipy.spatial.distance import cosine
import os
import cv2

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
#
UPLOAD_FOLDER = 'trial/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
#

from mtcnn.mtcnn import MTCNN
MTCNN = MTCNN()


def perform_predictions(uno,dos):
    image_uno = plt.imread(uno)
    image_dos = plt.imread(dos)
    util.draw_box(uno, MTCNN.detect_faces(image_uno), 1)
    util.draw_box(dos, MTCNN.detect_faces(image_dos), 2)
    extracted_face_uno = util.prep_image(uno)
    extracted_face_dos = util.prep_image(dos)

    plt.figure(figsize=(8, 8))  # specifying the overall grid size

    finalImage = []
    imageSimilarityScores = []

    model_scores_uno = compute_facial_datapoints(extracted_face_uno)
    model_scores_dos = compute_facial_datapoints(extracted_face_dos)

    for x, face_score_1 in enumerate(model_scores_uno):
        for y, face_score_2 in enumerate(model_scores_dos):
            score = cosine(face_score_1, face_score_2)

            if score <= .4:
                # Printing the IDs of faces and score
                print(x, y, score)
                # Displaying each matched pair of faces
                finalImage.append(extracted_face_uno[x])
                # plt.imshow(extracted_face_uno[x])
                # plt.show()
                finalImage.append(extracted_face_dos[y])
                imageSimilarityScores.append((1 - score) * 100)
                imageSimilarityScores.append((1 - score) * 100)
                # plt.imshow(extracted_face_dos[y])
                # plt.show()

    print(len(finalImage))
    for i in range(len(finalImage)):
        plt.subplot(len(finalImage)/2, 2, i + 1)
        # the number of images in the grid is 5*5 (25)
        plt.imshow(finalImage[i])
        plt.title(imageSimilarityScores[i])
        plt.axis('off')
    plt.savefig('static/result.png')
    plt.show()


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        files = request.files.getlist("file")
        count = 0
        for file in files:
            if count==0:
                file.filename="static/f1.jpg"
            else:
                file.filename="static/s1.jpg"
            file.save(file.filename)
            count+=1
        # filename = secure_filename(f.filename)
        # f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        perform_predictions("static/f1.jpg","static/s1.jpg")
        # img3 = cv2.imread('static/first.jpg')
        # cv2.imwrite('trial/afterI.jpg', img3)
        # img2 = cv2.imread('trial/second.jpg')
        # cv2.imwrite('trial/afterI2.jpg', img2)
        return render_template('after.html')



if __name__ == '__main__':
    app.run(debug=True)
@app.route('/')
def hello_world():

    return render_template('before.html')


if __name__ == '__main__':
    app.run(debug=True)


@app.route('/predict', methods=['POST'])
def home():


    return render_template('after.html')