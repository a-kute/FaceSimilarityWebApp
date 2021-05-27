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
from PIL import Image
from numpy import asarray

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1
#
UPLOAD_FOLDER = 'trial/'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



#


def store_image_url(image_url, file_destination):
    with urllib.request.urlopen(image_url) as resource:
        with open(file_destination,'wb') as filewriter:
            filewriter.write(resource.read())

def draw_box(image_path, faces, number):
    #
    img1 = cv2.imread(image_path)


    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(img1, (x, y), (x + w, y + h), (0, 0, 255), 2)



    #


    # image = plt.imread(image_path)
    # plt.imshow(image)
    # ax = plt.gca()
    # for face in faces:
    #     x,y,w,h = face['box']
    #     face_border = Rectangle((x,y),w,h,fill=False,color='red')
    #     ax.add_patch(face_border)
    #
    # plt.axis('off')
    if number==1:
        cv2.imwrite('static/a1.jpg', img1)
    else:
        cv2.imwrite('static/a2.jpg', img1)
    plt.clf()


def prep_image(image_path, required_size = (224,224)):
    from mtcnn.mtcnn import MTCNN

    # set the base width of the result
    basewidth = 300
    img = Image.open(image_path)
    # determining the height ratio

    # resize image and save
    img = img.resize((640, 480), Image.ANTIALIAS)
    img.save(image_path)
    image = plt.imread(image_path)



    # face_image = Image.fromarray(face_boundary)
    # face_image = face_image.resize(required_size)
    # face_array = asarray(face_image)
    # face_images.append(face_array)



    MTCNN = MTCNN()
    faces = MTCNN.detect_faces(image)
    face_images = []
    uno = image.shape[0]
    dos = image.shape[1]


    for face in faces:
        x1,y1,w1,h1 = face['box']
        x2,y2 = x1+w1,y1+h1


        if x2>dos:
           x2=dos
        if y2>uno:
            y2=uno
        if x1<0:
            x1=0

        print(x1)
        print(x2)
        print(y1)
        print(y2)

        face_boundary = image[y1:y2,x1:x2]
        face_image = Image.fromarray(face_boundary)
        face_image = face_image.resize(required_size)
        face_array = asarray(face_image)
        face_images.append(face_array)

    return face_images




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