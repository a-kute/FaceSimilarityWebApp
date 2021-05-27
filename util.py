import urllib.request

from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from numpy import asarray
import cv2


def store_image_url(image_url, file_destination):
    with urllib.request.urlopen(image_url) as resource:
        with open(file_destination,'wb') as filewriter:
            filewriter.write(resource.read())

def draw_box(image_path, faces, number):
    #
    img1 = cv2.imread(image_path)


    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(img1, (x, y), (x + w, y + h), (255, 0, 0), 1)



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


        if x2>uno:
            continue
        if y2>dos:
            continue
        if x1<0:
            continue

        face_boundary = image[y1:y2,x1:x2]
        face_image = Image.fromarray(face_boundary)
        face_image = face_image.resize(required_size)
        face_array = asarray(face_image)
        face_images.append(face_array)

    return face_images





