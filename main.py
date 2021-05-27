#
#
# import urllib.request
# from matplotlib import pyplot as plt
# from mtcnn.mtcnn import MTCNN
# from matplotlib.patches import Rectangle
# import util
# from FaceComparison import compute_facial_datapoints
# from scipy.spatial.distance import cosine
#
# ##user input
#
#
# url1 = input("enter the url of the first image: ")
# url2 = input("enter the url of the second image: ")
# if url1[-1]=="'":
#     url1 = url1[:-1]
# if url2[-1]=="'":
#     url2 = url2[:-1]
#
# print(url1)
# print(url2)
#
# util.store_image_url(url1,'12.jpg')
# util.store_image_url(url2,'13.jpg')
# image_uno = plt.imread('12.jpg')
# image_dos = plt.imread('13.jpg')
#
#
#
# MTCNN = MTCNN()
# util.draw_box('12.jpg', MTCNN.detect_faces(image_uno))
# util.draw_box('13.jpg', MTCNN.detect_faces(image_dos))
# extracted_face_uno = util.prep_image('12.jpg')
# extracted_face_dos = util.prep_image('13.jpg')
#
#
#
#
#
# plt.figure(figsize=(8,8)) # specifying the overall grid size
#
#
# finalImage = []
# imageSimilarityScores = []
#
# model_scores_uno = compute_facial_datapoints(extracted_face_uno)
# model_scores_dos = compute_facial_datapoints(extracted_face_dos)
# for x, face_score_1 in enumerate(model_scores_uno):
#   for y, face_score_2 in enumerate(model_scores_dos):
#     score = cosine(face_score_1, face_score_2)
#
#     if score <= .5:
#       # Printing the IDs of faces and score
#       print(x, y, score)
#       # Displaying each matched pair of faces
#       finalImage.append(extracted_face_uno[x])
#       #plt.imshow(extracted_face_uno[x])
#       #plt.show()
#       finalImage.append(extracted_face_dos[y])
#       imageSimilarityScores.append((1-score)*100)
#       imageSimilarityScores.append((1 - score) * 100)
#       #plt.imshow(extracted_face_dos[y])
#       #plt.show()
#
#
# for i in range(len(finalImage)):
#     plt.subplot(5,3,i+1)
#    # the number of images in the grid is 5*5 (25)
#     plt.imshow(finalImage[i])
#     plt.title(imageSimilarityScores[i])
#     plt.axis('off')
#
#
# plt.show()
# ##face = util.prep_image('1.jpg')
# #
#
# #faces = MTCNN.detect_faces(image)
#
#
#
#
# #for face in faces:
# #    print(face)
# #util.draw_box('1.jpg', faces)
#
#
