# FaceSimilarityWebApp
This flask web application allows a user to input 2 images and then detects which people are present in both pictures.
The MTCNN model from FACENet is used to detect faces while the VGGFACE model is used to compute embeddings for the faces. 
Cosine similarity is used to analyze which people are present in both photos from the embeddings.
