from flask import Flask,request, jsonify, Response,send_file
from tensorflow.python.platform import gfile
import numpy as np
import cv2
import tensorflow as tf
import mtcnn_detect
import jsonpickle
import facenet
import pickle

app = Flask(__name__)

#IMG_FOLDER = os.path.join('static')

MODEL_PATH = './models/frozen.pb'
SVM_PATH = 'svm_classifier.pkl'
minsize = 20
threshold = [0.6, 0.7, 0.7]
factor = 0.709
margin = 44
image_size = 160
input_map=None
embedding_size = 128
(model, class_names) = pickle.load(open(SVM_PATH, 'rb'))
sess = tf.Session()
#model_api = api.get_model_api()
pnet, rnet, onet = mtcnn_detect.create_mtcnn(sess, 'models')
def load_graph():
    with gfile.FastGFile(MODEL_PATH,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, input_map=input_map, name='')
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name=""
            )
    
    return graph


@app.route("/")
def index():
    """
    this is a root dir of my server
    :return: str
    """
    return "This is root!!!!"
graph=load_graph() 
@app.route('/post', methods=['POST','GET'])
def test():
    #read image file string data
    filestr = request.files['file'].read()
    #convert string data to numpy array
    npimg = np.fromstring(filestr, np.uint8)
    # convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
   
    bounding_boxes = []
    sess= tf.Session(graph=graph)
    emb_array = np.zeros((1, embedding_size))
    bounding_boxes, _ = mtcnn_detect.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    nrof_faces = bounding_boxes.shape[0]
    if nrof_faces > 0:
        det = bounding_boxes[:, 0:4]
        img_size = np.asarray(img.shape)[0:2]
   
        cropped = []
        scaled = []
        names = []
        scaled_reshape = []
        bb = np.zeros((nrof_faces,4), dtype=np.int32)
                #print(nrof_faces)
        for i in range(nrof_faces):
            emb_array = np.zeros((1, embedding_size))
            bb[i][0] = det[i][0]
            bb[i][1] = det[i][1]
            bb[i][2] = det[i][2]
            bb[i][3] = det[i][3]
               
            if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(img[0]) or bb[i][3] >= len(img):
                print('face is inner of range!')
                continue
            cropped.append(img[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
            cv2.rectangle(img,(bb[i][0],bb[i][1]),(bb[i][2],bb[i][3]),(0,0,255), thickness=2) #draw bounding box for the face
            for i in range(len(cropped)):  
                if cropped[i].shape[0] >70 and cropped[i].shape[1]>70:                       
                    scaled.append(cropped[i])
            for i in range(len(scaled)): 
                scaled[i] = cv2.resize(scaled[i], (image_size,image_size),interpolation=cv2.INTER_CUBIC)
                scaled[i] = facenet.prewhiten(scaled[i])
                scaled_reshape.append(scaled[i].reshape(-1,image_size,image_size,3))
                                #Feed forward
                feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                                #Đưa vector emb vào classifier 
                emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1) #Class đạt độ chính xác cao nhất
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]#Độ chính xác
                for j in range(len(best_class_indices)):
                               #print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                    if (best_class_probabilities[j] >=0.13): 
                        cv2.putText(img, class_names[best_class_indices[j]], (bb[i][0], bb[i][1]-10), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0,255,0), thickness=1, lineType=2)   
                        names.append(class_names[best_class_indices[j]])   
    # do some fancy processing here....
                    else:
                        cv2.putText(img, "Unknown", (bb[i][0], bb[i][1]-10), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0,255,0), thickness=1, lineType=2)   
    cv2.imwrite('img.jpg',img)
    # encode response using jsonpickle
    #response_pickled = jsonpickle.encode(bounding_boxes.tolist())

    return send_file('img.jpg', attachment_filename='img.jpg')

if __name__ == '__main__':
    graph=load_graph() 
    images_placeholder = graph.get_tensor_by_name("input:0")
    embeddings = graph.get_tensor_by_name("embeddings:0")
    phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]
    app.run(host='0.0.0.0', port=5000)

