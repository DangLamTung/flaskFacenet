from flask import Flask,request, jsonify, Response,send_file
from tensorflow.python.platform import gfile
import numpy as np
import cv2
import tensorflow as tf
import mtcnn_detect
import jsonpickle
import facenet
import pickle
import time 
import json 
import os
import sql
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import time

import mysql.connector
from mysql.connector import Error

def create_connection(host_name, user_name, user_password):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password
        )
        print("Connection to MySQL DB successful")
    except Error as e:
        print(f"The error '{e}' occurred")

    return connection

connection = create_connection("localhost", "root", "Tung3071999%")

def create_database(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        print("Database created successfully")
    except Error as e:
        print(f"The error '{e}' occurred")

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
@app.route("/test_post", methods=['POST','GET'])
def index_post():
    """
    this is a root dir of my server
    :return: str
    """
    name = request.get_data("name",as_text = True)
    return str(name)

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
            # for i in range(len(cropped)):  
            #     if cropped[i].shape[0] >70 and cropped[i].shape[1]>70:                       
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
                    

                        time_stamp = time.strftime('%Y-%m-%d %H:%M:%S')
# print("Current Time =", current_time)
                        sql = "INSERT INTO Checkin(name, time) VALUES (%s,%s)"
                        val = (class_names[best_class_indices[j]], time_stamp)
                        mycursor.execute(sql,val)
                        db_1.commit()
   
                    else:
                        cv2.putText(img, "Unknown", (bb[i][0], bb[i][1]-10), cv2.FONT_HERSHEY_COMPLEX_SMALL,1, (0,255,0), thickness=1, lineType=2)   
    cv2.imwrite('img.jpg',img)
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(bounding_boxes.tolist())

    return send_file('img.jpg', attachment_filename='img.jpg')

@app.route('/train', methods=['POST','GET'])
def train():

    #read image file string data
    filestr = request.files.getlist("file")
    namestr = request.files.get("name")
    name = []
    # print(namestr.read())
    person_data = (namestr.read().decode("utf-8")).split("\n")
    npimg = []
    img = []
    data_person = []
    bounding_boxes = []
    
    sess= tf.Session(graph=graph)
    emb_array = np.zeros((1, embedding_size))
    #convert string data to numpy array
    for i in filestr:
        npimg = np.fromstring(i.read(), np.uint8)
    # convert numpy array to image
        img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
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
                # for i in range(len(cropped)):  
                #     if cropped[i].shape[0] >70 and cropped[i].shape[1]>70:                       
                scaled.append(cropped[i])
                for i in range(len(scaled)): 
                    scaled[i] = cv2.resize(scaled[i], (image_size,image_size),interpolation=cv2.INTER_CUBIC)
                    scaled[i] = facenet.prewhiten(scaled[i])
                    scaled_reshape.append(scaled[i].reshape(-1,image_size,image_size,3))
                                    #Feed forward
                    feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                                    #Đưa vector emb vào classifier 
                    emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
        data_person.append(emb_array)

    labels = []
    embs = []
    class_names = []
    with open('class.txt') as file:
        for l in file.readlines():
            class_names.append(l.replace('\n', ''))
    file.close()

    print(class_names)
    with open('data.txt') as json_file:  
        data = json.load(json_file)
        for p in data['person']:
            embs.append(p['emb'])
            labels.append(p['name'])
    name = person_data[0]
    if name in class_names:
        person_label = class_names.index(name)
        print('This person is already in database')
    else:
        person_label = len(class_names) 
        file = open('class.txt','w')  
        class_names.append(str(name))
        for name in class_names:
            file.write(str(name) + '\n')
        file.close()           
    for i in range(len(emb_array)):
            data['person'].append({'name':person_label,'emb':emb_array[i].tolist()})
            labels.append(person_label)
            embs.append(emb_array[i])
    with open('data.txt', 'w') as outfile:
        json.dump(data, outfile)
    X_train, X_test, y_train, y_test = train_test_split(np.array(embs),np.array(labels), test_size=0.33, random_state=42)
    print('Training SVM classifier')
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test)
    best_class_indices = np.argmax(predictions, axis=1)
    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
    accuracy = np.mean(np.equal(best_class_indices, y_test))
    print('Accuracy: %.3f' % accuracy)
    with open('svm_classifier1.pkl', 'wb') as outfile:
        pickle.dump((model, class_names), outfile)
    print('Saved svm classifier')
    # encode response using jsonpickle

    sql = "INSERT INTO Student(name, address) VALUES (%s,%s)"
    val = (name, "132")
    mycursor.execute(sql,val)
    db_1.commit()
 
    response_pickled = jsonpickle.encode(bounding_boxes.tolist())

    return send_file('img.jpg', attachment_filename='img.jpg')

if __name__ == '__main__':
    graph=load_graph() 
    images_placeholder = graph.get_tensor_by_name("input:0")
    embeddings = graph.get_tensor_by_name("embeddings:0")
    phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]

    connection = create_connection("localhost", "root", "Tung3071999%")
    create_database(connection, "CREATE DATABASE data_student")
    create_database(connection, "CREATE DATABASE data_checkin")

    db_1 = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="Tung3071999%",
    database="sm_app"
    )
    mycursor = db_1.cursor(buffered=True)

    mycursor.execute("CREATE TABLE if not exists Student (id INT AUTO_INCREMENT PRIMARY KEY, name VARCHAR(255), address VARCHAR(255), time VARCHAR(255))")
    mycursor.execute("CREATE TABLE if not exists Checkin  (id INT AUTO_INCREMENT PRIMARY KEY,name VARCHAR(255), time TIME)")
    mycursor.execute("SELECT * FROM Checkin JOIN Student USING(name)")
    app.run(host='0.0.0.0', port=5000)

