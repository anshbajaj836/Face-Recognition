# Recognize faces using some classification alforithm - like Logistics,kNN, svm etc

# 1. Read a vedio stream using opencv
#2. Extract faces out of it
# 3. load the training data (numpy arrays of all the persons)
#        x- values are stored in the numpy array
#        y - values we needd to assihn for each person
# 4. use the knn to find teh prediction of face(int)
# 5. map the predicted is to the name of the user
# 6. deiplay the predictions on the screen - bouding box and name

import cv2
import numpy as np
import os

####### KNN Code ######

def distance(v1,v2):
    return np.sqrt(((v1-v2)**2).sum())



def knn(train,test,k=5):
    dist = []

    for i in range(train.shape[0]):
        # get the vector and label
        ix = train[i, :-1]
        iy = train[i,-1]
        # compute the dist from each point
        d = distance(test,ix)
        dist.append([d,iy])
    
    # now sort on the basis of distance and get the top k 
    dk = sorted(dist, key = lambda x: x[0])[:k]

    # Now Retreive only labels of top k
    labels = np.array(dk)[:,-1]

    # now get the freq of each label
    output = np.unique(labels, return_counts=True)
    # find the max of freq and correspoding label
    index = np.argmax(output[1])

    return output[0][index]

########################



# read the video stream

cap = cv2.VideoCapture(0)

# Face detection
face_cascade = cv2.CascadeClassifier("D:\C++ cp 1\project python\OpenCv\haarcascade.xml")
skip = 0

face_data =[]
labels = []

dataset_path = "D:\C++ cp 1\project python\OpenCv\Data"

class_id = 0 # labels for given file
names = {} # mapping btw id and name

# Data Preparation

# now we are going to iterate over our directory
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        print("Loaded " + fx)
        data_item = np.load(dataset_path + "\\" +fx)
        face_data.append(data_item)
        names[class_id] = fx[:-4] # this will create the mapping between class id and name
        # Create Labels for the class
        target = class_id*np.ones((data_item.shape[0]),)
        class_id +=1
        labels.append(target)

face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))
print(face_dataset.shape) # (7,30000)
print(face_labels.shape) # (7,)->(7,1)

## now we have to concat this data  to form training data, as it is combination of features and label
trainset = np.concatenate((face_dataset,face_labels),axis = 1)
print(trainset.shape)


# Preparing test data


while True: 
    ret ,frame = cap.read()
    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame,1.3,5)

    for face in faces:
        x,y,w,h = face

        # get the face Region of interest
        offset =10
        face_section = frame[y-offset:y+h+offset , x-offset : x+w+offset]
        face_section = cv2.resize(face_section,(100,100))

        # predicted output
        out = knn(trainset,face_section.flatten())

        # display ont the screen the name and rectangle aroung it
        pred_name = names[int(out)]
        cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y), (x+w,y+h),(0,255,255),2)

    
    cv2.imshow("Faces",frame)
    
    key = cv2.waitKey(1) & 0xFF
    if (key==ord('q')):
        break 