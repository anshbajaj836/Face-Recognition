# write a python script that captures images of your webcam video stream
# Extracts all faces from the image frame (using haarcascades)
# stores the face information into numpy arrays


# steps
# 1. read and show video stream and capture images
# 2. detect faces and show bouding box
# 3. flatten the largest face image and save it in the numpy array
# 4. Repeat the above for multiple people to generate training data

import cv2
import numpy as np

# init camera 
cap = cv2.VideoCapture(0)

# Face detection
face_cascade = cv2.CascadeClassifier("D:\C++ cp 1\project python\OpenCv\haarcascade.xml")
skip = 0
face_data =[]

dataset_path = "D:\C++ cp 1\project python\OpenCv\Data\ "

file_name = input("Enter the name of the person: ")

while True:

    ret,frame = cap.read()

    if ret == False:
        continue


    # doing everything for grey frame
    grey_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)


    faces = face_cascade.detectMultiScale(frame,1.3,5)
    # after we have scanned the faces
    # we are going to sort  them on the basis of area
    # considering only the one with the largest area
    # this will be helpful in case of multiple faces
    # area = w*h
    faces = sorted(faces,key = lambda f:f[2]*f[3], reverse=True)
# extract (crop out the required face): Region of interest
    offset = 10
    
    for (x,y,w,h) in faces[:1]:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        
        face_selection = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_selection = cv2.resize(face_selection,(100,100))
        skip=skip+1
        
        if(skip%10==0):
            face_data.append(face_selection)
            print(len(face_data))
        # going to add the padding of 10 pixels on each side, that is done with the help of offset and face selection


    

    cv2.imshow("Frame",frame)
 
    # each face is a tuple (x,y,w,h)
    # we will store every 10th face
    
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed==ord('q'):
        break



# Convert our face_data into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)


# save this file into the system
np.save(dataset_path+file_name +'.npy',face_data)
print("Data Successfully saved at " +dataset_path+ file_name +'.npy')



cap.release()
cv2.destroyAllWindows()