import cv2
import os
import numpy as np

test_img = cv2.imread(r"E:\University\Spring\ArtificialIntelligenceLab\Project\Still_Image_Detection_&_Recognition\testImage\imran.jpeg")

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
face_detected = face_cascade.detectMultiScale(test_img, scaleFactor=1.32, minNeighbors=5)

faces=[]
faceID=[]
for path, subdirnames, filenames in os.walk(r"E:\University\Spring\ArtificialIntelligenceLab\Project\Still_Image_Detection_&_Recognition\trainImage"):
    for filename in filenames:
        if filename.startswith("."):
            print("Skipping system file")
            continue
        
        id = os.path.basename(path)
        img_path = os.path.join(path, filename)
        print("img_path:", img_path)
        print("id:", id)
        
        test_img2 = cv2.imread(img_path)
        
        if test_img2 is None:
            print("Image not loaded properly")
            continue
        
        gray_img2 = cv2.cvtColor(test_img2, cv2.COLOR_BGR2GRAY)
        faces_rect = face_cascade.detectMultiScale(gray_img2, scaleFactor=1.32, minNeighbors=5)
        
        if len(faces_rect) != 1:
            continue
        (x,y,w,h) = faces_rect[0]
        roi_gray = gray_img2[y:y+w, x:x+h]
        faces.append(roi_gray)
        faceID.append(int(id))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(faceID))
face_recognizer.save("trainingData.yml")
face_recognizer.read(r"E:\University\Spring\ArtificialIntelligenceLab\Project\Still_Image_Detection_&_Recognition/trainingData.yml")

name = {0:"Salman", 1:"Amir", 2:"Imran", 3:"Anik"}
#name = ["Salman", "Amir"]
x,y, w, h
for face in face_detected:
    (x,y,w,h) = face
    roi_gray = gray_img[y:y+h, x:x+h]
    
    label, confidence = face_recognizer.predict(roi_gray)
    print("confidence:", confidence)
    print("label:", label)
    
    (x,y,w,h) = face
    cv2.rectangle(test_img, (x,y), (x+w, y+h), (0, 255, 0), 3)
    
    predicted_name = name[label]
    """for i in name:
        if label==i:
            predicted_name = name[label]
            flag=1
            break
    if flag==0:
        predicted_name="unknown"
        """
    if confidence>100:
          continue
    cv2.putText(test_img, predicted_name, (x,y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)


resized_img = cv2.resize(test_img, (500, 500))
cv2.imshow("face detection", resized_img)

#cv2.imshow("face detection tutorial", test_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
