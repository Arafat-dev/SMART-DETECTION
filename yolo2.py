

import cv2
import numpy as np


from tkinter import*


window = Tk()
window.geometry("1050x760")
window.minsize(480,480)
window.title("Prevision caisier")
window.iconbitmap("logo_IS4BI.ico")
window.config(background='#7AF5B6')

my_frame = Frame(window,bg="white")



def start_prog():
    btn_lancer.flash()   
   
    global som 
    som = 0
    global execu
    global cap
    global nb_caisse
    execu = 0
    nb_caisse = 0
    # load our YOLO object detector trained on COCO dataset (80 classes)
    net = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    
    
    DEFAULT_CONFIANCE = 0.5
    THRESHOLD = 0.4
    # load the COCO class labels our YOLO model was trained on
    with open('coco.names', 'r') as f:
        LABELS = f.read().splitlines()

    # initialize the video stream, pointer to output video file
    cap = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)


    while execu == 0:  

    
        _,image=cap.read()     
        height, width,_ = image.shape
   
      

        #camera 2
        _,image1=cap1.read()
        height, width,_ = image1.shape
       
        blob1 = cv2.dnn.blobFromImage(image1, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
        net.setInput(blob1)

        blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
        net.setInput(blob)
        layerOutputs = net.forward(ln)

        #initialize our lists of detected bounding boxes, confidences, and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

        #loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability)
                # of the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > DEFAULT_CONFIANCE:
                    # scale the bounding box coordinates back relative to
                    # the size of the image, keeping in mind that YOLO
                    # actually returns the center (x, y)-coordinates of
                    # the bounding box followed by the boxes' width and
                    # height
                    box = detection[0:4] * np.array([width, height, width, height])
                    (centerX, centerY, W, H) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top
                    # and and left corner of the bounding box
                    x = int(centerX - (W / 2))
                    y = int(centerY - (H / 2))
                    # update our list of bounding box coordinates,
                    # confidences, and class IDs
                    boxes.append([x, y, int(W), int(H)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
            
        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, DEFAULT_CONFIANCE, THRESHOLD)

        # initialize a list of colors to represent each possible class label
        COLORS = np.random.uniform(0,255,size=(len(boxes), 3))

        som = 0      
        # ensure at least one detection exists
        if len(indexes) > 0:
            
            # loop over the indexes we are keeping
            for i in indexes.flatten():
                                
                if LABELS[classIDs[i]] == "person":
                    # extract the bounding box coordinates
                    (x, y, w, h) = boxes[i]
                    # draw a bounding box rectangle and label on the frame
                    color = COLORS[i]
                    text = "{}".format(LABELS[classIDs[i]])
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image, text, (x, y + 20 ), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

                    ###########
                    cv2.rectangle(image1, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image1, text, (x, y + 20 ), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                    ##########
                    som = som + 1           
                if som <= 0:
                    nb_caisse = 0
                elif som < 4:
                    nb_caisse = 1
                elif som % 4 == 0:
                   nb_caisse = som/4
                else:
                    nb_caisse = som//4

                
       
                    
        cv2.imshow('Image', image)
        cv2.imshow('Image1', image1)
        
        if cv2.pollKey()==ord('q'):
            execu = 1    
    
    label2 = Label(window,text="Nombre de Personne :",font=("Courrier",20),bg='#7AF5B6',fg='white')
    label2.pack(side='top') 
    
    compter = Label(window,text=som,font=("Courrier",20),bg='#7AF5B6',fg='white')
    compter.pack(side='top')  

    label3 = Label(window,text="Nombre caisse necessaire :",font=("Courrier",20),bg='#7AF5B6',fg='white')
    label3.pack(side='top') 
    caisse = Label(window,text=nb_caisse,font=("Courrier",20),bg='#7AF5B6',fg='white')
    caisse.pack(side='top')  
                        
                   

def arreter(): 
    btn_arret.flash()  
   
    cap.release()
    cv2.destroyAllWindows()



my_frame = Frame(window,bg="white")

label_title = Label(window,text="Système de detection de nombre des visiteurs",font=("Courrier",20),bg='#7AF5B6',fg='white')

label_title.pack(side='top')

btn_lancer = Button(window,text="Lancer le programme",font=("Courrier",18), bg='white', fg='#7AF5B6',command=start_prog)
btn_lancer.pack(pady="20")



btn_arret = Button(window,text="Arreter le Comptage",font=("Courrier",18), bg='white', fg='#7AF5B6',command=arreter)
btn_arret.pack(pady="20")


window.mainloop()



