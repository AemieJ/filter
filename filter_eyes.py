#Importing libraries
import argparse 
import cv2
from imutils.video import VideoStream #it creates a really good video stream
from imutils import face_utils, translate, resize
#face_utils : something that converts dlib to numpy so it can be furthur used.
#translate : it's going to translate the current position of our eyes to the previous pos.
#resize :  for faster computation
import time
import dlib
import numpy as np

#Taking arguments from command line
parser = argparse.ArgumentParser() #you iniatize as such
parser.add_argument("-predictor", required=True, help="path to predictor")
#the add_argument tells you what needs to be given as an input sp its help 
args = parser.parse_args() #you take the arguments from command line

#Controls
print("Starting Program.")
print("Press 'Esc' to quit")

#Video from webcam 
video = VideoStream().start()
time.sleep(1.5) #to hault the code it will stop after 1.5 sec

#the detector responsible for detecting the face and predictor responsible for predict the 68 points on the face
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor(args.predictor) #used for getting outside loop

#For taking video 
counter = 0

#creating separate eye layer and eye mask to like extract only the eyes from the face and to work with it
eye_layer = np.zeros((300,400,3),dtype = 'uint8') 
eye_mask = eye_layer.copy()
eye_mask = cv2.cvtColor(eye_mask , cv2.COLOR_BGR2GRAY) #eyes are white on a black background

#create translated mask to know the mask of the previous pos of all the eyes 
translated = np.zeros((300,400,3),dtype = 'uint8') 
translated_mask = eye_mask.copy()

#creating the eye list for storing all the positions of eyes in different frames
class EyeList(object) : 
    def __init__(self,length) : 
        self.length = length #length will be the total number of pos to be stored
        self.eyes = []
    def push(self,newCoords) : 
        if len(self.eyes) < self.length : 
            self.eyes.append(newCoords) 
        #when we reach the max limit for the eyelist , we remove the oldest coordinates
        else : 
            self.eyes.pop(0)
            self.eyes.append(newCoords)
        
eye_list = EyeList(10) #10 coordinates/positions of eyes
 
#Making Video 
img_list = []
out = cv2.VideoWriter('Filter-Eyes.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (400,300))

while True : 
        frame = video.read()
        frame = resize(frame,width = 400)
        gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY) 
        rectangle = detector(gray,0) 
    
        # fill our masks and frames with 0 (black) on every draw loop
        eye_mask.fill(0) #this will only show the exact eye pos on the screen and not its continuous movement 
        eye_layer.fill(0) 
        translated.fill(0)
        translated_mask.fill(0)
     
        for rect in rectangle : 
            x,y,w,h = face_utils.rect_to_bb(rect) #gives the coordinates and size
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) 
            shape = predictor(gray,rect) #dlib output will be received after predicting 
            shape = face_utils.shape_to_np(shape) 
        
            lefteye = shape[36:42]
            righteye = shape[42:48] 
            # fill our mask in the shape of our eyes
            cv2.fillPoly(eye_mask,[lefteye],255) 
            cv2.fillPoly(eye_mask,[righteye],255) 
        
            #take the eyemask and do bitwise AND with the frame 
            '''What happens is that the bitwise AND will be performed between eyemask and frame , whichever gives the 
            true or 1 will be shown actually in the eyelayer'''
            eye_layer = cv2.bitwise_and(frame,frame,mask = eye_mask)
            #Getting the coordinates for the eye at diff position in each frame
            ex,ey,ew,eh = cv2.boundingRect(eye_mask)
            eye_list.push([ex,ey])
            #Accessing the coordinates in the reverse order 
            for i in reversed(eye_list.eyes) :     
                translated1 = translate(eye_layer, i[0] -ex ,i[1]-ey) #translate take x and y coords to translate/move from  
                translated1_mask = translate(eye_mask, i[0]-ex ,i[1]-ey) 
                translated_mask = np.maximum(translated_mask , translated1_mask) #when you've 255 in both and if you add them you get overflow in np hence you get max
                #cut out the new translated mask 
                translated = cv2.bitwise_and(translated,translated,mask=255-translated1_mask) 
                #paste in the newly translated eye position 
                translated += translated1
                
                '''for point in shape[36:48] : #we will only extract the eyes points in the entire face 
                    cv2.circle(frame,tuple(point),2,(0,255,0)) #marks the points embracing the detected shape of face'''
        #translated_mask will have all the previous eye position so we will black out those ones from the current eye
        frame = cv2.bitwise_and(frame,frame,mask = 255-translated_mask)
        frame += translated #paste in the translated eye image 
    
        cv2.imshow("Eye Glitch",frame) 
        img_list.append(frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 :
            break 

#Writing all frames to make a video file
for img in img_list : 
    out.write(img) 
    
out.release() 
video.stop()
cv2.destroyAllWindows()

