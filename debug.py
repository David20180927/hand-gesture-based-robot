#!C:\Users\user\AppData\Local\Programs\Python\Python39\python.exe
"""The image first processed using the hsv method to threshold away anything that is not skin color
then transfer the image to mediapipe to mark out hand"""
import serial
import glob
import cvzone
from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np
import math
from math import degrees,acos,asin,atan,pow,sqrt,sin,cos,radians
import os
from PIL import Image # For handling the images
"""AI"""
from keras.models import model_from_json#
import tensorflow as tf
from tensorflow.keras.models import load_model

'''general parameters'''
dirmodel = 'model/'
dirdata = 'dataset/'
#d_raw = [20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
#d_raw_corres = [300,245,200,170,145,130,112,103,93,87,80,75,70,67,62,59,57]
d_raw = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]
d_raw_corres =[52, 52, 53, 52, 53, 53, 53, 53, 54, 55, 55, 58, 58, 59, 61, 62, 63, 65, 66, 66, 67, 69, 70, 71, 73, 75, 76, 77, 78, 79, 78, 83, 84, 86, 88, 89, 91, 92, 92, 98, 99, 100, 100, 103, 107, 107, 111, 114, 116, 118, 119, 122, 123, 126, 129, 131, 135, 138, 142, 144, 147, 151, 155, 159, 166, 170, 173, 180, 187, 195, 199, 207, 211, 221, 229, 242, 249, 262, 273, 279, 290, 310, 317, 326, 349, 356, 369, 384, 401, 420, 428, 447, 461, 483, 515, 528, 564, 560, 582, 604]
d_fist = []
d_fist_corres = []
choise = 0
label = {}
x = 0
d = [1,1.1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
data = 0
ges_left,ges_right = '',''
"""skin color threshold parameters"""
hmin, smin, vmin = 0, 48, 80
Hmax, Smax, Vmax = 20, 255, 255
lower_hsv = np.array([hmin, smin, vmin])
upper_hsv = np.array([Hmax, Smax, Vmax])
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
"""Hands detection parameters"""
detector = HandDetector(detectionCon=0.8, maxHands=2)
"""Selfi segmentation parameters"""
blackimg = np.zeros((175, 175, 3), np.uint8)
blackimg[:] = (255,255,255)

"""user interface image output parameters"""
stick_center = [1000,300]
hand1= None
enlargen = 30
text_font = cv2.FONT_HERSHEY_COMPLEX
counter = 0
stick_counter = 0
dis_or,dis = 0,0
control_mode = 3 #0 is chassis, 1 is robotic arm
stick_opos,current_stick_opos,stick_img = [0,0],[0,0],[stick_center[0],stick_center[1]]
A,B,C = np.polyfit(d_raw,d_raw_corres,2)
ran = 10
"""Class Defination"""
class data_handle():
    def __init__(self):
        self.test,self.chassis_command,self.vx_set_order,self.vy_set_order,self.wz_set_order,self.pwm_one,self.pwm_two,self.pwm_three,self.pwm_four,self.pwm_five,self.pwm_six,self.dynamic_x,self.dynamic_y,self.d1,self.d2,self.d3,self.d4,self.d5,self.d6 = 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    def process_data(self,d):
        d[0] = self.test #0,1
        d[1] = self.chassis_command #1 = stop,2 = forward, 3 = backward, 4 = left, 5 = right
        '''On robot side, the value can only be 0 ~ 2, since usb cant send float, so the range here will be -100 ~ 100'''
        '''when robot receive it it should -100, and then /100'''
        d[11] = int(contrain(self.vx_set_order+100,0,200)) 
        d[12] = int(contrain(self.vy_set_order+100,0,200))
        d[13] = int(contrain(self.wz_set_order+100,0,200))
        '''Every range is different, but since it wont exceed 2600, I ll just /26'''
        '''robot should *26'''
        d[21] = int(contrain(self.pwm_one,600,2450)/26)
        d[22] = int(contrain(self.pwm_two,600,2300)/26)
        d[23] = int(contrain(self.pwm_three,1300,2300)/26)
        d[24] = int(contrain(self.pwm_four,600,2450)/26)
        d[25] = int(contrain(self.pwm_five,420,2500)/26)
        d[26] = int(contrain(self.pwm_six,1500,2300)/26)

        d[28] = self.dynamic_x
        d[29] = self.dynamic_y

        d[28] = self.dynamic_x
        d[29] = self.dynamic_y      

        d[41] = int(self.d1/26) 
        d[42] = int(contrain(self.d2,0,90))
        d[43] = int(contrain(self.d3+140,0,120))
        d[44] = int(contrain(self.d4+90,0,180))
        d[45] = self.d5
        d[46] = int(contrain(self.d6-40,0,250))
        return d
    
class round_button():
    def __init__(self,pos = [100,100],size = 30,text = 'undefined',tcolor = (0,0,0),color = (0,0,0)):
        self.cx ,self.cy, self.rad, self.text, self.color,self.scolor,self.tcolor,self.stcolor=pos[0], pos[1],size,text,color,color,tcolor,tcolor
    def draw(self,img):
        cv2.circle(img,(self.cx,self.cy),self.rad+4,(255,255,255),cv2.FILLED)
        cv2.circle(img,(self.cx,self.cy),self.rad,self.color,cv2.FILLED)
        cv2.putText(img,self.text,(self.cx-self.rad-2*len(self.text),self.cy-self.rad-10),text_font,1.6,self.tcolor)
        return img
    def pressing(self,press = False):
        if press:
            self.color = (139,131,131)
        else:
            self.color = self.scolor
        return True
    def color_mode1(self):
        self.scolor = (238,0,0)
        self.text = 'CHASSIS MODE'
        self.tcolor = (238,0,0)
        return
    def color_mode2(self):
        self.scolor = (0,0,255)
        self.text = 'ROBOT ARM MODE'
        self.tcolor = (0,0,255)
        return       
        
        
class stick():
    def __init__(self,pos = [200,200],c_size = 20, s_size = [10,10], text = 'undefined', tcolor = (0,0,0),color = (0,0,0), stick_center = [200,200]):
        self.cx,self.cy,self.rad,self.text,self.color,self.scolor,self.tcolor,self.stcolor=pos[0], pos[1],c_size,text,color,color,tcolor,tcolor
        self.sx,self.sy,self.sw,self.sh = stick_center[0],stick_center[1],s_size[0],s_size[1]
    def draw(self,img):
        cv2.circle(img,(self.cx,self.cy),self.rad+4,(192,192,192),cv2.FILLED)
        cv2.circle(img,(self.cx,self.cy),self.rad,(139,61,72),cv2.FILLED)
        cv2.line(img, (self.cx-self.rad,self.cy), (self.cx+self.rad,self.cy), (255,0,0), 4)
        cv2.line(img, (self.cx,self.cy-self.rad), (self.cx,self.cy+self.rad), (255,0,0), 4)
        cv2.putText(img,self.text,(self.cx-self.rad+len(self.text)+50,self.cy-self.rad-80),text_font,1.6,self.tcolor)#chassis
        cv2.putText(img,"F",(self.cx-10,self.cy-self.rad-20),cv2.FONT_HERSHEY_SIMPLEX,1.6,(0,255,0),3)#F
        cv2.putText(img,"B",(self.cx-10,self.cy+self.rad+50),cv2.FONT_HERSHEY_SIMPLEX,1.6,(0,255,0),3)#B
        cv2.putText(img,"L",(self.cx-self.rad-40,self.cy+10),cv2.FONT_HERSHEY_SIMPLEX,1.6,(0,255,0),3)#L
        cv2.putText(img,"R",(self.cx+self.rad+20,self.cy+10),cv2.FONT_HERSHEY_SIMPLEX,1.6,(0,255,0),3)#R
        cv2.rectangle(img,(self.sx-self.sw,(self.sy-self.sh)), (self.sx+self.sw,(self.sy+self.sh)),(255,244,244),cv2.FILLED)
        cv2.rectangle(img,(self.sx-self.sw+5,(self.sy-self.sh)+5), (self.sx+self.sw-5,(self.sy+self.sh)-5),self.color,cv2.FILLED)
        return img
    def pressing(self,press = False):
        if press:
            self.color = (139,131,131)
        else:
            self.color = self.scolor
        return True
    def running(self,new_pos):
        self.sx, self.sy = new_pos[0],new_pos[1]
        return
class stablelizer():
    def __init__(self):
        self.store,self.data = 0,0
    def stablelize(self,data,sen = 10):
        self.data = data
        if abs(self.data - self.store) < sen:
            return self.store
        self.store = self.data
        return self.data
        
ctrl_mode_botton = round_button([160,160],20,"Press to start")
chassis_control_stick = stick(stick_center,150,[40,25],'Chassis',(255,0,0),(0,0,0),stick_center)
transfer_data = data_handle()
stick_stable_x, stick_stable_y= stablelizer(),stablelizer()
d1_stable,d2_stable,d3_stable,d4_stable,d5_stable,d6_stable = stablelizer(),stablelizer(),stablelizer(),stablelizer(),stablelizer(),stablelizer()
"""Predefined functions"""
def import_label(label):
    count = 0
    for i in os.listdir(dirdata):
        label[count] = i
        count += 1
    return label
def find_label(predictions, label):
    p = predictions[0]
    for i in range(len(label)):
        if p[i]: # if p[i] = 1, then the i is the location in the array
            return label[i] #return the location dict
def preprocess_bf_handover(img):
    img = Image.fromarray(img).convert('L')
    img = img.resize((320, 120))    
    x = [np.array(img)]
    x = np.array(x, dtype = 'float32')
    x = x.reshape((1, 120, 320, 1))
    x /= 255
    return x

def contrain(data,min_d,max_d):
    if data < min_d:
        return min_d
    elif data> max_d:
        return max_d
    else:
        return data

def control_screen_load(img):
    '''code here'''
    img = cv2.flip(img,1)#lets flip  
    img = ctrl_mode_botton.draw(img)
    
    
    '''code here'''
    return img
def control_screen_load_chassis(img,current_stick_opos):
    
    img = chassis_control_stick.draw(img)
    img = cv2.putText(img,"Speed x: "+str(current_stick_opos[1]/20)+' m/s',(450,30),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2)
    img = cv2.putText(img,"Speed y: "+str(current_stick_opos[0]/20)+' m/s',(450,60),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2)
    cv2.rectangle(img,(440,2), (760,70),(181,228,255),2)
    chassis_control_stick.pressing(False)
    return img

def control_screen_load_arm(img):
    #img = draw_arm(img,w_angle)
    return img

def button_is_pressed(button,Left_hand,Right_hand,hand1, A,B,C):
    if Left_hand:
        lmlist1, drawnbox1, centerpoint1, handtype1 = Left_hand["lmList"], Left_hand["bbox"], Left_hand["center"], Left_hand["type"]
        x,y,w,h = drawnbox1
        x1,y1 = lmlist1[5]
        x2,y2 = lmlist1[17]
        distance = int(math.sqrt((y2-y1)**2+(x2-x1)**2))
        d_cm =int( A * distance**2 + B * distance +C)
        if x < button.cx < x+w and y < button.cy < y+h:
            if d_cm > 1900:
                return True
        else:
            return False
    return False
    
def measure_distance(hand1,A,B,C):
    lmlist1, drawnbox1, centerpoint1, handtype1 = hand1["lmList"], hand1["bbox"], hand1["center"], hand1["type"]
    x1,y1 = lmlist1[5]
    x2,y2 = lmlist1[17]
    distance = int(math.sqrt((y2-y1)**2+(x2-x1)**2))
    dis =int( A * distance**2 + B * distance +C)
    return dis
def run_hand_detection(img):  
    # Find the hand and its landmark
    if(img is not None):
        #hands, img = detector.findHands(img)#draw
        hands = detector.findHands(img, draw=False)#notdraw
    # Display
        if hands:
            hand1 = hands[0]
            hand1_type = hand1["type"]
            #crop1 = img[drawnbox1[1]-enlargen:drawnbox1[1]+drawnbox1[3]+enlargen,drawnbox1[0]-enlargen:drawnbox1[0]+drawnbox1[2]+enlargen]
            if len(hands) == 2:
                hand2 = hands[1]
                hand2_type = hand2["type"]
                #crop2 = img[drawnbox2[1]-enlargen:drawnbox2[1]+drawnbox2[3]+enlargen,drawnbox2[0]-enlargen:drawnbox2[0]+drawnbox2[2]+enlargen]
                if hand1_type == 'Left':
                    return img,hand1,hand2
                else:
                    return img,hand2,hand1
            if hand1_type == 'Left':
                return img,hand1,None
            else:
                return img,None,hand1
        else:
            return img,None,None 
    else:
        return img,None,None
    

def draw_arm(img,angle = 0):
    r = 200
    angle = radians(angle)
    x,y = int(r*cos(angle)),int(r*sin(angle))
    cv2.line(img, (600,400), (600-x,400+y), (255,0,0), 4)
    return img

def get_wrist_angle(drawnbox,wristpoint,ran):
    '''    y1
            |
    w_x.w_y |
    x1-----x2,y2
    '''
    w_x,w_y = wristpoint
    x,y,w,h = drawnbox[0],drawnbox[1],drawnbox[2],drawnbox[3]
    x1,y1,x2,y2 = x,y,x+w,y+h
    mid_dx,mid_dy =(x2-x1), (y2-y1)
    if(x2-w_x>y2-w_y) and (x2-w_x > ran):#go x
        angle = (-1*(x2-w_x)/mid_dx*90)
    elif(y2-w_y>x2-w_x) and (y2-w_y > ran):#go y
        angle = ((y2-w_y)/mid_dy*90)*0.7
    else:
        angle = 0.0
    return angle
def connect_robot():
    for a in range(6):
        com = 'COM'+str(a)
        try:
            s = serial.Serial(com)
        except:
            continue
        else:
            return s    
def calculation(img,x,y,a1 = 10.4, a2 = 12.7):
    #lmlist not exceed the box
    x,y = contrain(x,480,1150),contrain(y,250,600)
    
    #from lmlist[0] to scale
    x = ((x-480)/670*12)+10
    y = ((y-250)/350*20)-4
    #move cartesian
    y = -y
    x = x-32
    y = y+8
     #flip cartesian
    x = -x
    x,y = contrain(x,10,22),contrain(y,-4,20) #the true scale of robotic arm

    d0,a0,d1,d2 = 0,0,0,0,
    d0 = degrees(atan(y/x))
    a0 = sqrt(pow(x,2)+pow(y,2))

    d1 = degrees(-acos(contrain((pow(x,2)+pow(y,2)+pow(a1,2)-pow(a2,2))/(2*a0*a1),0,1)))+d0
    d2 = degrees(+acos(contrain((pow(x,2)+pow(y,2)+pow(a2,2)-pow(a1,2))/(2*a0*a2),0,1)))+d0-d1
    if (d1 <= 0) or (d1 > 90) or (d2 > -20) or (d2 < -140):
        d1 = degrees(+acos(contrain((pow(x,2)+pow(y,2)+pow(a1,2)-pow(a2,2))/(2*a0*a1),0,1)))+d0
        d2 = degrees(-acos(contrain((pow(x,2)+pow(y,2)+pow(a2,2)-pow(a1,2))/(2*a0*a2),0,1)))+d0-d1        
    d1 = contrain(d1,0,90)
    d2 = contrain(d2,-140,-20)
    return d1,d2

"""Main loop"""
if __name__ == "__main__":
    '''connect usb'''
    s = connect_robot()
    print("Loading pretrained model\n")
    json_file = open(dirmodel+'model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)   
    loaded_model.load_weights(dirmodel+"model.h5")
    label = import_label(label)
    print("Loaded model from disk\nOpening camera\n")
    cap = cv2.VideoCapture(0) #capture video
    cap.set(3,1366) 
    cap.set(4,500)
    while True: 
        success, img = cap.read()
        img = control_screen_load(img) #load button, user interface
        #img, Left_hand,Right_hand = run_hand_detection(img)#detect hands, Return None if failed   
        img, Right_hand, Left_hand = run_hand_detection(img) #Flip control Hands
        current_stick_opos = [0,0]
        if Left_hand or Right_hand: #if hand detected
            if Left_hand:
                lmlistLeft_hand, drawnboxLeft_hand, centerpointLeft_hand, handtypeLeft_hand, fingerLeft_hand = Left_hand["lmList"], Left_hand["bbox"], Left_hand["center"], Left_hand["type"],detector.fingersUp(Left_hand)
                cropleft = img[drawnboxLeft_hand[1]-enlargen:drawnboxLeft_hand[1]+drawnboxLeft_hand[3]+enlargen,drawnboxLeft_hand[0]-enlargen:drawnboxLeft_hand[0]+drawnboxLeft_hand[2]+enlargen]
                #x = preprocess_bf_handover(cropleft)
                #ges_left = find_label((loaded_model.predict(x) > 0.5).astype("int32"),label)
                hand1=Left_hand
                #ges_1 = ges_left
                #cv2.rectangle(img,(drawnboxLeft_hand[0],drawnboxLeft_hand[1]), (drawnboxLeft_hand[0]+drawnboxLeft_hand[2],drawnboxLeft_hand[1]+drawnboxLeft_hand[3]),(255,244,244))
                #cv2.putText(img,ges_left,(drawnboxLeft_hand[0],drawnboxLeft_hand[1]-50),cv2.FONT_HERSHEY_SIMPLEX,1.6,(0,255,0),3)
            if Right_hand:
                lmlistRight_hand, drawnboxRight_hand, centerpointRight_hand, handtypeRight_hand, fingerRight_hand = Right_hand["lmList"], Right_hand["bbox"], Right_hand["center"], Right_hand["type"],detector.fingersUp(Right_hand)
                cropright = img[drawnboxRight_hand[1]-enlargen:drawnboxRight_hand[1]+drawnboxRight_hand[3]+enlargen,drawnboxRight_hand[0]-enlargen:drawnboxRight_hand[0]+drawnboxRight_hand[2]+enlargen]
                #x = preprocess_bf_handover(cropright)
                #ges_right = find_label((loaded_model.predict(x) > 0.5).astype("int32"),label)                
                hand1=Right_hand   
                #ges_1 = ges_right
                #cv2.rectangle(img,(drawnboxRight_hand[0],drawnboxRight_hand[1]), (drawnboxRight_hand[0]+drawnboxRight_hand[2],drawnboxRight_hand[1]+drawnboxRight_hand[3]),(255,244,244))
                #cv2.putText(img,ges_right,(drawnboxRight_hand[0],drawnboxRight_hand[1]-50),cv2.FONT_HERSHEY_SIMPLEX,1.6,(0,255,0),3)                
            lmlist1, drawnbox1, centerpoint1, handtype1, finger1 = hand1["lmList"], hand1["bbox"], hand1["center"], hand1["type"],detector.fingersUp(hand1)
            ctrl_mode_botton.pressing(False)
            '''control mode botton'''
            if button_is_pressed(ctrl_mode_botton,Left_hand,Right_hand,hand1,A,B,C):
                counter = 1
            if counter:
                counter += 1
                ctrl_mode_botton.pressing(True)
                if counter == 3:
                    counter = 0
                    if control_mode:
                        control_mode = 0
                        ctrl_mode_botton.color_mode1()
                    else:
                        control_mode = 1
                        ctrl_mode_botton.color_mode2()
                        '''control mode botton'''
            '''chassis control when control node = 0'''
            if control_mode == 0:

                if finger1 == [0,0,0,0,0]:
                    if stick_counter == 0:
                        stick_opos = centerpoint1
                        stick_counter += 1
                        dis_or = int(measure_distance(hand1,A,B,C)/3)
                    else:
                        dis = int(measure_distance(hand1,A,B,C)/3)
                        current_stick_opos = [contrain(int((centerpoint1[0] - stick_opos[0])/3),-100,100), contrain(dis - dis_or,-100,100)]#(dx,dy)             
                        
                        current_stick_opos[0] = stick_stable_x.stablelize(current_stick_opos[0],10)
                        current_stick_opos[1] = stick_stable_y.stablelize(current_stick_opos[1],10)
                        if current_stick_opos[0]<15 and current_stick_opos[0]>-20 : #makesure it goes straight line
                            current_stick_opos[0] = 0
                        if current_stick_opos[1]<15 and current_stick_opos[1]>-20 : #makesure it goes straight line
                            current_stick_opos[1] = 0                        
                        stick_img[0] = stick_center[0] + current_stick_opos[0]
                        stick_img[1] = stick_center[1] - current_stick_opos[1]
                        chassis_control_stick.running(stick_img)
                        chassis_control_stick.pressing(True)
                        
                         
                else:
                    stick_counter = 0
                    stick_opos, current_stick_opos = (0,0), [0,0]
                    chassis_control_stick.running(stick_center)
                    
                img = control_screen_load_chassis(img,current_stick_opos)
            if control_mode == 1: #robotic arm mode

                if True: #if craw
                    '''n1'''
                    transfer_data.d1 =  0 #forever 900?
                    '''n2,n3'''
                    d2,d3 = calculation(img,lmlist1[0][0],lmlist1[0][1])
                    
                    transfer_data.d2 =  d2_stable.stablelize(d2,5)#every 5 degree
                    transfer_data.d3 =  d3_stable.stablelize(d3,5)
                    '''n4'''
                    w_angle = get_wrist_angle(drawnbox1,lmlist1[0],ran)
                    
                    transfer_data.d4 =  d4_stable.stablelize(w_angle,5)#every 5 degree
                    '''n5'''
                    lor = 0 #0 = none(return to 90degree --> pwm 910), 1 = left, 2 = right
                    '''use if (ges_left) /(ges_right) to determine lor value'''
                    transfer_data.d5 =  lor
                    '''n6'''
                    l,_ = detector.findDistance(lmlist1[4],lmlist1[8])
                    l = int(contrain(l,40,290)) #interger between 40 290
                    transfer_data.d6 =  l
                    #cv2.rectangle(img,(280,60), (1260,650),(255,244,244))
                    #cv2.putText(img,str(lmlist1[0]),(280,60),cv2.FONT_HERSHEY_SIMPLEX,1.6,(0,255,0),3)
                img = control_screen_load_arm(img)#arm or not arm, lets draw it


        '''send data to robot''' 
        
        transfer_data.vx_set_order = current_stick_opos[1]
        transfer_data.vy_set_order = current_stick_opos[0]        
        d = transfer_data.process_data(d)
        try:
            s.write(d) #connect robot
        except:
            cv2.putText(img,'Connection lost',(20,30),cv2.FONT_HERSHEY_SIMPLEX,1.0,(34,34,178),2)
            s = connect_robot()
        else:
            cv2.putText(img,'Robot connected',(20,30),cv2.FONT_HERSHEY_SIMPLEX,1.0,(50,205,50),2)
            
        cv2.imshow("img", img)#show everythings
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break    
    cap.release()
    cv2.destroyAllWindows()
