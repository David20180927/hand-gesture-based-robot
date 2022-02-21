#!C:\Users\user\AppData\Local\Programs\Python\Python39\python.exe
"""The image first processed using the hsv method to threshold away anything that is not skin color
then transfer the image to mediapipe to mark out hand"""
import serial
import cvzone
from cvzone.HandTrackingModule import HandDetector
import cv2
import numpy as np
from math import degrees,acos,asin,atan,pow,sqrt,sin,cos,radians
import os
from PIL import Image # For handling the images
import time
"""AI"""
from keras.models import model_from_json#
from tensorflow.keras.models import load_model

'''general parameters'''
dirmodel = 'model/'
dirdata = 'dataset/'
dirimg = 'image/'
#this two list is to construct polynomial relationship
d_raw = [70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410]
d_raw_corres =[498, 452, 396, 352, 320, 298, 276, 256, 243, 228, 218, 199, 191, 178, 170, 167, 163, 157, 148, 147, 142, 140, 134, 130, 127, 124, 120, 117, 113, 110, 108, 107, 105, 103, 101]
'''variable initialization'''
ges_left,ges_right = '','' 
w_angle, n1lor,  = 0, 0
record_swipe, record_swipe_counter, counter, stick_counter = 0, 0, 0, 0
dis_or,dis = 0,0
A,B,C = np.polyfit(d_raw,d_raw_corres,2)
'''param initialization'''
error_img  = Image.open(r'image/error.png')
text_font = cv2.FONT_HERSHEY_COMPLEX
stick_center = [896,582] #the center of stick
tar_distance = 200 #swiping distance to change mode
enlargen = 30 #crop image enlargen param
hand_sen = 30 #stablelization of hand misdetect
ran = 5
stick_opos,current_stick_opos,stick_img = [0,0],[0,0],[stick_center[0],stick_center[1]]
'''AI switch'''
AI_ONOFF = 1
'''UI switch'''
UI_MODE = 0 #0 is UI off, 1 is UI normal on
'''CONTROL switch'''
control_mode = 3 #0 is chassis, 1 is robotic arm, 3 is pre-started value

#the data transmited outside code
#the first bit of d must be 1 to let robot start receive what coming next
d = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
"""Hands detection parameters"""
detector = HandDetector(detectionCon=0.8, maxHands=2, minTrackCon= 0.8)
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
        #d1 is fixed
        d[41] = int(self.d1) 
        #d2 and d3 is calculated
        d[42] = int(contrain(self.d2,0,170))
        d[43] = int(contrain(self.d3+140,0,130))
        #d4 is w_angle
        d[44] = int(contrain(self.d4+90,0,250))
        #d5 is turn
        d[45] = int(self.d5)
        #d6 is craw
        d[46] = int(contrain(self.d6-40,0,250))
        return d
    
class stablelizer():
    def __init__(self):
        self.store,self.data = 0,0
    def stablelize(self,data,sen = 10):
        self.data = data
        if abs(self.data - self.store) < sen:
            return self.store
        self.store = self.data
        return self.data
    def overshoot(self,data,max_sen = 20):
        self.data = data
        if abs(self.data - self.store)>max_sen:
            self.data = self.store - (max_sen/10)
            return self.data
        self.store = self.data
        return self.data
    
class hand_watchdog():
    def __init__(self):
        self.oldhand={'lmList': [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]], 'bbox': (0, 0, 0, 0), 'center': (0, 0), 'type': ''}
    def watchdog(self,hand,sen = 40):
        if self.oldhand['bbox'][2] == 0 and self.oldhand['bbox'][3] == 0:
            self.oldhand = hand
        n_area = hand['bbox'][2]*hand['bbox'][3]
        o_area = self.oldhand['bbox'][2]*self.oldhand['bbox'][3]
        if n_area<o_area and abs(n_area-o_area)>sen:
            o = self.oldhand
            self.oldhand = hand
            return o
        self.oldhand = hand
        return hand
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
        if press: self.color = (139,131,131)
        else: self.color = self.scolor
        return True
    def running(self,new_pos):
        self.sx, self.sy = new_pos[0],new_pos[1]
        return
    
class swiper():
    def __init__(self, tar_distance):
        self.tar_distance = tar_distance
        self.ready = Image.open(r'image/ready.png')
        self.one = Image.open(r'image/1.png')
        self.two = Image.open(r'image/2.png')
        self.three = Image.open(r'image/3.png')
        self.four = Image.open(r'image/4.png')
        self.five = Image.open(r'image/5.png')
        self.six = Image.open(r'image/6.png')
        self.show_img, self.ready_i, self.mode = None,  False, 3
    def draw(self,img):
        if self.ready_i: img.paste(self.ready, (0,0), mask = self.ready)
        if self.show_img: img.paste(self.show_img, (0,0), mask = self.show_img)
    def state(self, state,  Press = False): #state should be just raw x distance 
        state += 0.01
        if Press:
            self.ready_i = True
            if (self.tar_distance/state) > 6/1:
                self.show_img = self.one
            elif (self.tar_distance/state) > 6/2:
                self.show_img = self.two
            elif (self.tar_distance/state) > 6/3:
                self.show_img = self.three
            elif (self.tar_distance/state) > 6/4:
                self.show_img = self.four
            elif (self.tar_distance/state) > 6/5:
                self.show_img = self.five
            elif (self.tar_distance/state) > 6/6:
                self.show_img = self.six   
        else:
            self.ready_i = False
            self.show_img = None
        return
    def save_mode(self,mode):
        self.mode = mode
        return
    

class chassis_ui():
    def __init__(self,stick_pos):
        self.all_nor = Image.open(r'image/chassis_nor.png')
        self.turn_right = Image.open(r'image/turn_right.png')
        self.turn_left =  Image.open(r'image/turn_left.png')
        self.stick = Image.open(r'image/release.png')
        self.holded_stick = Image.open(r'image/hold.png')
        self.right_point = Image.open(r'image/pointer_right.png')
        self.left_point = Image.open(r'image/pointer_left.png')
        self.center = stick_pos
        '''param'''
        self.mode, self.stick_pos, self.hold, self.speed = 0, stick_pos, False, [0,0]
        self.left_min, self.left_max, self.right_min,self.right_max, self.r = 40,-45,40,-45, 455
    def draw(self,img):
        '''chassis layer'''
        img.paste(self.all_nor, (0,0), mask = self.all_nor) 
        ''' stick '''
        if self.hold:
            img.paste(self.holded_stick, (int(self.stick_pos[0]-18),int(self.stick_pos[1]-18)), mask = self.holded_stick)
        else:
            img.paste(self.stick, (self.stick_pos[0],self.stick_pos[1]), mask = self.stick)
        ''' mode '''
        if self.mode == 1: #left
            img.paste(self.turn_left, (0,0), mask = self.turn_left)
        elif self.mode == 2: #right
            img.paste(self.turn_right, (0,0), mask = self.turn_right)
        '''speed indicator'''
        # whrite hereeeeee@!!!!!!!!!!!!!
        speed_x = radians((abs(self.speed[0])/100*(self.right_max-self.right_min))+self.right_min) #
        speed_y = radians((abs(self.speed[1])/100*(self.left_max-self.left_min))+self.left_min) # #front
        left_x, left_y = self.center[0]-100, self.center[1]+55
        right_x, right_y = self.center[0]+200, self.center[1]+55
        #left
        img.paste(self.left_point, (left_x-int(self.r*cos(speed_y)),left_y+int(self.r*sin(speed_y))), mask = self.left_point) #left
        img.paste(self.right_point, (right_x+int(self.r*cos(speed_x)),left_y+int(self.r*sin(speed_x))), mask = self.right_point)#right
        
        
    def update(self,new_pos,speed,mode):
        #self.stick_pos[0],self.stick_pos[1] = new_pos[0],new_pos[1] will make change the value of new pos! 
        self.stick_pos = new_pos
        self.mode = mode
        self.speed = speed
        return    
        
        
class arm_ui():
    def __init__(self,wrist_pos = [0,0]): 
        self.all_nor = Image.open(r'image/background_robot_arm.png')
        self.craw = Image.open(r'image/craw.png')
        self.w_angle, self.point,self.hand = 0,[0,0],True
    def draw(self,img):
        img.paste(self.all_nor, (0,0), mask = self.all_nor) #arm layer
        if self.hand:
            img_craw = self.craw
            img_craw = img_craw.rotate(w_angle,expand = 1)
            img.paste(img_craw, (self.point[0]+200, self.point[1]+50), mask = img_craw) #arm layer
            
    def update(self,w_angle,point,hand):
        self.hand = hand
        self.w_angle = w_angle
        self.point = point
        
        return
        
        
        
'''class'''

transfer_data = data_handle()
stick_stable_x, stick_stable_y= stablelizer(),stablelizer()
d1_stable,d2_stable,d3_stable,d4_stable,d5_stable,d6_stable = stablelizer(),stablelizer(),stablelizer(),stablelizer(),stablelizer(),stablelizer()
Right_hand_watchdog, Left_hand_watchdog = hand_watchdog(),hand_watchdog()
chassis_ui, arm_ui = chassis_ui(stick_center), arm_ui([0,0])
ctrl_mode_swiper = swiper(tar_distance)
chassis_control_stick = stick([1000,300],150,[40,25],'Chassis',(255,0,0),(0,0,0),stick_center)
'''''''''''''''UI function'''''''''''''''
def img_reset():
    img = Image.open(r'image/black.png')
    return img

def img_basic_layer(img,control_mode):
    if control_mode == 0: #chassis
        chassis_ui.draw(img)
        #if ctrl_mode_swiper.mode == 1: time.sleep(2)
        ctrl_mode_swiper.save_mode(0)
    elif control_mode == 1: #arm
        arm_ui.draw(img)
        #if ctrl_mode_swiper.mode == 0: time.sleep(2)
        ctrl_mode_swiper.save_mode(1)
    else: return
    return
    
def img_swiper_layer(img):
    ctrl_mode_swiper.draw(img)
    return

def control_screen_load_chassis(img,current_stick_opos):
    img = chassis_ui.draw_chassis(img,current_stick_opos[1]/20,current_stick_opos[0]/20)
    chassis_ui.hold = False
    return img

def control_screen_load_arm(img,w_angle):
    img = draw_arm(img,w_angle)
    return img  
'''''''''''''''UI function'''''''''''''''
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
    if data < min_d: return min_d
    elif data> max_d:  return max_d
    else: return data
    
def connect_robot():
    for a in range(6):
        com = 'COM'+str(a)
        try:  s = serial.Serial(com)
        except:  continue
        else: return s 
        

def hand_is_swiping(finger):
    if finger[1]==1 and finger[2]==1 and finger[3]==1 and finger[4]==1:  return True
    else: return  False
    return False
    
def measure_distance(lmlistRight_hand,A,B,C):
    x1,y1 = lmlistRight_hand[5]
    x2,y2 = lmlistRight_hand[17]
    distance = int(sqrt((y2-y1)**2+(x2-x1)**2))
    dis =int( A * distance**2 + B * distance +C)
    return dis

def run_hand_detection(img):  
    # Find the hand and its landmark
    if(img is not None):
        #hands, img = detector.findHands(img)#draw
        hands = detector.findHands(img, draw=False)#notdraw
        if hands:
            hand1 = hands[0]
            hand1_type = hand1["type"]
            if len(hands) == 2:
                hand2 = hands[1]
                hand2_type = hand2["type"]
                if hand1_type == 'Left':  return img,hand1,hand2
                else: return img,hand2,hand1
            if hand1_type == 'Left': return img,hand1,None
            else:  return img,None,hand1
        else: return img,None,None 
    else: return img,None,None
    
def toggle(control):
    if control == 1: return 0
    elif control == 0:  return 1
    return 0

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
        angle = ((y2-w_y)/mid_dy*90)
    else:
        angle = 0.0
    return angle

def img_preprocess(img):
    img = cv2.flip(img,1)#lets flip  
    return img

def arm_middle(transfer_data):
    transfer_data.d1 = 0
    transfer_data.d2 = 45
    transfer_data.d3 = 100
    transfer_data.d4 = 45
    transfer_data.d5 = 0
    transfer_data.d6 = 100
    return 

def calculation(x,y,a1 = 10.4, a2 = 12.7):
    
    '''
    x(w2,h2)
        w, h  = abs(w2-w1), abs(h2 - h1)
    
    
                      x(w1,h1)
    '''
    #lmlist not exceed the box
    x,y = contrain(x,480,1150),contrain(y,250,600) # cv value of lmlist[0]
    #from lmlist[0] to scale
    x = ((x-480)/670*10)+7.5
    y = ((y-250)/350*26.5)-12.5
    #move cartesian
    y = -y
    x = x-25
    y = y+1.5
     #flip cartesian
    x = -x
    
    x,y = contrain(x,7.5,17.5),contrain(y,-12.5,14) #the true scale of robotic arm
    d0,a0,d1,d2 = 0,0,0,0
    d0 = degrees(atan(y/x))
    a0 = sqrt(pow(x,2)+pow(y,2))

    d1 = degrees(-acos(contrain((pow(x,2)+pow(y,2)+pow(a1,2)-pow(a2,2))/(2*a0*a1),0,1)))+d0
    d2 = degrees(+acos(contrain((pow(x,2)+pow(y,2)+pow(a2,2)-pow(a1,2))/(2*a0*a2),0,1)))+d0-d1
    if (d1 <= 0) or (d1 > 90) or (d2 > -20) or (d2 < -140):
        d1 = degrees(+acos(contrain((pow(x,2)+pow(y,2)+pow(a1,2)-pow(a2,2))/(2*a0*a1),0,1)))+d0
        d2 = degrees(-acos(contrain((pow(x,2)+pow(y,2)+pow(a2,2)-pow(a1,2))/(2*a0*a2),0,1)))+d0-d1    
    #the contrain of output degree must be widen
    d1 = contrain(d1,0,170)#0,90
    d2 = contrain(d2,-140,-10)#-140,-20

    return d1,d2


def data_reset():
    #lor,n1lor
    return 0,0,False,1, [0,0]

def load_model(name):
    print("Loading pretrained model: "+ str(name)+"\n")
    json_file = open(dirmodel+str(name)+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(dirmodel+str(name)+'.h5')
    print("Loaded "+str(name)+" from disk\n")
    return loaded_model

def show_everything(img,ui_mode):
    if not ui_mode:
        cv2.imshow('UI',img)
    else:
        open_cv_image = np.array(img) 
        open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)        
        cv2.imshow('UI',open_cv_image)#show everythings
        
def control_screen_load_chassis_none(img,current_stick_opos):
    img = chassis_control_stick.draw(img)
    img = cv2.putText(img,"Speed x: "+str(current_stick_opos[1]/20)+' m/s',(450,30),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2)
    img = cv2.putText(img,"Speed y: "+str(current_stick_opos[0]/20)+' m/s',(450,60),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2)
    cv2.rectangle(img,(440,2), (760,70),(181,228,255),2)
    chassis_control_stick.pressing(False)
    return img

"""""""""""""""Main loop"""""""""""""""
if __name__ == "__main__":
    '''connect usb/bluetooth'''
    s = connect_robot()
    loaded_model = load_model('model')
    loaded_model1 = load_model('model1')
    label = {0:'craw',1:'left_craw',2:'right_craw'}
    label1 = {0:'fist',1:'left_fist',2:'right_fist'}
    print("Opening camera\n")
    cap = cv2.VideoCapture(0) #capture video
    cap.set(3,1366) 
    cap.set(4,500)
    cv2.namedWindow("UI", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("UI", 1920,1080)
    if not UI_MODE: stick_center = [1000,300]
    while True: 

        success, raw_img = cap.read()
        if not success: continue
        '''draw ui using updated info'''
        raw_img = img_preprocess(raw_img)
        if UI_MODE:
            img = img_reset() #layer 0
            img_basic_layer(img,control_mode) #layer 1, decide chassis or arm, do nothing if other then 0,1
            img_swiper_layer(img) #layer 2, swiper layer
            '''reset'''
            ctrl_mode_swiper.state(0,False)
            chassis_ui.update(stick_center,current_stick_opos,0)
            arm_ui.update(w_angle,[0,0],False)     
            '''reset'''
        else: 
            img = raw_img
            if control_mode == 0: cv2.putText(img, 'chassis', (100,100),text_font, 2, (255,0,0),3)
            elif control_mode == 1: cv2.putText(img, 'arm',(100,100),text_font, 2, (0,0,255),3)
            else: cv2.putText(img, 'start', (100,100),text_font, 2, (0,0,255),3)
        
        '''raw img is for detection, img is for drawing'''

        #raw_img, Left_hand,Right_hand = run_hand_detection(raw_img)#detect hands, Return None if failed   
        raw_img, Right_hand, Left_hand = run_hand_detection(raw_img) #Flip control Hands
        lor, n1lor,  chassis_ui.hold, transfer_data.test,  current_stick_opos = data_reset() #0,0,False,1, [0,0]
        arm_middle(transfer_data)
        '''IF HAND IS DETECTED'''
        if Left_hand or Right_hand: 
            if Left_hand:
                #Left_hand = Left_hand_watchdog.watchdog(Left_hand,hand_sen)
                lmlistLeft_hand, drawnboxLeft_hand, centerpointLeft_hand, handtypeLeft_hand, fingerLeft_hand = Left_hand["lmList"], Left_hand["bbox"], Left_hand["center"], Left_hand["type"],detector.fingersUp(Left_hand)
                '''AI'''
                cropleft = raw_img[drawnboxLeft_hand[1]-enlargen:drawnboxLeft_hand[1]+drawnboxLeft_hand[3]+enlargen,drawnboxLeft_hand[0]-enlargen:drawnboxLeft_hand[0]+drawnboxLeft_hand[2]+enlargen]
                if (cropleft.shape[0] != 0 and cropleft.shape[1] != 0 and cropleft.shape[2] == 3 and AI_ONOFF == 1): #prevent crash
                    x = preprocess_bf_handover(cropleft)
                    if control_mode == 0: ges_left = find_label((loaded_model1.predict(x) > 0.5).astype("int32"),label1)
                    elif control_mode == 1: ges_left = find_label((loaded_model.predict(x) > 0.5).astype("int32"),label)
                    cv2.putText(img, str(ges_left), (drawnboxLeft_hand[0]-30,drawnboxLeft_hand[1]-30), text_font , 1, (255,0,0) , 2, cv2.LINE_AA)
            if Right_hand:
                #Right_hand = Right_hand_watchdog.watchdog(Right_hand,hand_sen)
                lmlistRight_hand, drawnboxRight_hand, centerpointRight_hand, handtypeRight_hand, fingerRight_hand = Right_hand["lmList"], Right_hand["bbox"], Right_hand["center"], Right_hand["type"],detector.fingersUp(Right_hand)
                '''AI'''
                cropright = raw_img[drawnboxRight_hand[1]-enlargen:drawnboxRight_hand[1]+drawnboxRight_hand[3]+enlargen,drawnboxRight_hand[0]-enlargen:drawnboxRight_hand[0]+drawnboxRight_hand[2]+enlargen]
                if (cropright.shape[0] != 0 and cropright.shape[1] != 0 and cropright.shape[2] == 3 and AI_ONOFF == 1):
                    x = preprocess_bf_handover(cropright)
                    if control_mode == 0: ges_right = find_label((loaded_model1.predict(x) > 0.5).astype("int32"),label1)   
                    elif control_mode == 1: ges_right = find_label((loaded_model.predict(x) > 0.5).astype("int32"),label)
                    cv2.putText(img, str(ges_right), (drawnboxRight_hand[0]-30,drawnboxRight_hand[1]-30), text_font , 1, (255,0,0) , 2, cv2.LINE_AA)                    
            '''control mode swiper'''
            if Right_hand and hand_is_swiping(fingerRight_hand):
                if record_swipe_counter == 0:
                    record_swipe = lmlistRight_hand[5][0]
                    record_swipe_counter = 1
                    ctrl_mode_swiper.state(0,True)
                else:
                    ctrl_mode_swiper.state(lmlistRight_hand[5][0]-record_swipe,True)
                    if(lmlistRight_hand[5][0]-record_swipe)>=tar_distance:
                        control_mode = toggle(control_mode)
                        record_swipe_counter = 0
                        continue
            else:
                record_swipe_counter = 0
                ctrl_mode_swiper.state(0,False)
            '''chassis control when control node = 0'''
            if control_mode == 0:
                if Right_hand and ges_right and fingerRight_hand[1]==0 and lmlistRight_hand[5][1] < lmlistRight_hand[15][1]:
                    if stick_counter == 0:
                        stick_opos = centerpointRight_hand
                        stick_counter += 1
                        dis_or = int(measure_distance(lmlistRight_hand,A,B,C))
                        chassis_ui.hold = True
                    else:
                        dis = int(measure_distance(lmlistRight_hand,A,B,C))
                        current_stick_opos = [contrain(int((centerpointRight_hand[0] - stick_opos[0])/3),-100,100), contrain(-(dis - dis_or),-100,100)]#(dx,dy)             
                        '''stablelization process'''
                        
                        current_stick_opos[0], current_stick_opos[1] = stick_stable_x.stablelize(current_stick_opos[0],10),  stick_stable_y.stablelize(current_stick_opos[1],10)
                        
                        if current_stick_opos[0]<15 and current_stick_opos[0]>-20 : #makesure it goes straight line
                            current_stick_opos[0] = 0
                        if current_stick_opos[1]<15 and current_stick_opos[1]>-20 : #makesure it goes straight line
                            current_stick_opos[1] = 0
                        
                        stick_img[0], stick_img[1] = stick_center[0] + int(current_stick_opos[0]*1.9), stick_center[1] - int(current_stick_opos[1]*1.9)
                        if UI_MODE:
                            chassis_ui.update(stick_img,current_stick_opos,0) #youuuuuu
                            chassis_ui.hold = True
                        else: 
                            chassis_control_stick.running(stick_img)
                            chassis_control_stick.pressing(True)
                        
                    if ges_right == 'left_fist':
                        transfer_data.wz_set_order = 50
                        if UI_MODE: chassis_ui.update(stick_img,current_stick_opos,1)
                        else: cv2.putText(img,"Turning Left",(850,30),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2)
                    elif ges_right == 'right_fist':
                        transfer_data.wz_set_order = -50
                        if UI_MODE: chassis_ui.update(stick_img,current_stick_opos,2)
                        else: cv2.putText(img,"Turning Right",(850,30),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2)
                else:
                    stick_counter = 0
                    stick_opos, current_stick_opos = (0,0), [0,0]
                    if UI_MODE:
                        chassis_ui.update(stick_center,current_stick_opos,0)
                        chassis_ui.hold = False
                    else: 
                        chassis_control_stick.running(stick_center)
                if not UI_MODE: img = control_screen_load_chassis_none(img,current_stick_opos)   
            if control_mode == 1: #robotic arm mode
                if Right_hand and ges_right: #and craw
                    if lmlistRight_hand[12][0]>lmlistRight_hand[4][0]:
                        '''n1'''
                        #n1lor  = 0#forever 900?
                        '''n2,n3'''
                        d2,d3 = calculation(lmlistRight_hand[0][0],lmlistRight_hand[0][1])
                        d2 = d2_stable.overshoot(d2_stable.stablelize(d2,5),10)
                        d3 = d3_stable.overshoot(d3_stable.stablelize(d3,5),10)  
                        '''n4'''
                        w_angle = d4_stable.stablelize(get_wrist_angle(drawnboxRight_hand,lmlistRight_hand[0],ran),3)
                        w_angle = d4_stable.overshoot(w_angle,8)
                        #0 = none, 1 = left, 2 = right
                        '''use if (ges_left) /(ges_right) to determine lor value'''
                        '''n6'''
                        l,_ = detector.findDistance(lmlistRight_hand[4],lmlistRight_hand[8])
                        index = contrain((8500/(drawnboxRight_hand[2]*drawnboxRight_hand[3]))+1,1,2)
                        l = int(contrain(l*1.3*index,40,290)) #interger between 40 290
                        '''n5'''
                        transfer_data.d1 =  n1lor
                        transfer_data.d2 =  d2
                        transfer_data.d3 =  d3        
                        transfer_data.d4 =  w_angle   
                        transfer_data.d6 =  l
                        if not hand_is_swiping(fingerRight_hand): arm_ui.update(w_angle,[lmlistRight_hand[0][0],lmlistRight_hand[0][1]],True)
                    else:
                        if ges_right == 'left_craw': lor = 1
                        elif ges_right == 'right_craw': lor = 2       
                        else: lor = 0
                        transfer_data.d5 =  lor
       
        '''send data to robot'''     
        transfer_data.vx_set_order = current_stick_opos[1]
        transfer_data.vy_set_order = current_stick_opos[0]        
        d = transfer_data.process_data(d)
        try:
            s.write(d) #connect robot
        except:
            if s: s.close()
            s = connect_robot()
            if UI_MODE: img.paste(error_img, (0,0), mask = error_img) #layer 3, comment it if disable self checking
            
        show_everything(img,UI_MODE) #throw in pil image
        if cv2.waitKey(1) & 0xFF == ord("q"): #press q to stop program
            break            

    cap.release()
    cv2.destroyAllWindows()
