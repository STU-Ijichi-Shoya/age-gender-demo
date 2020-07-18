import cv2
import predict
import numpy as np




casfile="scripts/haarcascade_frontalface_default.xml"
weight_file="weights/eff-face-rate.hdf5"

face_cascade = cv2.CascadeClassifier(casfile)


FACE_RATE=0.799999999
DNN_IMAGE_SIZE=(200,200)    

DEBUG=False

dt=5

if not DEBUG:
    predict.load_model(weight_file)


def Face_paste2empty(src,width,height):
    ratio=DNN_IMAGE_SIZE[1]/height
    width_n=int(width*ratio)
    resized=cv2.resize(src,(width_n,DNN_IMAGE_SIZE[1]))
    base=np.full((*DNN_IMAGE_SIZE,3),255,dtype="float32")
    nx=int(0.5*(DNN_IMAGE_SIZE[0]-width_n))
    # print(resized.shape)
    base[0:DNN_IMAGE_SIZE[0],nx:width_n,0:3]=resized
    return base

def detect_draw(src,debug=False,r_dest=False):
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)    
    faces = face_cascade.detectMultiScale(src_gray,minNeighbors=10)

    if len(faces)<=0:
        if r_dest:
            return src,["none"]
        else:
            return src
    return_txt=[]
    resized_faces=[]
    for x,y,w,h in faces:
        Y=int(y-(h*0.5*FACE_RATE))
        if Y<0:
            Y=0
        face = src[Y: y + h, x: x + w]
        resized=Face_paste2empty(face,w,h)

        if DEBUG:
             cv2.imshow("debug_resize",resized)
             cv2.waitKey(1)
             return 0

        # resized=np.array(resized,dtype="float32")
        resized_faces.append(resized)
    if debug:
        return 0
    gender_list,genrate_list,age_list=predict.recognize_r(resized_faces)

    

    for gender,genrate,age,(x,y,w,h) in zip(gender_list,genrate_list,age_list,faces):
            color=(255,0,0) if gender=="M" else (0,0,255)
            cv2.rectangle(src,(x,y),(x+w,y+h),color,2)
            s="%s :%.2f  age: %.1f"%(gender,genrate,age)
            # draw str
            cv2.putText(src,s,(x,y-dt),cv2.FONT_HERSHEY_SIMPLEX,1.0,color)
            return_txt.append(s)
    
    if r_dest:
        return src,return_txt
    
    return src

if __name__ == "__main__":
    v=cv2.VideoCapture("endou.mp4")
    # img=cv2.imread("cache/3.png")
    # print(img)
    r,f=v.read()
    while r:
        detect_draw(f,debug=True)
        r,f=v.read()

