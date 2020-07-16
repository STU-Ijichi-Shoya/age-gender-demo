import cv2
import predict
casfile="haarcascade_frontalface_default.xml"


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

DNN_IMAGE_SIZE=(200,200)    
dt=5
## Attenssion ##
## regression model only 
## [gender-rate , age] scalar only
predict.load_model("eff-im4_r_down-jpn-cn-limit-35-50.hdf5")
def detect_draw(src):
    faces = face_cascade.detectMultiScale(src,minNeighbors=6)
    if len(faces)<=0:
        return src
    # select face of max
    x, y, w, h = max(faces,key=lambda x:x[2])
    
    face = src[y: y + h, x: x + w]
    resized=cv2.resize(face,DNN_IMAGE_SIZE)
    resized=cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)
    gender,genrate,age=predict.recognize_r(resized)
    print(gender,genrate,age)
    color=(255,0,0) if gender=="M" else (0,0,255)
    cv2.rectangle(src,(x,y),(x+w,y+h),color,2)

    # draw str
    cv2.putText(src,"%s:%.1f%  %.1f"%(gender,genrate,age),(x,y-dt),cv2.FONT_HERSHEY_COMPLEX,1.0,color)

    return src

def main():
    import sys
    v=cv2.VideoCapture("http://192.168.0.14:4747/video")
    if not v.isOpened():
         print("er! can't open video",file=sys.stderr)
         return
    ret,frame=v.read()
    while ret:
        img=detect_draw(frame)
        cv2.imshow("age estimator for demo",img)
        cv2.waitKey(1)
        ret,frame=v.read()
    
    del v
main()




    
