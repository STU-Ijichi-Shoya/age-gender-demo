import cv2
import predict,detect


VIDEO_SOURCE="endou.mp4"
DNN_IMAGE_SIZE=(200,200)    
dt=5
## Attenssion ##
## regression model only 
## model.predict --> [gender-rate , age] scalar only
predict.load_model("weights/eff-im6_down.hdf5")


import time
def main():
    import sys
    v=cv2.VideoCapture(VIDEO_SOURCE)
    if not v.isOpened():
         print("er! can't open video",file=sys.stderr)
         return
    ret,frame=v.read()
    while ret:
        s=time.time()
        img=detect.detect_draw(frame,r_dest=False)
        fps=1/(time.time()-s)
        print(f"\rfps={fps}",end="")
        cv2.imshow("age estimator for demo",img)
        cv2.waitKey(1)

        ret,frame=v.read()
        del img
    del v
main()




    
