import cv2
import numpy as np

def onChange(x):
    pass

def setting_bar():
    cv2.namedWindow('HSV_settings')

    cv2.createTrackbar('H_MAX', 'HSV_settings', 0, 255, onChange)
    cv2.setTrackbarPos('H_MAX', 'HSV_settings', 255)
    cv2.createTrackbar('H_MIN', 'HSV_settings', 0, 255, onChange)
    cv2.setTrackbarPos('H_MIN', 'HSV_settings', 0)
    cv2.createTrackbar('S_MAX', 'HSV_settings', 0, 255, onChange)
    cv2.setTrackbarPos('S_MAX', 'HSV_settings', 255)
    cv2.createTrackbar('S_MIN', 'HSV_settings', 0, 255, onChange)
    cv2.setTrackbarPos('S_MIN', 'HSV_settings', 0)
    cv2.createTrackbar('V_MAX', 'HSV_settings', 0, 255, onChange)
    cv2.setTrackbarPos('V_MAX', 'HSV_settings', 255)
    cv2.createTrackbar('V_MIN', 'HSV_settings', 0, 255, onChange)
    cv2.setTrackbarPos('V_MIN', 'HSV_settings', 0)
    cv2.createTrackbar('binary', 'HSV_settings', 0, 255, onChange)
    cv2.setTrackbarPos('binary', 'HSV_settings', 0)

setting_bar()


def showimage():
    try:
        video = cv2.VideoCapture("2.mp4")
        print('open video')
    except:
        print ('Not working')
        return
   
    frame_counter = 0
    while True:
        ret, frame = video.read()
        frame = cv2.resize(frame,dsize=(640,480))
        frame_counter += 1
        #If the last frame is reached, reset the capture and the frame_counter
        if frame_counter == video.get(cv2.CAP_PROP_FRAME_COUNT):
            frame_counter = 0 #Or whatever as long as it is the same as next line
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        H_MAX = cv2.getTrackbarPos('H_MAX', 'HSV_settings')
        H_MIN = cv2.getTrackbarPos('H_MIN', 'HSV_settings')
        S_MAX = cv2.getTrackbarPos('S_MAX', 'HSV_settings')
        S_MIN = cv2.getTrackbarPos('S_MIN', 'HSV_settings')
        V_MAX = cv2.getTrackbarPos('V_MAX', 'HSV_settings')
        V_MIN = cv2.getTrackbarPos('V_MIN', 'HSV_settings')
        
        #find HSV
        lower = np.array([0,0,139])
        higher = np.array([255,255,255])
        #lower = np.array([H_MIN, S_MIN, V_MIN])
        #higher = np.array([H_MAX, S_MAX, V_MAX])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        Gmask = cv2.inRange(hsv, lower, higher)
        G = cv2.bitwise_and(frame, frame, mask = Gmask)

        #blur
        image_blur = cv2.GaussianBlur(G, (0, 0), 3)
        
        #binaryimage
        gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
        #dst = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        dst = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
        
        copy_video = frame.copy()

        #boundingbox
        idx=0
        contours = cv2.findContours(dst,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(copy_video,(x,y),(x+w,y+h),(200,0,0),2)
            cv2.circle(copy_video,((x+x+w)//2,(y+y+h)//2),1,(200,0,0),2)
            idx +=1
        
        cv2.imshow('G',G)
        cv2.imshow('dst',dst)
        cv2.imshow('video',frame)
        cv2.imshow('cam_load',copy_video)
        copy_video = frame.copy()
        if cv2.waitKey(30) == ord('q'):
            break
    
    video.release()
    cv2.destroyAllWindows()

showimage()