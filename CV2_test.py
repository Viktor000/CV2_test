import cv2
import numpy as np
image_path = "C:\\Users\\v_varich\\Downloads\\clock1.jpg"
face_cascade = cv2.CascadeClassifier('C:\\Users\\v_varich\\AppData\\Roaming\\Python\\Python38\\site-packages\\cv2\\data\\haarcascade_frontalface_default.xml')
image = cv2.imread(image_path)
image2=image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,150,255,apertureSize = 3)
kernel = np.ones((2,2),np.uint8)
output = cv2.bitwise_not(edges)
def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX,mouseY = x,y
        cv2.circle(image2,(x,y),100,(255,0,0),-1)
        print(mouseX)
        print(mouseY)

#thresh = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel)

#faces = face_cascade.detectMultiScale(
#    gray,
#    scaleFactor= 1.1,
#    minNeighbors= 5,
#    minSize=(10, 10)
#    )
#faces_detected = "Faces: " + format(len(faces))
#print(faces_detected)
#for (x, y, w, h) in faces:
#    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
#dst = cv2.Mat()
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if len(contours) != 0:
    # draw in blue the contours that were founded
    cv2.drawContours(output, contours, -1, 255, 3)

    # find the biggest countour (c) by the area
    c = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    z=cv2.boundingRect(c)
    rect=cv2.minAreaRect(c)
    box =cv2.boxPoints(rect)
    box=np.int0(box)
    #pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
    #M = cv2.getPerspectiveTransform(np.float32(c), pts2)
    #dsize =  cv2.Size(output.rows, output.cols)
    #dst=output.copy()
    #cv.warpPerspective(output, dst, M, dsize, cv.INTER_LINEAR, cv2.BORDER_CONSTANT, cv2.Scalar())
    # draw the biggest contour (c) in green
    #cv2.rectangle(image2,(x,y),(x+w,y+h),(0,0,255),2) 
    #print(c.tolist()[1][0][0])
    x1=c.tolist()[0][0][0]
    y1=c.tolist()[0][0][1]
    #print(type(c.tolist()[1][0]))
    image2 = cv2.circle(image2, (x1,y1), radius=2, color=(0, 0, 255), thickness=-1)
    #cv2.drawContours(image2,[c],0,(0,255,0),2)



cv2.imshow("image2", image2)
cv2.setMouseCallback('image2',draw_circle)
#cv2.imshow("output", output)
#cv2.imshow("dst", dst)
cv2.waitKey(0)


