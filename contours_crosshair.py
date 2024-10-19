import cv2
import numpy

window_width = 480
window_height = 320

cv2.namedWindow("Webcam")
cv2.namedWindow("Composite")
cv2.namedWindow("Trackbar")

cv2.createTrackbar("Low_Hue1", "Trackbar", 50, 179, lambda _: None)
cv2.createTrackbar("High_Hue1", "Trackbar", 179, 179, lambda _: None)
cv2.createTrackbar("Low_Hue2", "Trackbar", 50, 179, lambda _: None)
cv2.createTrackbar("High_Hue2", "Trackbar", 179, 179, lambda _: None)
cv2.createTrackbar("Low_Sat", "Trackbar", 50, 255, lambda _: None)
cv2.createTrackbar("High_Sat", "Trackbar", 255, 255, lambda _: None)
cv2.createTrackbar( "Low_Val", "Trackbar", 50, 255, lambda _: None)
cv2.createTrackbar( "High_Val", "Trackbar", 255, 255, lambda _: None)

cv2.moveWindow("Webcam", 0, 0)
cv2.moveWindow("Composite", window_width, 0)
cv2.moveWindow("Trackbar", window_width * 3, 0)

# stream_url = "https://192.168.190.42:4343"

# cam = cv2.VideoCapture(stream_url)
cam = cv2.VideoCapture(0)
while True:
    _, frame = cam.read()
    
    frame = cv2.resize(frame, (window_width, window_height))
    frame = cv2.flip(frame, 1)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    High_Hue1 = cv2.getTrackbarPos("High_Hue1", "Trackbar")
    Low_Hue1 = cv2.getTrackbarPos("Low_Hue1", "Trackbar")
    High_Hue2 = cv2.getTrackbarPos("High_Hue2", "Trackbar")
    Low_Hue2 = cv2.getTrackbarPos("Low_Hue2", "Trackbar")
    High_Sat = cv2.getTrackbarPos("High_Sat", "Trackbar")
    Low_Sat = cv2.getTrackbarPos("Low_Sat", "Trackbar")
    High_Val = cv2.getTrackbarPos("High_Val", "Trackbar")
    Low_Val = cv2.getTrackbarPos("Low_Val", "Trackbar")
    
    lower_bound1 = numpy.array([Low_Hue1, Low_Sat, Low_Val])
    upper_bound1 = numpy.array([High_Hue1, High_Sat, High_Val])
    
    lower_bound2 = numpy.array([Low_Hue2, Low_Sat, Low_Val])
    upper_bound2 = numpy.array([High_Hue2, High_Sat, High_Val])

    mask1 = cv2.inRange(hsv, lower_bound1, upper_bound1)
    mask2 = cv2.inRange(hsv, lower_bound2, upper_bound2)
    FG_mask = cv2.add(mask1, mask2)
    BG_mask = cv2.bitwise_not(FG_mask)
    
    FG = cv2.bitwise_and(frame, frame, mask=FG_mask)
    BG = cv2.cvtColor(BG_mask, cv2.COLOR_GRAY2BGR)
    
    composite_frame = cv2.add(FG, BG)
    
    contours = cv2.findContours(FG_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    
    for contour in contours:
        if cv2.contourArea(contour) > 50:
            x, y, w, h = cv2.boundingRect(contour)
            print(f"X: {x}, Y: {y}, W: {w}, H: {h}")
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.line(frame, (x + w // 2, 0), (x + w // 2, window_height), (255, 0, 0), 2)
            cv2.line(frame, (0, y + h // 2), (window_width, y + h // 2), (255, 0, 0), 2)
    
    cv2.imshow("Webcam", frame)
    cv2.imshow("Composite", composite_frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()