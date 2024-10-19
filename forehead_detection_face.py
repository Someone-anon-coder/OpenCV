import cv2

window_width = 640
window_height = 480

streamurl = "https://192.168.28.42:4343/video"

def get_positions(cascade_filename: str, frame: cv2.VideoCapture) -> tuple[int, int, int, int]:
    """Get position of cascades detected in the frame

    Args:
        cascade_filename (str): Name of the cascade file

    Returns:
        tuple[int, int, int, int]: Position of the cascades
    """
    
    cascade = cv2.CascadeClassifier(cascade_filename)
    object_postitions = cascade.detectMultiScale(frame, 1.3, 5)
    
    return object_postitions

def detect_forehead(face_position: tuple) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    """Detect forehead in the frame

    Args:
        face_position (tuple): Position of the face

    Returns:
        tuple: Position of the forehead and face
    """
    
    face_x, face_y, face_w, face_h = face_position

    forhead_position_x = int(face_x + (face_w // 2))
    forhead_position_y = int(face_y + (face_w * (2/10)))
    
    return (forhead_position_x, forhead_position_y), face_position

cam = cv2.VideoCapture(0)
# cam = cv2.VideoCapture(streamurl)

while True:
    _, frame = cam.read()
    
    frame = cv2.resize(frame, (window_width, window_height))
    frame = cv2.flip(frame, 1)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    face_positions = get_positions("xml_files/haarcascade_frontalface_default.xml", gray_frame)
    
    for face_position in face_positions:
        
        forhead_position, face_position = detect_forehead(face_position)
        
        if len(forhead_position) != 0:
            forehead_position_x, forehead_position_y = forhead_position
            face_x, face_y, face_w, face_h = face_position

            cv2.rectangle(frame, (face_x, face_y), (face_x + face_w, face_y + face_h), (0, 255, 0), 2)

            cv2.line(frame, (forehead_position_x, 0), (forehead_position_x, window_height), (0, 0, 255), 2)
            cv2.line(frame, (0, forehead_position_y), (window_width, forehead_position_y), (0, 0, 255), 2)
            
            cv2.circle(frame, (forehead_position_x, forehead_position_y), 5, (0, 0, 255), -1)
            cv2.circle(frame, (forehead_position_x, forehead_position_y), 30, (0, 0, 255), 2)
    
    cv2.imshow("Webcam", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()