import random
import cv2

window_width = 640
window_height = 480

rectangle_size = 50
current_x_pos = 0
current_y_pos = 0
current_end_x_pos = current_x_pos + rectangle_size
current_end_y_pos = current_y_pos + rectangle_size

COLOR = (0, 255, 0)

previous_x_pos = 0
previous_y_pos = 0
previous_end_x_pos = previous_x_pos + rectangle_size
previous_end_y_pos = previous_y_pos + rectangle_size

change_pos_step = 10
move_directions = ['left-up', 'left-down', 'right-up', 'right-down']
move_direction = move_directions[3]

cam = cv2.VideoCapture(0)
while True:
    _, frame = cam.read()
    
    frame = cv2.resize(frame, (window_width, window_height))
    frame = cv2.flip(frame, 1)
    
    if move_direction == move_directions[0]:
        current_x_pos -= change_pos_step
        current_end_x_pos -= change_pos_step
        current_y_pos -= change_pos_step
        current_end_y_pos -= change_pos_step
    
    elif move_direction == move_directions[1]:
        current_x_pos -= change_pos_step
        current_end_x_pos -= change_pos_step
        current_y_pos += change_pos_step
        current_end_y_pos += change_pos_step
    
    elif move_direction == move_directions[2]:
        current_x_pos += change_pos_step
        current_end_x_pos += change_pos_step
        current_y_pos -= change_pos_step
        current_end_y_pos -= change_pos_step
    
    elif move_direction == move_directions[3]:
        current_x_pos += change_pos_step
        current_end_x_pos += change_pos_step
        current_y_pos += change_pos_step
        current_end_y_pos += change_pos_step
    
    if current_x_pos == 0 or current_y_pos == 0 or current_end_x_pos == window_width or current_end_y_pos == window_height:
        if (current_x_pos == 0 and previous_y_pos < current_y_pos) or (current_y_pos == 0 and previous_x_pos < current_x_pos):
            move_direction = move_directions[3]
        
        elif (current_x_pos == 0 and previous_y_pos > current_y_pos) or (current_end_y_pos == window_height and previous_end_x_pos < current_end_x_pos):
            move_direction = move_directions[2]
        
        elif (current_y_pos == 0 and previous_x_pos > current_x_pos) or (current_end_x_pos == window_width and previous_end_y_pos < current_end_y_pos):
            move_direction = move_directions[1]
        
        elif (current_end_x_pos == window_width and previous_end_y_pos > current_end_y_pos) or (current_end_y_pos == window_height and previous_end_x_pos > current_end_x_pos):
            move_direction = move_directions[0]

        COLOR = random.choices(range(256), k=3)
        
    frame = cv2.rectangle(frame, (current_x_pos, current_y_pos), (current_end_x_pos, current_end_y_pos), COLOR, -1)
    cv2.imshow("Webcam", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break
    
    previous_x_pos = current_x_pos
    previous_y_pos = current_y_pos
    previous_end_x_pos = current_end_x_pos
    previous_end_y_pos = current_end_y_pos

cam.release()
cv2.destroyAllWindows()