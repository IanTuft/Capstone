import cv2
import numpy as np

def draw_grid(frame):
    height, width, _ = frame.shape
    
    third_width = width // 3
    third_height = height // 3
    
    cv2.line(frame, (third_width, 0), (third_width, height), (0, 0, 255), 2)
    cv2.line(frame, (2 * third_width, 0), (2 * third_width, height), (0, 0, 255), 2)
    
    cv2.line(frame, (0, third_height), (width, third_height), (0, 0, 255), 2)
    cv2.line(frame, (0, 2 * third_height), (width, 2 * third_height), (0, 0, 255), 2)
    
    return frame, third_width, third_height

def get_grid_positions(x, y, w, h, third_width, third_height):
    grid_positions = set()
    
    corners = [
        (x, y),     
        (x + w, y),   
        (x, y + h),     
        (x + w, y + h)   
    ]
    
    for corner_x, corner_y in corners:
        if corner_x < third_width:
            col = 0
        elif corner_x < 2 * third_width:
            col = 1
        else:
            col = 2
            
        if corner_y < third_height:
            row = 0
        elif corner_y < 2 * third_height:
            row = 1
        else:
            row = 2
            
        grid_number = row * 3 + col + 1
        grid_positions.add(grid_number)
    
    return grid_positions

def detect_orange_circle(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # This is where I'm defining the orange color range. You can modify this to whatever color ranges you like.
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([15, 255, 255])
    
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 500:  
            x, y, w, h = cv2.boundingRect(max_contour)
            center = (x + w//2, y + h//2)
            return center, (x, y, w, h)
    
    return None, None

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error reading")
        break
    
    frame = cv2.flip(frame, 1)
    
    frame, third_width, third_height = draw_grid(frame)
    
    center, bbox = detect_orange_circle(frame)
    
    if center is not None and bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        
        grid_positions = get_grid_positions(x, y, w, h, third_width, third_height)
        grid_text = f"Grids: {sorted(grid_positions)}"
        cv2.putText(frame, grid_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Grid detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()