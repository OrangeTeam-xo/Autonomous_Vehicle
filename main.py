import cv2
import numpy as np

def detect_color(frame, lower_bound, upper_bound):
    """Returns mask and contours for given color bounds."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return mask, contours

# Red has two HSV ranges
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])

lower_green = np.array([40, 70, 70])
upper_green = np.array([80, 255, 255])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get masks and contours
    _, red_contours1 = detect_color(frame, lower_red1, upper_red1)
    _, red_contours2 = detect_color(frame, lower_red2, upper_red2)
    _, green_contours = detect_color(frame, lower_green, upper_green)

    center_x = frame.shape[1] // 2
    center_y = frame.shape[0] // 2
    cv2.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)  # Draw center point

    # Process red contours
    for contour in red_contours1 + red_contours2:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            if abs((x + w//2) - center_x) < 50 and abs((y + h//2) - center_y) < 50:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Red Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Process green contours
    for contour in green_contours:
        area = cv2.contourArea(contour)
        if area > 500:
            x, y, w, h = cv2.boundingRect(contour)
            if abs((x + w//2) - center_x) < 50 and abs((y + h//2) - center_y) < 50:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, "Green Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Color Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
