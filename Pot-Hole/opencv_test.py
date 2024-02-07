import cv2

# Load the video
video_path = 'Videos/demo.mp4'
cap = cv2.VideoCapture(video_path)

# Display each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    cv2.imshow('Frame', frame)
    if cv2.waitKey(25) == ord('q'):
        break

# Release resources and close windows
cap.release()
cv2.destroyAllWindows()