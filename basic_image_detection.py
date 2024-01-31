from ultralytics import YOLO
model = YOLO("Models/yolov8n.pt")
model.predict(source="Images/tawthars.jpg", save=True, conf=0.5, save_txt=True)
