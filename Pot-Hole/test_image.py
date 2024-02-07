from PIL import Image, ImageDraw
from roboflow import Roboflow

# Initialize Roboflow model
rf = Roboflow(api_key="Y1uEhWI1EFvz6IKyqn38")
project = rf.workspace().project("pothole-detection-yolov8-jcnms")
model = project.version(1).model

# Predict on a local image
result = model.predict("Images/test2.jpg", confidence=40, overlap=30).json()
print(result)

# Extract predictions
predictions = result['predictions']

# Load the image
image = Image.open('Images/test2.jpg')
draw = ImageDraw.Draw(image)

# Draw bounding box for each detection
for pred in predictions:
    x_center, y_center = pred['x'], pred['y']
    width, height = pred['width'], pred['height']
    # Calculate the top left and bottom right corners
    x1 = x_center - (width / 2)
    y1 = y_center - (height / 2)
    x2 = x_center + (width / 2)
    y2 = y_center + (height / 2)

    # Draw the rectangle
    draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    # Optional: Put the label with the confidence
    label = f"{pred['class']} {pred['confidence']:.2f}"
    draw.text((x1, y1 - 10), label, fill="red")

# Save the image with the bounding boxes
image.show()

# Display the image with bounding boxes (optional, for Jupyter notebooks)
# display(image)
