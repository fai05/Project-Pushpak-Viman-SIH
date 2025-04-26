'''import cv2
import tflite_runtime.interpreter as tflite
import numpy as np




# Load TensorFlow Lite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Initialize OpenCV for capturing video
cap = cv2.VideoCapture('/home/siddharth/Documents/ProjectMayura/testpicture')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the image (resize and convert to RGB)
    input_image = cv2.resize(frame, (224, 224))  # Adjust size according to the model
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = np.expand_dims(input_image, axis=0).astype(np.float32) / 255.0
    
    # Set TensorFlow Lite model input
    interpreter.set_tensor(input_details[0]['index'], input_image)
    
    # Run inference
    interpreter.invoke()
    
    # Get model output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # Post-process output (assuming object detection)
    # Draw boxes or use other post-processing as needed
    for detection in output_data:
        ymin, xmin, ymax, xmax = detection
        start_point = (int(xmin * frame.shape[1]), int(ymin * frame.shape[0]))
        end_point = (int(xmax * frame.shape[1]), int(ymax * frame.shape[0]))
        cv2.rectangle(frame, start_point, end_point, (0, 255, 0), 2)
    
    # Display the result
    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
'/home/siddharth/Documents/ProjectMayura/testpicture'''


import torch
from PIL import Image


# Load the YOLOv5 model
model_path = '/home/siddharth/yolov5/runs/train/exp2/weights/best.pt'
model = torch.hub.load('yolov5', 'custom', path=model_path, source='local')

# Load an image
img_path = '/home/siddharth/Documents/ProjectMayura/Images/pic3.jpg'
img = Image.open(img_path)

# Perform inference
results = model(img)

# Print results
results.print()

# Show results
results.show()








