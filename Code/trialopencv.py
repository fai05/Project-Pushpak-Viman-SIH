import cv2
import numpy

# Load the image from file
image = cv2.imread('/home/siddharth/Documents/ProjectMayura/testpicture')  # Replace with the path to your image

# Define coordinates for the rectangle (x, y, width, height)
x, y, width, height = 50, 50, 200, 100  # Example coordinates, adjust as needed

# Define rectangle color (BGR format) and thickness
rectangle_color = (0, 255, 0)  # Green color
thickness = 2  # Thickness of the rectangle lines

# Draw the rectangle on the image
cv2.rectangle(image, (x, y), (x + width, y + height), rectangle_color, thickness)

# Display the image with the rectangle
cv2.imshow('Image with Rectangle', image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the image with the rectangle
cv2.imwrite('image_with_rectangle.jpg', image)  # Save the image if needed
