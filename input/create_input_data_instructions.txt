## Creating Semantic Label file 

Input semantic label is created by putting each pixel_value for objects from [objectInfo150](https://github.com/nitish11/SPADE/blob/master/datasets/ade20k_stuff/objectInfo150.txt). Sample code is given below.

```bash

# Let's load a simple image
image = cv2.imread("test_image.png")
image = cv2.resize(image, (256,256))

# Grayscale 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
  
# Find Canny edges 
edged = cv2.Canny(gray, 30, 200) 

#Finding contours
_, contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
print("Number of Contours found = " + str(len(contours))) 
  
# Draw all contours 
#Drawing semantic label
base_image = np.ones(np.shape(image))

#Color code for chair
color_code = (20, 20, 20)

#Drawing contours
cv2.drawContours(base_image, contours, -1, color_code, 8) 
cv2.drawContours(base_image, contours, -1, color_code, -1) 
print(np.shape(base_image))

cv2.imwrite('test_val_00000004.png', base_image) 
display(filename = 'test_val_00000004.png')

```