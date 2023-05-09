import torch
import easyocr
import cv2
import numpy as np

def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    
    plate = [] 
    print(ocr_result)
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate

# Model
model = torch.hub.load('.', 'custom', path='best.pt', source='local')

# Image
img_path = "1.jpg"
cv_img = cv2.imread(img_path)
height, width, _ = cv_img.shape
new_width = int(width * 0.5)
new_height = int(height * 0.5)
# Inference
results = model(cv_img)

# OCR
reader = easyocr.Reader(['en'])  # set OCR language

crops = results.crop(save=False)
lp_text = ""
for crop in crops:
    x1,y1,x2,y2 = crop["box"]
    x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
    if crop["conf"]>.5:
        license_plate = cv_img[y1:y2,x1:x2]
        
        ocr_result = reader.readtext(license_plate)
        for result in ocr_result:
            lp_text+=" " + result[1]
            # Draw bounding box and text on image
        cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(cv_img, ' '.join(lp_text), (x1-20, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        print(lp_text)

image = cv2.resize(cv_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
# Display annotated image
cv2.imshow("Annotated image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
