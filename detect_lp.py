import torch
import easyocr
import cv2
import numpy as np
import argparse
import os


# Create an argument parser
parser = argparse.ArgumentParser(description='Process some images.')

# Add a --source flag with two options: 0 or a path to an image
parser.add_argument('--source', type=str, default='0',
                    help='the source of the image to process')

# Parse the arguments
args = parser.parse_args()

def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    
    plate = [] 
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        
        if length*height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate

# Model
model = torch.hub.load('.', 'custom', path='best.pt', source='local')
# OCR
reader = easyocr.Reader(['en'], model_storage_directory=".", download_enabled=False)  # set OCR language

if args.source != '0':

    # Image
    img_path = args.source
    files = os.listdir(img_path)
    for f in files:
        ext = f.split(".")
        if ext[-1] not in ('jpg', 'jpeg'):
            continue

        cv_img = cv2.imread(os.path.join(img_path, f))
        # Inference
        results = model(cv_img)

        crops = results.crop(save=False)
        lp_text = ""
        for crop in crops:
            x1,y1,x2,y2 = crop["box"]
            x1,y1,x2,y2 = int(x1),int(y1)+3,int(x2),int(y2)
            if crop["conf"]>.5:
                license_plate = cv_img[y1:y2,x1:x2]
                
                ocr_result = reader.readtext(license_plate)
                for result in ocr_result:
                    lp_text+=" " + result[1]
                    # Draw bounding box and text on image
                cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cv_img, ' '.join(lp_text), (x1-40, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
        # Display annotated image
        # Write the processed image to file
        cv2.imwrite(f"./files/res/{f}", cv_img)     

else:
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        # Capture a frame from the webcam
        ret, cv_img = cap.read()
        if not ret:
            break

        # Inference
        results = model(cv_img)

        # OCR
        crops = results.crop(save=False)
        lp_text = ""
        for crop in crops:
            x1, y1, x2, y2 = crop["box"]
            x1, y1, x2, y2 = int(x1), int(y1)+40, int(x2), int(y2)-40
            if crop["conf"] > .5:
                license_plate = cv_img[y1:y2, x1:x2]

                ocr_result = reader.readtext(license_plate)
                for result in ocr_result:
                    lp_text += " " + result[1]
                # Draw bounding box and text on image
                cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cv_img, ' '.join(lp_text), (x1-20, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display annotated image
        cv2.imshow("Annotated image", cv_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Release the webcam and destroy windows
    cap.release()
cv2.destroyAllWindows()
