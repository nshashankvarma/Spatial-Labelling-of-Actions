import joblib
import numpy as np
import cv2
from skimage.transform import pyramid_gaussian
from skimage import color
from HoG import extract_features
import imutils
from imutils.object_detection import non_max_suppression
import sliding_window as sd

detections = []
previous_box = []
current_box = []


# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
model = joblib.load('./model.dat')
size = (64, 128)
step_size = (9, 9)
downscale = 1.25
cap = cv2.VideoCapture('./test/WalkAndStop.mp4')


def processFrame(image, i):
    """
    Function to process the frame and draw the bounding box.

    NOTE : We skip every 2 frames. This is governed by the initial condition
    
    Processes : 
        1. Scaling up and down the images and extracting features using HOG
        2. Using the model to predict if the given frame contains HUMAN based on the features
        3. Applying a Non Max Suppression (using threshold = 0.15) to get the best bounding box
        4. Drawing the bounding box
        5. Labelling if there is action or not.
    """

    if (i % 3 == 0) :
        scale = 0
        for im_scaled in pyramid_gaussian(image, downscale=downscale):
            # The list contains detections at the current scale
            if im_scaled.shape[0] < size[1] or im_scaled.shape[1] < size[0]:
                break
            for (x, y, window) in sd.sliding_window(im_scaled, size, step_size):
                if window.shape[0] != size[1] or window.shape[1] != size[0]:
                    continue
                # window = color.rgb2gray(window)
                fd = extract_features(window, pixel_per_cell=(8, 8), cells_per_block=(3, 3))

                # fd=hog(window, orientations=9,pixels_per_cell=(8,8),visualize=False,cells_per_block=(3,3))
                fd = fd.reshape(1, -1)
                pred = model.predict(fd)
                if pred == 1:
                    if model.decision_function(fd) > 0.5:
                        detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), model.decision_function(fd),
                                        int(size[0] * (downscale**scale)),
                                        int(size[1] * (downscale**scale))))
            scale += 1

        clone = image.copy()

        rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])
        sc = [score[0] for (x, y, score, w, h) in detections]
        print("sc: ", sc)
        sc = np.array(sc)


        # if there is a probability of human existance
        if len(sc)>0:
            # get the bounding boxes
            pick = non_max_suppression(rects, probs=sc/np.max(sc), overlapThresh=0.15)
            
            current_box.clear()
            current_box.extend(pick[0])

            # get the difference in bounding box coordinates to find if action is taking place
            if len(previous_box) != 0 and abs(np.sum(np.array(previous_box)-np.array(current_box))):
                for(x1, y1, x2, y2) in pick:
                    cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(clone, 'Action', (x1-2, y1-2), 1, 0.75, (121, 12, 34), 1)
            else:
                for(x1, y1, x2, y2) in pick:
                    cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(clone, 'No Action', (x1-2, y1-2), 1, 0.75, (121, 12, 34), 1)

            previous_box.clear()
            previous_box.extend(current_box)
            
        return clone

    # else:
    #     clone = image.copy()
    #     x1, y1, x2, y2 = tuple(current_box)
    #     if len(previous_box) != 0 and abs(np.sum(np.array(previous_box)-np.array(current_box))):
    #         cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #         cv2.putText(clone, 'Action', (x1-2, y1-2), 1, 0.75, (121, 12, 34), 1)
    #     else:
    #         cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #         cv2.putText(clone, 'No Action', (x1-2, y1-2), 1, 0.75, (121, 12, 34), 1)
    
    


# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Error opening video stream or file")
i = -1
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    detections=[]
    ret, frame = cap.read()
    i += 1
    if ret == True:        
        image = cv2.resize(frame, (400, 256))
        # Display the resulting frame   
        # if i%3==0:    
        frame = processFrame(image, i)        
        # else:
            # frame = image
        cv2.imwrite("./output_images2/" + str(i) + ".png", frame)
        # cv2.imshow('Frame', frame)
        # cv2.waitKey(500)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break  
          
        # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()
# Closes all the frames
cv2.destroyAllWindows()
