import glob
import os

import cv2

path = sorted(glob.glob("./output_imgs/*.png"), key=lambda x: int(x.split('\\')[-1].split('.')[0]))
print([i.split('\\')[-1] for i in path])

for filename in path:
    cv2.imshow("Frame", cv2.imread(filename, 1))
    cv2.waitKey(50)
