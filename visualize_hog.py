from HoG import extract_features
from matplotlib import pyplot as plt
import cv2
# img = cv2.imread('./test.jpg', 1)
img = cv2.imread('./test/g.jpg', 1)
features, hog_image = extract_features(img=img, pixel_per_cell=(8,8), cells_per_block=(3,3), visualize=True)
edges = cv2.Canny(img, 7,7)
fig, axarr = plt.subplots(1, 3, figsize=(16, 4))
axarr[0].axis('off')
axarr[1].axis('off')

axarr[0].imshow(img[:,:,::-1])
axarr[0].set_title("Image")
axarr[1].imshow(255-hog_image, cmap="gray")
axarr[1].set_title("Our HOG")
axarr[2].imshow(edges, cmap="gray")
axarr[2].set_title("Normal edge detector")
plt.show()
