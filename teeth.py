# tooth extractor

import cv2
import matplotlib.pyplot as plt
import numpy as np


# show image
from skimage import io, color
from sklearn.cluster import KMeans


def plot(image):
    cv2.imshow("", image)
    cv2.waitKey(0)

def k_means_clustering(data_pts, cluster_size):
    clt = KMeans(n_clusters=cluster_size)
    clt.fit(data_pts)
    return clt

def GetRGBColor(roi_image):
    roi_image = cv2.cvtColor(roi_image, cv2.COLOR_BGR2Lab)
    locs = np.where((roi_image[:, :, 0] != 0))
    data_pts = roi_image[locs[0], locs[1], :]
    # Now, compute KMeans clustering and extract dominant colors (or centroids or cluster centers)
    clt = k_means_clustering(data_pts, cluster_size=1)  # major 3 cluster/color from ROI
    dom_col = clt.cluster_centers_.astype("uint8")  # LAB dominant colors
    new_col = color.label2rgb(dom_col)
    print(dom_col)
    #a,b -127


def GetTeeth(labelNumber):
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[:, :, 0] = output
    mask[:, :, 1] = output
    mask[:, :, 2] = output
    for i in range(0, len(mask)):
        if i != labelNumber:
            mask[mask == i] = 0
    mask[mask == labelNumber] = 255

    # Mask input image with binary mask
    result = cv2.bitwise_and(image, mask)
    #gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # edged = cv2.Canny(gray, 30, 200)
    # contours, hierarchy = cv2.findContours(edged,
    #                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    GetRGBColor(result)
    plot(result)

# load image
image = cv2.imread("C://Users//nithinap\PycharmProjects\DentScan//clean.png")
labelNumber = 1
# covert to gray
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# hist equalizaton
img_hist = cv2.equalizeHist(img_gray)
kernel = np.ones((2, 5), dtype=np.uint8)
img_erosion = cv2.erode(img_hist, kernel, iterations=1)
img_dilation = cv2.dilate(img_erosion, kernel, iterations=1)
# canny edge detection
edges = cv2.Canny(img_dilation, 60, 70)
(thresh, im_bw) = cv2.threshold(img_dilation, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# num_labels, labels_im = cv2.connectedComponents(im_bw)

nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(im_bw, connectivity=4)


mask_2 = np.zeros(image.shape, dtype=np.uint8)
mask_2[:, :, 0] = output
mask_2[:, :, 1] = output
mask_2[:, :, 2] = output
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
for i in range(1, len(centroids)):
    mask_2[mask_2 == i] = 255
    a = int(centroids[i][0])
    b = int(centroids[i][1])
    cv2.putText(mask_2, str(i), (a, b), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

plot(mask_2)
GetTeeth(labelNumber)
