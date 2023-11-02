import cv2, numpy as np, sys
from math import cos, sin, pi, atan2

# print(sys.argv[1])
KERNEL = np.array(
    [
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
    ], dtype=np.uint8
)

AREA_THRESH = 200
BIN_THRESH = 200
LINE_LENGTH = 200

img_path = sys.argv[1]
img_path = "../AssignmentIII/AssignmentIII/PartB/images/er7-3.jpg"
img = cv2.imread(img_path)
blank = np.zeros(img.shape, dtype=np.uint8)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gauss_img = cv2.GaussianBlur(gray, (3, 3), 0, 0)

rt, bin_img = cv2.threshold(gauss_img, BIN_THRESH, 255, cv2.THRESH_BINARY)

# open_img = cv2.erode(bin_img, KERNEL, iterations=1)
# dilate_img = cv2.dilate(open_img, KERNEL, iterations=1)

# find contours
contours, hiearchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if area < AREA_THRESH:
        continue
    M = cv2.moments(cnt)

    cx = int(M["m10"]/M["m00"])
    cy = int(M["m01"]/M["m00"])
    # centroid
    cv2.circle(img, (cx, cy), 5, (255,0, 0), -1)

    angle = atan2(2*M["mu11"], M["mu20"]-M["mu02"])/2
    point1 = int(cx + LINE_LENGTH*cos(angle)), int(cy + LINE_LENGTH*sin(angle))
    point2 = int(cx - LINE_LENGTH*cos(angle)), int(cy - LINE_LENGTH*sin(angle))
    cv2.line(img, point1, point2, (0, 255, 0), 3)

cv2.imshow("dilate", bin_img)
cv2.imshow("og", img)
# cv2.imshow("ell", blank)
cv2.waitKey(0)
cv2.destroyAllWindows()
