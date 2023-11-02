import cv2
import numpy as np
import matplotlib.pyplot as plt
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

for i in range(1,5):
    img_path = "../AssignmentIII/AssignmentIII/PartB/images"
    img = cv2.imread(f"{img_path}/er7-{i}.jpg")
    blank = np.zeros(img.shape, dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gauss_img = cv2.GaussianBlur(gray, (3, 3), 0, 0)

    rt, bin_img = cv2.threshold(gauss_img, BIN_THRESH, 255, cv2.THRESH_BINARY)

    # open_img = cv2.erode(bin_img, KERNEL, iterations=1)
    # dilate_img = cv2.dilate(open_img, KERNEL, iterations=1)

    # find contours
    contours, hiearchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    x_lab=""

    object_cnt=1
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # perimeter = cv2.arcLength(cnt, True)

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
        cv2.line(bin_img, point1, point2, (0, 255, 0), 3)
        x_lab+=f"Centroid {object_cnt} = ({cx},{cy})\nPrinciple angle {object_cnt}= {(angle*180/pi):.6f} degree\n"
        object_cnt+=1

    plt.xlabel(x_lab)
    plt.imshow(img)
    plt.savefig(f"{img_path}/result_{i}.png",dpi=200,bbox_inches="tight")
    plt.show()
    plt.clf()
    #cv2.imshow("og", img)
# cv2.imshow("ell", blank)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
