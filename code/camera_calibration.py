import cv2
import numpy as np

CORNER_COUNT = 6, 8

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CORNER_COUNT[0]*CORNER_COUNT[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CORNER_COUNT[0],0:CORNER_COUNT[1]].T.reshape(-1,2)


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = []

for i in range(1, 21):
    img = cv2.imread(f"../AssignmentIII/AssignmentIII/PartA/images/img{i}.jpg", cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img.shape[1]//4, img.shape[0]//4))
    images.append(img)
    print(img.shape)
    # exit()
    # print(img)
    # cv2.imshow("corners", img)
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()

for img in images:
    # print(img.shape[::-1])
    # img = cv2.resize(img, (img.shape[1]//10, img.shape[0]//10))
    # print(img.shape)
    # cv2.imshow("corners", img)
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()
    ret, corners = cv2.findChessboardCorners(img, CORNER_COUNT, None)
    # print(ret)
    # print(corners)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        blank = np.zeros(img.shape, dtype=np.uint8)
        cv2.drawChessboardCorners(img, CORNER_COUNT, corners2, ret)
        cv2.drawChessboardCorners(blank, CORNER_COUNT, corners2, ret)

        # cv2.imshow("corners", img)
        # cv2.waitKey(0)

        # cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)

newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,img.shape[::-1],1,img.shape[::-1])
# ret: RMS reprojection error
# mtx: camera matrix
# dist: distortion coefficients [k1, k2, p1, p2, k3]
# k: radial distortion values
# p: tangential distortion values
# rvecs: rotation vector
# tvecs: tranlation vector
img2 = cv2.imread(f"../AssignmentIII/AssignmentIII/PartA/images/img20.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.resize(img2, (img2.shape[1]//4, img2.shape[0]//4))
# img2 = cv2.resize(img2, img2.shape)
dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
# blk_dst = cv2.undistort(blank, mtx, dist, None, newcameramtx)
print(ret)
print(mtx)
print(dist)
print(rvecs)
print(tvecs)
print(newcameramtx)
print(roi)

cv2.imshow("img2", img2)
cv2.imshow("undistort", dst)
cv2.waitKey(0)

cv2.destroyAllWindows()

for img in images:
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    
    cv2.imshow("img2", img)
    cv2.imshow("undistort", dst)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
