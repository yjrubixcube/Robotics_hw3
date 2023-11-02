import cv2
import numpy as np

CORNER_COUNT = 6, 9

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((CORNER_COUNT[0]*CORNER_COUNT[1],3), np.float32)
objp[:,:2] = np.mgrid[0:CORNER_COUNT[0],0:CORNER_COUNT[1]].T.reshape(-1,2)


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = []

for i in range(1):
    img = cv2.imread(f"../AssignmentIII/AssignmentIII/PartA/sf.jpg", cv2.IMREAD_GRAYSCALE)
    images.append(img)
    # print(img)
    # cv2.imshow("corners", img)
    # cv2.waitKey(0)

    # cv2.destroyAllWindows()

for img in images:
    print(img.shape)
    # img = cv2.resize(img, (img.shape[1]//10, img.shape[0]//10))
    print(img.shape)
    cv2.imshow("corners", img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
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

        cv2.imshow("corners", img)
        cv2.waitKey(0)

        cv2.destroyAllWindows()
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)

        dst = cv2.undistort(img, mtx, dist, None, mtx)
        blk_dst = cv2.undistort(blank, mtx, dist, None, mtx)

        print(ret)
        print(mtx)
        print(dist)
        print(rvecs)
        print(tvecs)

        cv2.imshow("dis", img)
        cv2.imshow("undis", dst)
        cv2.imshow("dots", blk_dst)
        cv2.waitKey(0)

        cv2.destroyAllWindows()