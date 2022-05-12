import cv2  
import numpy as np
  
vid = cv2.VideoCapture(0)
i = 0
chessboard_flags = (cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
corner_flags = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 40, 0.001)
board_dims = (7,7)

objp = np.zeros((7*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

objpoints = [] # 3d point in real world space
corner_list = []    #2d points, ~imgpoints

box_size = 6
axisBoxes = np.float32([
    [0,0,0],
    [0,box_size,0],
    [box_size,box_size,0],
    [box_size,0,0],
    [0,0,-box_size],
    [0,box_size,-box_size],
    [box_size,box_size,-box_size],
    [box_size,0,-box_size]
    ])


def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img

while(True):      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  
    if ret:
        found, corners = cv2.findChessboardCorners(gray, board_dims, chessboard_flags)
        if found:
            objpoints.append(objp)
            corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), corner_flags)
            corner_list.append(corners)
            cv2.drawChessboardCorners(gray, board_dims, corners, found)

            ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, corner_list, gray.shape[::-1], None, None)
            retval, rvec, tvec = cv2.solvePnP(objp, corners, mtx, dist)

            imgpts, jac = cv2.projectPoints(axisBoxes, rvec, tvec, mtx, dist)

            frame = draw(frame, corners, imgpts)
            
        print(i, found)
        cv2.imshow('frame', frame)

    i+=1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
vid.release()
cv2.destroyAllWindows()