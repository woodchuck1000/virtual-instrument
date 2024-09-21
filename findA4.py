import cv2
import numpy as np

stable_frames = 0
display_quad = None
last_center = None

def order_point(pts):#确定轮廓点位
    rect=np.zeros((4,2),dtype='float32')
    s=pts.sum(axis=1)
    rect[0]=pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff=np.diff(pts,axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def four_point_transform(image,pts):
    rect=order_point(pts)
    (tl,tr,br,bl)=rect

    widthA=np.sqrt(((br[0]-bl[0])**2)+((br[1]-bl[1])**2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth=max(int(widthA),int(widthB))
    heighthA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heighthB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeighth=max(int(heighthA),int(heighthB))

    dst=np.array([
        [0,0],
        [maxWidth-1,0],
        [maxWidth-1,maxHeighth-1],
        [0,maxHeighth-1]],dtype='float32')

    M=cv2.getPerspectiveTransform(rect,dst)
    warped=cv2.warpPerspective(image,M,(maxWidth,maxHeighth))
    return warped

def find_quadrilaterals(frame):
    global last_center, stable_frames, display_quad
    frame1 =frame
    frame2 = np.zeros((900, 1280, 3), dtype=np.uint8)
    updated_quad = 0  # 初始化更新标志为0，表示未更新新的轮廓
    # 转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 应用高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 边缘检测
    edged = cv2.Canny(blurred, 50, 120)

    # 查找轮廓
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = 0
    current_quad = None
    frame_area = frame.shape[0] * frame.shape[1]  # 计算画面总面积
    for contour in contours:
        # 多边形逼近
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

        # 如果逼近的顶点数为4，则认为找到了四边形
        if len(approx) == 4:
            # 计算四边形的面积
            area = cv2.contourArea(approx)
            if area > largest_area:
                largest_area = area
                current_quad = approx
    # 检测到的最大四边形轮廓的面积是否至少达到了整个画面面积的1/10
    if current_quad is not None and largest_area >= frame_area / 10:
        M = cv2.moments(current_quad)
        if M["m00"] != 0:
            current_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            if last_center is not None:
                delta_x = abs(current_center[0] - last_center[0])
                delta_y = abs(current_center[1] - last_center[1])
                if delta_x >= 10 or delta_y >= 10:  # 显著变化
                    stable_frames = 0
                    display_quad = current_quad
                    updated_quad = 1  # 更新标志置为1，表示更新了新的轮廓
                else:
                    stable_frames += 1
            else:
                display_quad = current_quad  # 第一次检测到四边形时直接更新
                updated_quad = 1  # 更新标志置为1，表示更新了新的轮廓
            last_center = current_center
    else:
        # 如果没有检测到满足条件的轮廓，维持stable_frames计数
        stable_frames = min(stable_frames + 1, 5)  # 防止超过5
    # 如果连续5帧内四边形位置没有显著变化或未检测到新轮廓，则继续显示最后一次检测到的轮廓
    if display_quad is not None:
        frame1=cv2.drawContours(frame.copy(),[display_quad],-1,(0,0,255),4)
        frame2 = four_point_transform(frame.copy(), display_quad.reshape(4, 2))
    return frame1,frame2, updated_quad

