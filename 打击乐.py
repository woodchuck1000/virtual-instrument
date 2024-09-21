import cv2
import numpy as np
import os
import sounddevice as sd
import pygame
from PIL import Image
import findA4
import CNN

def get_images(image_folder_path):
    image_paths = os.listdir(image_folder_path)
    images=[]
    for image_path in image_paths:
        full_image_path = os.path.join(image_folder_path, image_path)
        images.append(Image.open(full_image_path).resize((60,80 )).convert('RGBA'))
    return images

def get_audios(audio_folder_path):
    audio_paths = os.listdir(audio_folder_path)
    audios = {}
    pygame.mixer.init()
    for audio_path in audio_paths:
        full_audio_path = os.path.join(audio_folder_path, audio_path)
        sound = pygame.mixer.Sound(full_audio_path)
        audios[audio_path] = sound  # 使用文件名作为键
    return audios

def resize(image,width=None,height=None,inter=cv2.INTER_AREA):
    (h,w)=image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r=height/float(h)
        dim=(width,int(h*r))
    resized=cv2.resize(image,dim,interpolation=inter)
    return  resized

def preprocess_image(img):
    """
    对输入图像进行预处理，包含翻转、灰度化、边缘检测和形态学操作。
    :param img: 输入图像
    :return: 处理后的图像
    """
    img = cv2.flip(img, 0)  # 垂直翻转图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图像
    edges = cv2.Canny(gray, 100, 200)  # Canny边缘检测
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=4)  # 形态学闭运算
    return closed

def find_contours(img):
    """
    查找图像中的轮廓，并计算每个轮廓的边界框。
    :param img: 处理后的图像
    :return: 轮廓列表和对应的边界框列表
    """
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_images = []
    boxes = []
    yinjies=[]
    height, width = img.shape[:2]

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # 过滤掉不符合要求的轮廓
        if width * 0.05 < x < width * 0.95 and height * 0.05 < y < height * 0.95 and height * 0.04 < h < height * 0.2:
            side_length = max(w, h)
            a = side_length * 1.2
            x1 = max(0, int(x + w / 2 - a / 2))
            y1 = max(0, int(y + h / 2 - a / 2))
            x2 = min(img.shape[1] - 1, int(x + w / 2 + a / 2))
            y2 = min(img.shape[0] - 1, int(y + h / 2 + a / 2))
            digit_images.append(img[y1:y2, x1:x2])
            # 计算上下扩展区域的边距
            top_margin = y1 - (y2-y1)// 3
            bottom_margin = y2 + (y2-y1) // 3

            top_margin = max(0, top_margin)
            bottom_margin = min(img.shape[0], bottom_margin)

            # 提取上下区域
            top_region = img[top_margin:y1, x1:x2]
            white_pixels_top = np.sum(top_region == 255)
            has_white_top = white_pixels_top >= 20

            bottom_region = img[y2:bottom_margin, x1:x2]
            white_pixels_bottom = np.sum(bottom_region == 255)
            has_white_bottom = white_pixels_bottom >= 20

            # 根据是否有足够的白点调整边界框
            if has_white_top and not has_white_bottom:
                y1 = top_margin
                yinjies.append(20)
            elif has_white_bottom and not has_white_top:
                y2 = bottom_margin
                yinjies.append(0)
            else:
                yinjies.append(10)
            boxes.append((x1, y1, x2, y2))
    return digit_images, boxes ,yinjies

def draw_annotations(img, boxes, test_labels,yinjies):
    """
    在图像上绘制标注信息，包括矩形框、文本和圆点。
    :param img: 原始图像
    :param boxes: 数字框坐标列表
    :param test_labels: 识别出的数字标签列表
    :param closed: 处理后的图像，用于检查白色像素
    :return: 带标注的图像
    """
    for box, test_label ,yinjie in zip(boxes, test_labels,yinjies):
        x1, y1, x2, y2 = box
        dot_radius = 5
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        if yinjie==20:
            cv2.putText(img, str(test_label), (x1 + 10, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,cv2.LINE_AA)
            dot_center = (x1 + 20, y1 - 40)
            cv2.circle(img, dot_center, dot_radius, (0, 0, 255), -1)
        elif yinjie==0:
            cv2.putText(img, str(test_label), (x1 + 10, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,cv2.LINE_AA)
            dot_center = (x1 + 20, y1 - 10)
            cv2.circle(img, dot_center, dot_radius, (0, 0, 255), -1)
        else:
            cv2.putText(img, str(test_label), (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,cv2.LINE_AA)
    return img

def recognize(img, updated_quad):
    """
    主函数，处理图像并识别数字。
    :param img: 输入图像
    :param updated_quad: 是否更新图像的标志
    :return: 带标注的图像和是否开始播放的标志
    """
    global frame_count, frame_count2, prev_boxes, prev_labels,prev_yinjies, num
    startplay = 0
    if updated_quad == 1:
        frame_count2 = 0
    frame_count += 1
    if frame_count % 3 == 0 and frame_count2 < 10:
        closed = preprocess_image(img)
        digit_images, boxes, yinjies = find_contours(closed)
        test_labels = CNN.recognize_digits(digit_images)
        prev_labels = test_labels
        prev_boxes = boxes
        prev_yinjies=yinjies
        frame_count2 += 1
    else:
        test_labels = prev_labels
        boxes = prev_boxes
        yinjies=prev_yinjies

    if frame_count2 == 10:
        startplay = 1

    img = cv2.flip(img, 0)
    img = draw_annotations(img, boxes, test_labels, yinjies)

    return img, startplay

def play(img):
    global prev_boxes, prev_labels,prev_yinjies, num, hit
    check=0
    for box, label ,yinjie in zip(prev_boxes, prev_labels,prev_yinjies):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        region_top_left_x = x1 + width * 0.1
        region_top_left_y = y1 + height * 0.1
        region_bottom_right_x = x2 - width * 0.1
        region_bottom_right_y = y2 - height * 0.1

        region = img[int(region_top_left_y):int(region_bottom_right_y),
                 int(region_top_left_x):int(region_bottom_right_x)]

        # 计算低亮度区域的面积占整个区域的百分比
        brightness_threshold = 110
        low_brightness_pixels = np.sum(region < brightness_threshold)
        total_pixels = region.size
        low_brightness_area_percentage = (low_brightness_pixels / total_pixels) * 100


        if low_brightness_area_percentage>20 :
            num = label+yinjie
            hit=1
            check+=1
            img = yueqi(img, x1, y2,1)
        else:
            img = yueqi(img, x1, y2, 0)
    if not check:
        hit=0
    return img

def yueqi(img, x1, y2,which):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(img.astype(np.uint8))
    img = img.convert('RGBA')
    img.alpha_composite(images[which], dest=(x1-15, y2))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGR)
    return img

def main_view():
    global num
    while True:
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        if not ret:
            break
        frame1, frame2,updated_quad = findA4.find_quadrilaterals(img)
        frame2 = resize(frame2, width=1280, height=720)
        frame2,startplay = recognize(frame2,updated_quad)
        if startplay:
            frame2=play(frame2)
        small_frame1 = cv2.resize(frame1, (0, 0), fx=0.2, fy=0.2)
        small_height, small_width = small_frame1.shape[:2]
        frame2[0:small_height, 0:small_width] = small_frame1
        cv2.imshow("Camera2", frame2)

        if cv2.waitKey(1) in [ord('q'), 27]:
            break

def audio_callback(indata, outdata, frames, time):
    global num, channel, now, hit
    if hit==1:
        audio_name = f"{num}.wav"  # 假设音频文件名是根据num命名的
        if now==0:  # 判断num是否发生变化
            if channel is not None and channel.get_busy():
                channel.fadeout(150)  # 停止当前正在播放的音频
            now = 1  # 更新now的值
            channel = audios[audio_name].play()  # 直接从预加载的字典中获取并播放
    else:
        now = 0

cap = cv2.VideoCapture(0)
cap.set(3,1280)
ret, img = cap.read()
frame_count = 0  # 计数器
frame_count2=0
prev_boxes = []  # 上一帧的数字位置信息
prev_labels = []  # 上一帧的数字标签
prev_yinjies=[]
num=now=0
hit=0
channel = None
audios=get_audios('bianzhong_audio')
images=get_images('bianzhong_image')
stream = sd.InputStream(callback=audio_callback)
with stream:
    main_view()
pygame.mixer.quit()
cap.release()
cv2.destroyAllWindows()