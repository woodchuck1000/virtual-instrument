import cv2
import numpy
import os
import sounddevice
import pygame
from PIL import Image
from findA4 import find_quadrilaterals
from CNN import recognize_digits


class DajiYue:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.ret, self.img = self.cap.read()
        self.frame_count = 0
        self.frame_count2 = 0
        self.prev_boxes = []
        self.prev_labels = []
        self.prev_yinjies = []
        self.num = 0
        self.now = 0
        self.hit = 0
        self.channel = None
        self.audios = self.get_audios('bianzhong_audio')
        self.images = self.get_images('bianzhong_image')
        self.stream = sounddevice.InputStream(callback=self.audio_callback)

    def get_images(self, image_folder_path):
        image_paths = os.listdir(image_folder_path)
        images = []
        for image_path in image_paths:
            full_image_path = os.path.join(image_folder_path, image_path)
            images.append(Image.open(full_image_path).resize((60, 80)).convert('RGBA'))
        return images

    def get_audios(self, audio_folder_path):
        audio_paths = os.listdir(audio_folder_path)
        audios = {}
        pygame.mixer.init()
        for audio_path in audio_paths:
            full_audio_path = os.path.join(audio_folder_path, audio_path)
            sound = pygame.mixer.Sound(full_audio_path)
            audios[audio_path] = sound
        return audios

    def resize(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = height / float(h)
            dim = (width, int(h * r))
        resized = cv2.resize(image, dim, interpolation=inter)
        return resized

    def preprocess_image(self, img):
        img = cv2.flip(img, 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        kernel = numpy.ones((3, 3), numpy.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=4)
        return closed

    def find_contours(self, img):
        contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digit_images = []
        boxes = []
        yinjies = []
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
                top_margin = y1 - (y2 - y1) // 3
                bottom_margin = y2 + (y2 - y1) // 3

                top_margin = max(0, top_margin)
                bottom_margin = min(img.shape[0], bottom_margin)

                # 提取上下区域
                top_region = img[top_margin:y1, x1:x2]
                white_pixels_top = numpy.sum(top_region == 255)
                has_white_top = white_pixels_top >= 20

                bottom_region = img[y2:bottom_margin, x1:x2]
                white_pixels_bottom = numpy.sum(bottom_region == 255)
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
        return digit_images, boxes, yinjies

    def draw_annotations(self, img, boxes, test_labels, yinjies):
        for box, test_label, yinjie in zip(boxes, test_labels, yinjies):
            x1, y1, x2, y2 = box
            dot_radius = 5
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            if yinjie == 20:
                cv2.putText(img, str(test_label), (x1 + 10, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                dot_center = (x1 + 20, y1 - 40)
                cv2.circle(img, dot_center, dot_radius, (0, 0, 255), -1)
            elif yinjie == 0:
                cv2.putText(img, str(test_label), (x1 + 10, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                dot_center = (x1 + 20, y1 - 10)
                cv2.circle(img, dot_center, dot_radius, (0, 0, 255), -1)
            else:
                cv2.putText(img, str(test_label), (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return img

    def recognize(self, img, updated_quad):
        self.frame_count += 1
        if updated_quad == 1:
            self.frame_count2 = 0
        if self.frame_count % 3 == 0 and self.frame_count2 < 10:
            closed = self.preprocess_image(img)
            digit_images, boxes, yinjies = self.find_contours(closed)
            test_labels = recognize_digits(digit_images)
            self.prev_labels = test_labels
            self.prev_boxes = boxes
            self.prev_yinjies = yinjies
            self.frame_count2 += 1
        else:
            test_labels = self.prev_labels
            boxes = self.prev_boxes
            yinjies = self.prev_yinjies

        if self.frame_count2 == 10:
            startplay = 1
        else:
            startplay = 0

        img = cv2.flip(img, 0)
        img = self.draw_annotations(img, boxes, test_labels, yinjies)
        return img, startplay

    def play(self, img):
        check = 0
        for box, label, yinjie in zip(self.prev_boxes, self.prev_labels, self.prev_yinjies):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            region_top_left_x = x1 + width * 0.1
            region_top_left_y = y1 + height * 0.1
            region_bottom_right_x = x2 - width * 0.1
            region_bottom_right_y = y2 - height * 0.1

            region = img[int(region_top_left_y):int(region_bottom_right_y),
                         int(region_top_left_x):int(region_bottom_right_x)]

            brightness_threshold = 120
            low_brightness_pixels = numpy.sum(region < brightness_threshold)
            total_pixels = region.size
            low_brightness_area_percentage = (low_brightness_pixels / total_pixels) * 100

            if low_brightness_area_percentage > 30:
                self.num = label + yinjie
                self.hit = 1
                check += 1
                img = self.yueqi(img, x1, y2, 1)
            else:
                img = self.yueqi(img, x1, y2, 0)
        if not check:
            self.hit = 0
        return img

    def yueqi(self, img, x1, y2, which):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img.astype(numpy.uint8))
        img = img.convert('RGBA')
        img.alpha_composite(self.images[which], dest=(x1 - 15, y2))
        img = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGBA2BGR)
        return img

    def start_play(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        with self.stream:
            while True:
                ret, img = self.cap.read()
                if not ret:
                    break

                img = cv2.flip(img, 1)  # 水平翻转图像

                # 使用findA4模块的find_quadrilaterals函数来检测四边形并获取updated_quad
                frame1, frame2, updated_quad = find_quadrilaterals(img)
                # 调整frame2的大小
                frame2 = self.resize(frame2, width=1280, height=720)

                # 调用recognize方法，传递updated_quad参数
                frame2, startplay = self.recognize(frame2, updated_quad)

                if startplay:
                    # 如果startplay为True，则调用play方法
                    frame2 = self.play(frame2)

                # 将frame1的缩略图显示在frame2的左上角
                small_frame1 = cv2.resize(frame1, (0, 0), fx=0.2, fy=0.2)
                small_height, small_width = small_frame1.shape[:2]
                frame2[0:small_height, 0:small_width] = small_frame1
                # 显示处理后的frame2
                cv2.imshow("Camera2", frame2)

                # 按'q'或ESC键退出
                if cv2.waitKey(1) in [ord('q'), 27]:
                    break

        # 释放摄像头资源并关闭所有窗口
        pygame.mixer.quit()
        self.cap.release()
        cv2.destroyAllWindows()

    def audio_callback(self, indata, outdata, frames, time):
        # 如果检测到打击动作
        if self.hit == 1:
            audio_name = f"{self.num}.wav"
            if self.now == 0:
                if self.channel is not None and self.channel.get_busy():
                    self.channel.fadeout(150)  # 如果当前有音频播放，先淡出
                self.now = 1
                # 从音频字典中获取音频并播放
                audio_source = self.audios.get(audio_name)  # 安全获取音频源
                if audio_source:  # 检查音频源是否存在
                    self.channel = audio_source.play()
        else:
            self.now = 0


if __name__ == "__main__":
    app = DajiYue()
    app.start_play()