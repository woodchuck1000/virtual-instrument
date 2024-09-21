import cv2
import os
import mediapipe
import sounddevice
import numpy
import pygame
from PIL import Image, ImageDraw, ImageFont


class ChuizouYue:

    def __init__(self, instrument_name):
        self.instrument_name = instrument_name
        self.hands = mediapipe.solutions.hands.Hands(static_image_mode=False,
                                                     max_num_hands=2,
                                                     model_complexity=1,
                                                     min_detection_confidence=0.5,
                                                     min_tracking_confidence=0.5)
        self.numbers = self.dupu()
        self.mpDraw = mediapipe.solutions.drawing_utils
        self._init_globals()
        self._init_audio_and_images(instrument_name)
        self.stream = sounddevice.InputStream(callback=self.audio_callback)

    def _init_globals(self):
        self.number = 0
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        self.ret, self.img = self.cap.read()
        self.img = cv2.flip(self.img, 1)
        self.h, self.w, _ = self.img.shape
        self.num = -1
        self.now = -1
        self.check = 0
        self.window_closed = False
        self.RATE = 44100  # 采样率为44100Hz
        self.CHANNELS = 1  # 单声道
        self.channel = None
        self.volume_data = [0] * 100
        self.pitch_data = [0] * 100

    def _init_audio_and_images(self, instrument_name):
        self.names = {'taodi': 0, 'hulusi': 1, 'xun': 2}
        self.yueqi = self.names[instrument_name]
        prefix_type = instrument_name
        self.inf = self.PatternHandler(prefix_type, self)
        self.audios = self.get_audios(f'{prefix_type}_audio')
        self.images, self.yinfu_black, self.yinfu_red = self.get_images(f'{prefix_type}_image')

    def dupu(self):
        # 打开并读取文件内容
        with open('results.txt', 'r') as file:
            content = file.read()

        # 使用 '\' 分隔符将内容分割成列表
        number_strings = content.split('\\')

        # 将字符串列表转换为整数列表
        numbers = [int(num) for num in number_strings if num.isdigit()]
        return numbers

    def get_audios(self, audio_folder_path):
        audio_paths = os.listdir(audio_folder_path)
        audios = {}
        pygame.mixer.init()
        for audio_path in audio_paths:
            full_audio_path = os.path.join(audio_folder_path, audio_path)
            sound = pygame.mixer.Sound(full_audio_path)
            audios[audio_path] = sound  # 使用文件名作为键
        return audios

    def get_images(self, image_folder_path):
        image_paths = os.listdir(image_folder_path)
        images = {}  # 使用字典存储图片，键是编号，值是图片对象
        for image_path in image_paths:
            full_image_path = os.path.join(image_folder_path, image_path)
            if os.path.isfile(full_image_path) and image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img = Image.open(full_image_path).resize((500, 320)).convert('RGBA')
                    # 提取文件名中的数字作为索引
                    filename, _ = os.path.splitext(image_path)
                    try:
                        index = int(filename)  # 尝试将文件名转换为整数
                        images[index] = img
                    except ValueError:
                        print(f"File name {filename} is not a valid number.")
                except Exception as e:
                    print(f"Unable to open image {full_image_path}: {e}")
        yinfu_black = [None] * 28
        yinfu_red = [None] * 28
        for i in list(range(1, 8)) + list(range(11, 18)) + list(range(21, 28)):
            # 读取图片
            image_path = os.path.join('yinfu_black', f'{i}.png')
            img = Image.open(image_path)
            # 存储图片到 yinfu_black 数组中
            yinfu_black[i] = img
            image_path = os.path.join('yinfu_red', f'{i}.png')
            img = Image.open(image_path)
            yinfu_red[i] = img
            continue

        return images, yinfu_black, yinfu_red

    class Fingertip:
        def __init__(self, id, num, result, w, h):
            self.type = result.multi_handedness[id].classification[0].label
            self.cx = int(result.multi_hand_landmarks[id].landmark[num].x * w)
            self.cy = int(result.multi_hand_landmarks[id].landmark[num].y * h)
            self.cz = result.multi_hand_landmarks[id].landmark[num].z
            self.xy = [self.cx, self.cy]

    class PatternHandler:
        def __init__(self, prefix_type, instrument):
            self.instrument = instrument
            # 初始化模式数据和索引
            if prefix_type == 'taodi':
                self.indices = [[4, 8, 12, 16, 20], [4, 8, 12, 16, 20]]
                self.size = 11
            elif prefix_type == 'hulusi':
                self.indices = [[4, 8, 12, 16], [8, 12, 16]]
                self.size = 9
            elif prefix_type == 'xun':
                self.indices = [[4, 8, 12, 16, 20], [4, 8, 12, 16, 20]]
                self.size = 12
            else:
                raise ValueError(f"Unknown prefix_type: {prefix_type}")
            self.pattern = [[32] * 2 for _ in range(40)]
            self._set_pattern_and_indices(prefix_type)

        def _set_pattern_and_indices(self, prefix_type):
            if prefix_type == 'taodi':
                self.minpitch = 11  # 中音1
                self.pattern[11] = [31, 31]
                self.pattern[12] = [31, 15]
                self.pattern[13] = [31, 7]
                self.pattern[14] = [31, 3]
                self.pattern[15] = [31, 1]
                self.pattern[16] = [23, 1]
                self.pattern[17] = [19, 1]
                self.pattern[21] = [17, 1]
                self.pattern[22] = [16, 1]
                self.pattern[23] = [16, 0]
                self.pattern[24] = [0, 0]
            elif prefix_type == 'hulusi':
                self.minpitch = 5  # 低音5
                self.pattern[5] = [15, 14]
                self.pattern[6] = [15, 6]
                self.pattern[7] = [15, 2]
                self.pattern[11] = [15, 0]
                self.pattern[12] = [7, 0]
                self.pattern[13] = [3, 0]
                self.pattern[14] = [13, 14]
                self.pattern[15] = [1, 0]
                self.pattern[16] = [0, 0]
            elif prefix_type == 'xun':
                self.minpitch = 5  # 低音5
                self.pattern[5] = [31, 31]
                self.pattern[6] = [15, 31]
                self.pattern[7] = [23, 31]
                self.pattern[11] = [7, 31]
                self.pattern[12] = [3, 31]
                self.pattern[13] = [3, 15]
                self.pattern[14] = [15, 7]
                self.pattern[15] = [3, 7]
                self.pattern[16] = [3, 3]
                self.pattern[17] = [3, 2]
                self.pattern[21] = [2, 2]
                self.pattern[22] = [0, 0]
            else:
                # 如果 prefix_type 不匹配任何已定义类型，则引发异常
                raise ValueError(f"Unknown prefix_type: {prefix_type}")

    def handmodel(self, img):
        result = self.hands.process(img)
        if result.multi_hand_landmarks:
            fingertips = [[0] * 21 for _ in range(4)]
            for id in range(len(result.multi_hand_landmarks)):
                landmark_color = (0, 0, 0)
                connection_color = (255, 255, 255)
                landmark_drawing_spec = mediapipe.solutions.drawing_utils.DrawingSpec(color=landmark_color, thickness=2, circle_radius=2)
                connection_drawing_spec = mediapipe.solutions.drawing_utils.DrawingSpec(color=connection_color, thickness=2, circle_radius=2)
                mediapipe.solutions.drawing_utils.draw_landmarks(img, result.multi_hand_landmarks[id], mediapipe.solutions.hands.HAND_CONNECTIONS, landmark_drawing_spec, connection_drawing_spec)
                for point in range(21):
                    fingertips[id][point] = self.Fingertip(id, point, result, self.w, self.h)
                if id == 1:
                    self.num, img = self.gesture(img, fingertips)
        else:
            self.num = -1
        return img

    def gesture(self, img, fingertips):
        pattern = self.inf.pattern
        handnum = [0, 0]
        for id in range(2):
            if fingertips[id][1].type == 'Left':
                indice = self.inf.indices[0]
            else:
                indice = self.inf.indices[1]
            pts = []
            for i in [13, 9, 5, 6, 10, 14, 18, 17]:
                pts.append(fingertips[id][i].xy)
            pts = numpy.array(pts, numpy.int32)
            distance_517 = ((fingertips[id][17].cx - fingertips[id][5].cx) ** 2 + (fingertips[id][17].cy - fingertips[id][5].cy) ** 2) ** 0.5
            slope = abs((fingertips[id][4].cy - fingertips[id][2].cy)) / max(abs((fingertips[id][4].cx - fingertips[id][2].cx)), 1)
            for j in indice:
                distance = cv2.pointPolygonTest(pts, fingertips[id][j].xy, measureDist=True)
                if (slope < 1.2 and j == 4) or (j > 4 and distance < -distance_517 / 2.5):
                    handnum[int(fingertips[id][j].type == 'Right')] += 2 ** (j / 4 - 1)
                    img = cv2.circle(img, (fingertips[id][j].cx, fingertips[id][j].cy), 20, (0, 0, 0), -1)
                else:
                    img = cv2.circle(img, (fingertips[id][j].cx, fingertips[id][j].cy), 20, (255, 255, 255), -1)
                    img = cv2.circle(img, (fingertips[id][j].cx, fingertips[id][j].cy), 20, (0, 0, 0), 3)
        for i in range(len(pattern)):
            if handnum == pattern[i]:
                return i, img
        return -1, img

    def draw_volume_curve(self,image):
        draw = ImageDraw.Draw(image)
        width, height = draw.im.size
        max_volume = 200  # y轴最大值
        min_volume = 1  # y轴最小值
        # 转换为图像坐标系中的 y 值
        y_values = [(height - 1) * (1 - (value - min_volume) / (max_volume - min_volume)) for value in self.volume_data]
        # 绘制曲线
        for i in range(1, len(y_values)):
            x1 = (i - 1) * (width - 1) / (len(self.volume_data) - 1)
            y1 = y_values[i - 1]
            x2 = i * (width - 1) / (len(self.volume_data) - 1)
            y2 = y_values[i]
            draw.line([(x1, y1), (x2, y2)], fill="blue", width=2)
        # 绘制红色水平线
        mid_y = height // 2  # 画布中间的 y 坐标
        draw.line([(0, mid_y), (width - 1, mid_y)], fill="red", width=2)

    def draw_pitch_curve(self,image):
        draw = ImageDraw.Draw(image)
        width, height = draw.im.size
        max_pitch = self.inf.size  # y轴最大值
        min_pitch = -1  # y轴最小值
        axis_x = 50  # 坐标轴位置（用于标记）

        # 加载字体
        font = ImageFont.truetype("simhei.ttf", size=25)   # 或使用 ImageFont.load_default()

        text = "虚拟演奏——爱乐智呈\n官方网址：www.hitaiyuezhicheng.com"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]  # 计算文本宽度
        text_height = text_bbox[3] - text_bbox[1]  # 计算文本高度
        text_x = (width - text_width) // 2  # 计算文本的x坐标以居中
        text_y = (height - text_height) // 2  # 计算文本的y坐标以居中
        # 在图像中绘制文字
        draw.text((text_x, text_y), text, font=font, fill=(220, 220, 220))  # 使用浅灰色

        # 计算 y 值
        processed_values = [value - self.inf.minpitch - 3 * (value // 10 - self.inf.minpitch // 10)
                            for value in self.pitch_data]
        # 计算 y_values
        y_values = [(height - 1) * (1 - (processed_value - min_pitch) / (max_pitch - min_pitch))
                    for processed_value in processed_values]
        # 绘制纵坐标标记（从1到11）
        previous_point = None

        # 绘制音符
        for i in range(len(processed_values)):
            x = axis_x + i * (width - axis_x - 10) / (len(processed_values) - 1)
            y = y_values[i]
            if processed_values[i] >= 0:  # 只处理值非负的点
                if previous_point is not None:
                    prev_x, prev_y = previous_point
                    # 只在 y 值相同时连接
                    if y == prev_y:
                        draw.line([(prev_x, prev_y), (x, y)], fill="blue", width=10)
            previous_point = (x, y)
        # 加载加粗字体

        pitch = self.inf.minpitch
        for i in range(self.inf.size):
            y = height - (i + 1) * (height / (self.inf.size + 1))
            if pitch % 10 > 7:
                pitch += 3
            a = int(height / self.inf.size)
            if i == processed_values[99]:
                img = self.yinfu_red[pitch].convert("RGBA")
            else:
                img = self.yinfu_black[pitch].convert("RGBA")
            img = img.resize((a, a), Image.LANCZOS)
            image.paste(img, (0, int(y - 15)), img)
            pitch += 1

    def draw_zhidao(self, image):
        # 判断是否需要更新 number
        if self.now == self.numbers[self.number] and self.now == self.numbers[self.number - 1] and self.check == 1:
            self.number += 1
            self.check = 0
        elif self.numbers[self.number] == self.numbers[self.number - 1] and self.now == -1:
            self.check = 1
        elif self.now == self.numbers[self.number] and self.now != self.numbers[self.number - 1]:
            self.number += 1

        if 0 <= self.number < len(self.numbers):
            draw = ImageDraw.Draw(image)

            try:
                font = ImageFont.truetype("simhei.ttf", 25)
            except IOError:
                font = ImageFont.load_default()

            # 写入 numbers[number] 到 numbers[number+4]
            num_count = min(self.number + 5, len(self.numbers))
            for i in range(self.number, num_count):
                img = self.yinfu_black[self.numbers[i]].convert("RGBA")
                img = img.resize((50, 50), Image.LANCZOS)
                image.paste(img, (0 + 70 * (i - self.number) + 5, 40), img)

            next_method_text = f"下一个简谱的演奏指法 :"
            bbox = draw.textbbox((0, 0), next_method_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = (image.width - text_width) // 2
            draw.text((text_x, 100), next_method_text, font=font, fill=(0, 0, 0))

            next_method_text = f"乐谱（后续5个简谱）"
            bbox = draw.textbbox((0, 0), next_method_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = (image.width - text_width) // 2
            draw.text((text_x, 5), next_method_text, font=font, fill=(0, 0, 0))

            # 图像合成
            if self.numbers[self.number] in self.images:
                img = self.yinfu_red[self.numbers[self.number]].convert("RGBA")
                image.paste(img, (100, 420), img)
                img = self.images[self.numbers[self.number]].convert("RGBA")
                image.paste(img, (-70, 130), img)

    def add_canvases(self, image):
        # 创建两个画布
        canvas1 = Image.new('RGBA', (image.width + 360, 300), (255, 255, 255, 0))
        canvas2 = Image.new('RGBA', (360, 120), (255, 255, 255, 0))
        boundary = Image.new('RGBA', (360, 10), (0, 0, 0, 0))
        canvas3 = Image.new('RGBA', (360, 580), (255, 255, 255, 0))
        # 绘制 volume_data 曲线在 canvas2 上
        self.draw_pitch_curve(canvas1)
        self.draw_volume_curve(canvas2)
        self.draw_zhidao(canvas3)
        # 创建一个合成图像，尺寸为原图的宽度加上两个画布的宽度
        new_image = Image.new('RGBA', (image.width + 360, image.height + 360), (255, 255, 255, 0))  # 高度增加

        # 粘贴两个画布到合成图像的顶部
        new_image.paste(canvas1, (0, 0))
        new_image.paste(boundary, (0, 300))
        new_image.paste(canvas2, (0, 310))
        new_image.paste(boundary, (0, 430))
        new_image.paste(canvas3,(0, 440))
        # 粘贴原图到合成图像的下方
        new_image.paste(image, (360, 300))
        return new_image

    def start_play(self):
        with self.stream:
            self.main_view()

    def main_view(self):
        while True:
            ret, img = self.cap.read()
            if not ret:
                break
            img = cv2.flip(img, 1)  # 水平翻转图像
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换颜色空间
            img = self.handmodel(img)  # 处理图像
            img = Image.fromarray(img.astype(numpy.uint8))  # 转换为PIL图像
            img = img.convert('RGBA')  # 确保图像是RGBA模式
            if self.num >= 1:
                img.alpha_composite(self.images[self.num], dest=(-70, 0))  # 叠加乐器图像

            img = self.add_canvases(img)
            img = numpy.array(img)  # 转换回numpy数组
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)  # 转换颜色空间回BGR
            cv2.imshow("Camera", img)  # 显示图像
            if cv2.waitKey(1) in [ord('q'), 27]:  # 按'q'或Esc退出
                break

        pygame.mixer.quit()
        self.cap.release()  # 释放摄像头
        cv2.destroyAllWindows()  # 关闭所有窗口

    def audio_callback(self, indata, outdata, frames, time):
        # 计算音量规范化值
        volume_norm = numpy.linalg.norm(indata) * 1000
        self.volume_data.append(min(190, volume_norm))

        if len(self.volume_data) > 100:  # 限制历史数据长度为100
            self.volume_data.pop(0)

        if volume_norm >= 100:
            if self.num >= 0:
                audio_name = f"{self.num}.wav"
                if self.now != self.num:  # 判断num是否发生变化
                    if self.channel is not None and self.channel.get_busy():
                        self.channel.fadeout(150)  # 停止当前正在播放的音频
                    self.now = self.num
                    if audio_name in self.audios:  # 检查音频源是否存在
                        self.channel = self.audios[audio_name].play()  # 直接从预加载的字典中获取并播放
                    else:
                        self.channel = None  # 找不到音源时不报错
        else:
            if self.channel is not None and self.channel.get_busy():
                self.channel.fadeout(150)
                self.now = -1
        self.pitch_data.append(self.now)
        if len(self.pitch_data) > 100:  # 限制历史数据长度为100
            self.pitch_data.pop(0)


# 全局变量和函数转换为类属性和方法后，可以实例化类并调用方法
if __name__ == "__main__":
    instrument = ChuizouYue('taodi')
    instrument.start_play()
