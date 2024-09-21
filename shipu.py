import cv2
import CNN
import numpy as np


# 读取图像
image = cv2.imread('lzlh.png')
ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# 转换为灰度图像

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用阈值化，将灰度图像转换为二值图像，并反转颜色
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# 寻找图像中的轮廓
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 初始化最小正方形的边长出现次数的字典
side_lengths_count = {}

# 计算边长允许的误差范围
allowed_error = 0.1

# 遍历轮廓
for contour in contours:
    # 获取轮廓的边界框
    x, y, w, h = cv2.boundingRect(contour)
    # 计算最小正方形的边长
    side_length = max(w, h)
    # 将边长添加到字典中并更新出现次数
    side_lengths_count[side_length] = side_lengths_count.get(side_length, 0) + 1

# 找到边长出现次数最多的最小正方形的边长
most_common_side_length = max(side_lengths_count, key=side_lengths_count.get)

# 初始化同一y轴上的轮廓计数器
y_counts = {}
# 存储筛选后的轮廓

selected_contours = []
# 统计同一y轴上的轮廓数量

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    side_length = max(w, h)
    allowed_difference = side_length * allowed_error
    if (abs(side_length - most_common_side_length) <= allowed_difference):
        rounded_y = round(y / 10) * 10
        y_counts[rounded_y] = y_counts.get(rounded_y, 0) + 1
        y_counts[rounded_y+10] = y_counts.get(rounded_y+10, 0) + 1
        y_counts[rounded_y - 10] = y_counts.get(rounded_y - 10, 0) + 1

# 筛选轮廓
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    side_length = max(w, h)
    allowed_difference = side_length * allowed_error
    if (abs(side_length - most_common_side_length) <= allowed_difference) and (y_counts[round(y / 10) * 10] >10):
        selected_contours.append(contour)
    else:
        if side_length >= most_common_side_length / 2.5:
            cv2.drawContours(image, [contour], 0, (255, 255, 255), -1)

# 定义排序规则：先按 y 坐标排序，再按 x 坐标排序
def sort_key(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return (y, x)


# 对筛选后的轮廓进行排序
selected_contours.sort(key=sort_key)
results = []

for contour in selected_contours:
    # 获取轮廓的边界框
    x, y, w, h = cv2.boundingRect(contour)

    top_left_x = int(x+w/ 2-max(h,w)/2-2)
    top_left_y = y-1
    bottom_right_x = int(x+w/2+max(h,w)/2+2)
    bottom_right_y = y+h+2
    color = (0, 255, 0)  # 绿色框
    thickness = 2  # 框的厚度
    image_with_rectangle = cv2.rectangle(image,
                                         (top_left_x, top_left_y),
                                         (bottom_right_x, bottom_right_y),
                                         color,
                                         thickness)
    # 识别轮廓的图像
    digit_image = thresh[top_left_y-2:bottom_right_y, top_left_x:bottom_right_x+2]
    digit_label = CNN.recognize_digits([digit_image])[0]  # 获取识别结果

    # 计算上下延伸的高度

    extended_height=h//3

    # 提取上下部分的图像区域
    upper_part = thresh[max(top_left_y - extended_height, 0):top_left_y-2, top_left_x:bottom_right_x]
    lower_part = thresh[bottom_right_y+2:bottom_right_y + extended_height, top_left_x:bottom_right_x]

    # 定义阈值
    MIN_WHITE_PIXELS = 4

    # 计算每个区域中白色像素的数量
    upper_white_pixel_count = np.sum(upper_part >= 200)
    lower_white_pixel_count = np.sum(lower_part >= 200)

    # 检查是否有足够数量的白色像素
    upper_has_white = upper_white_pixel_count >= MIN_WHITE_PIXELS
    lower_has_white = lower_white_pixel_count >= MIN_WHITE_PIXELS

    # 根据白色像素的存在和数量调整结果数字
    if upper_has_white:
        adjusted_digit_label = digit_label + 20
    elif lower_has_white:
        adjusted_digit_label = digit_label
    else:
        adjusted_digit_label = digit_label + 10

    # 显示切片图像（可选）
    # 计算文本位置
    text_x = top_left_x + h // 2
    text_y = top_left_y + h // 2
    # 绘制识别结果文本
    cv2.putText(image, str(adjusted_digit_label), (text_x, text_y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    if 1 <= adjusted_digit_label <= 7 or 11 <= adjusted_digit_label <= 17 or 21 <= adjusted_digit_label <= 27:
        results.append(str(adjusted_digit_label))

# 写入到文本文件
with open('results.txt', 'w') as f:
    f.write('\\'.join(results))
# 显示结果
image = cv2.resize(image, (800,1000))
cv2.imshow('Filtered and Sorted Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()