import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from tensorflow.keras.models import load_model

model_path = 'mnist_model_2.h5'
model = load_model(model_path)

def recognize_digits(images):
    # 初始化一个空列表，用于存储所有预测结果
    predictions = []
    # 遍历输入的图像列表
    for img in images:
        # 将图像调整为模型所需的大小
        img = cv2.resize(img, (28, 28))
        # 如果图像是彩色的，将其转换为灰度图像
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 将图像reshape成模型所需的形状，并进行归一化处理
        img = img.reshape(1, 28, 28, 1).astype('float32') / 255
        # 使用模型进行预测
        pred = model.predict(img)
        # 获取预测类别并加入到预测结果列表中
        pred_class = np.argmax(pred, axis=1)[0]
        predictions.append(pred_class)
    # 返回所有图像的预测结果列表
    return predictions
