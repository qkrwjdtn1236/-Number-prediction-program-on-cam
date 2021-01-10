import tensorflow as tf
import cv2
import numpy as np
# import math


def process(img_input):
    gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)  #그레이스케일

    gray = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)#가로 세로 28 28픽셀로 축소

    (thresh, img_binary) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU) #
    # cv2.imshow('test',img_binary)
    h, w = img_binary.shape
    print(h,w)

    ratio = 100 / h
    new_h = 100
    new_w = w * ratio

    img_empty = np.zeros((110, 110), dtype=img_binary.dtype) #검은화면 0~1사이 값 범위 초기값 0
    img_binary = cv2.resize(img_binary, (int(new_w), int(new_h)), interpolation=cv2.INTER_AREA)
    img_empty[:img_binary.shape[0], :img_binary.shape[1]] = img_binary

    img_binary = img_empty

    cnts = cv2.findContours(img_binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 컨투어의 무게중심 좌표를 구합니다.
    M = cv2.moments(cnts[0][0])
    center_x = (M["m10"] / M["m00"])
    center_y = (M["m01"] / M["m00"])

    # 무게 중심이 이미지 중심으로 오도록 이동시킵니다.
    height, width = img_binary.shape[:2]
    shiftx = width / 2 - center_x
    shifty = height / 2 - center_y

    Translation_Matrix = np.float32([[1, 0, shiftx], [0, 1, shifty]])
    img_binary = cv2.warpAffine(img_binary, Translation_Matrix, (width, height))

    img_binary = cv2.resize(img_binary, (28, 28), interpolation=cv2.INTER_AREA)
    flatten = img_binary.flatten() / 255.0

    return flatten


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10,activation='softmax')
])

model.load_weights('mnist_checkpoint')

cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while (True):

    ret, img_color = cap.read()

    if ret == False:
        break

    img_input = img_color.copy()
    cv2.rectangle(img_color, (250, 150), (width - 250, height - 150), (0, 0, 255), 3)
    cv2.imshow('bgr', img_color)

    img_roi = img_input[150:height - 150, 250:width - 250]

    key = cv2.waitKey(1)

    if key == 27:
        break
    elif key == 32:
        flatten = process(img_roi)

        predictions = model.predict(flatten[np.newaxis, :])

        with tf.compat.v1.Session() as sess:
            print(tf.argmax(predictions, 1).eval())

        cv2.imshow('img_roi', img_roi)
        cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()