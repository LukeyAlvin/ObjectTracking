from absl import flags
import sys

FLAGS = flags.FLAGS
FLAGS(sys.argv)

import os
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import gc
import queue

import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from multiprocessing import Process, Manager
from _collections import deque

pts = [deque(maxlen=30) for _ in range(1000)]
counter = []

q = queue.Queue()

# 调用海康威视摄像头
url = "rtsp://admin:WL19990620@192.168.1.64/Streaming/Channels/1"
# url = './data/video/test.mp4'

# cap = cv2.VideoCapture('./data/video/test.mp4')

class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights('./weights/yolov3.tf')

max_cosine_distance = 0.5
nn_budget = None
nms_max_overlap = 0.8

model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)


#<editor-fold desc="设置显示分辨率">
def img_resize(image):
    height, width = image.shape[0], image.shape[1]
    # 设置新的图片分辨率框架 640x369 1280×720 1920×1080
    width_new = 1280
    height_new = 720
    # 判断图片的长宽比率
    if width / height >= width_new / height_new:
        img_new = cv2.resize(image, (width_new, int(height * width_new / width)))
    else:
        img_new = cv2.resize(image, (int(width * height_new / height), height_new))
    return img_new


#</editor-fold>

# #<editor-fold desc="读取本地视频并输出">
# def local_video(cap):
#     codec = cv2.VideoWriter_fourcc(*'XVID')
#     cap_fps =int(cap.get(cv2.CAP_PROP_FPS))
#     cap_width,cap_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     out = cv2.capeoWriter('./data/video/Result.avi', codec, cap_fps, (cap_width, cap_height))
#     return out
# #</editor-fold>
#
#

#<editor-fold desc="yolo模型处理">
def yolo_deal(img):
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in = tf.expand_dims(img_in, 0)
    img_in = transform_images(img_in, 416)

    t1 = time.time()

    boxes, scores, classes, nums = yolo.predict(img_in)

    classes = classes[0]
    names = []
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])
    names = np.array(names)
    converted_boxes = convert_boxes(img, boxes[0])
    features = encoder(img, converted_boxes)

    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores[0], names, features)]

    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    tracker.predict()
    tracker.update(detections)

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

    # current_count = int(0)
    i = 0
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue
        bbox = track.to_tlbr()
        class_name = track.get_class()
        if class_name == "person":
            i += 1
        if class_name == "chair":
            continue
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]

        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1] - 30)),
                      (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17,
                       int(bbox[1])), color, -1)
        cv2.putText(img, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75,
                    (255, 255, 255), 2)

        center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
        pts[track.track_id].append(center)

        for j in range(1, len(pts[track.track_id])):
            if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                continue
            thickness = int(np.sqrt(64 / float(j + 1)) * 2)
            cv2.line(img, (pts[track.track_id][j - 1]), (pts[track.track_id][j]), color, thickness)

        height, width, _ = img.shape
        # cv2.line(img, (0, int(3 * height / 6 + height / 20)), (width, int(3 * height / 6 + height / 20)), (0, 255, 0),
        #          thickness=2)
        # cv2.line(img, (0, int(3 * height / 6 - height / 20)), (width, int(3 * height / 6 - height / 20)), (0, 255, 0),
        #          thickness=2)

        center_y = int(((bbox[1]) + (bbox[3])) / 2)

        if center_y <= int(3 * height / 6 + height / 20) and center_y >= int(3 * height / 6 - height / 20):
            if class_name == 'person':
                counter.append(int(track.track_id))
                # current_count += 1

    total_count = i
    # cv2.putText(img, "Current Person Count: " + str(current_count), (0, 80), 0, 1, (0, 0, 255), 2)
    cv2.putText(img, "Total person Count: " + str(total_count), (900, 60), 0, 1, (0, 0, 255), 2)

    if total_count >= 3:
        attendance = "HIGH"
        cv2.putText(img, "Attendance: " + str(attendance), (900, 90), 0, 1, (0, 255, 0), 2)
    else:
        attendance = "LOW"
        cv2.putText(img, "Attendance: " + str(attendance), (900, 90), 0, 1, (255, 0, 0), 2)


    fps = 1. / (time.time() - t1)
    cv2.putText(img, "FPS: {:.2f}".format(fps), (900, 30), 0, 1, (0, 0, 255), 2)
    return img


#</editor-fold>


#<editor-fold desc="保存帧图像">
def save_img(img):
    charp_name = str(time.time()) + '.jpg'
    # charp_name = '2.jpg'
    path = os.path.join('D:\workspace\Object-Detection-and-Tracking\data\img', charp_name)
    cv2.imwrite(path, img)


#</editor-fold>


# 向共享缓冲栈中写入数据:
def write(stack, cam, top: int) -> None:
    """
    :param cam: 摄像头参数
    :param stack: Manager.list对象
    :param top: 缓冲栈容量
    :return: None
    """
    print('Process to write: %s' % os.getpid())
    cap = cv2.VideoCapture(cam)
    while True:
        _, img = cap.read()
        if _:
            stack.append(img)
            # 每到一定容量清空一次缓冲栈
            # 利用gc库，手动清理内存垃圾，防止内存溢出
            if len(stack) >= top:
                del stack[:]
                gc.collect()


# 在缓冲栈中读取数据:
def read(stack) -> None:
    print('Process to read: %s' % os.getpid())
    while True:
        if len(stack) != 0:
            value = stack.pop()
            img_new = img_resize(value)
            yolo_img = yolo_deal(img_new)
            cv2.imshow("img", yolo_img)
            save_img(yolo_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break


if __name__ == '__main__':
    # 父进程创建缓冲栈，并传给各个子进程：
    print("1")
    q = Manager().list()
    pw = Process(target=write, args=(q, url, 1000))
    pr = Process(target=read, args=(q,))
    # 启动子进程pw，写入:
    pw.start()
    # 启动子进程pr，读取:
    pr.start()
    # 等待pr结束:
    pr.join()
    # pw进程里是死循环，无法等待其结束，只能强行终止:
    pw.terminate()
