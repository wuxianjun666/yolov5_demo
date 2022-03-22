import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import re
from tqdm import tqdm
import shutil
# import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
# name_list = ['D00', 'D10', 'D20', 'D40']
name_list = ['crack']
train_labels_dir = os.path.join(ROOT,'my_data/labels/train')
train_images_dir = os.path.join(ROOT,'my_data/images/train')

val_labels_dir = os.path.join(ROOT, 'my_data/labels/val')
val_images_dir = os.path.join(ROOT, 'my_data/images/val')

test_labels_dir = os.path.join(ROOT, 'my_data/labels/test')
test_images_dir = os.path.join(ROOT, 'my_data/images/test')


images_dir = [train_images_dir,val_images_dir,test_images_dir]
labels_dir = [train_labels_dir,val_labels_dir,test_labels_dir]

def xywh2xyxy(size,box):
    xmin = (box[0] - box[2] / 2) * size[1]
    ymin = (box[1] - box[3] / 2) * size[0]
    xmax = (box[0] + box[2] / 2) * size[1]
    ymax = (box[1] + box[3] / 2) * size[0]
    box = (int(xmin), int(ymin), int(xmax), int(ymax))
    return box

def sum_class():
    sum_classes = np.zeros(1)

    for image_dir,label_dir in zip(images_dir,labels_dir):
        draw_dir = os.path.join('/home/wuxj/PycharmProjects/yolov5_demo/draw', image_dir.split('/')[-1])
        # if not os.path.exists(draw_dir):
        #     os.mkdir(draw_dir)
        if os.path.exists(draw_dir):
            shutil.rmtree(draw_dir)  # delete output folder
        os.makedirs(draw_dir)
        files = os.listdir(label_dir)
        files = sorted(files)
        index = 0
        pbar = tqdm(files)
        for file_name in pbar:
            pbar.set_description("Processing %s" % file_name)
            image_name = re.findall(r'(.+?)\.',file_name)[0] + '.jpg'
            img = Image.open(os.path.join(image_dir,image_name))
            with open(os.path.join(label_dir,file_name)) as f:
                data = f.readlines()
                if len(data) == 0:
                    index += 1
                else:
                    labels = []
                    boxes = []
                    for obj in data:
                        message = obj.split(' ')
                        sum_classes[int(message[0])] += 1
                        labels.append(name_list[int(message[0])])
                        boxes.append(xywh2xyxy(img.size,np.array(message[1:5],dtype=float)))
                    boxes = np.array(boxes, dtype=float)
                    labels = np.array(labels, dtype=str)
                    img = draw_bbox(img,boxes,labels)

                img = np.asarray(img)
                cv2.imwrite(os.path.join(draw_dir, image_name), img)
        print(index)
        autolabel(plt.bar(range(len(name_list)), sum_classes, tick_label=name_list))
        plt.show()


def draw_bbox(img, bbox, lbls):
    img = np.array(img, dtype=float)
    img = np.around(img)
    img = np.clip(img, a_min=0, a_max=255).astype(np.uint8)
    for box, lbl in zip(bbox, lbls):
        xmin = int(box[0])
        xmax = int(box[2])
        ymin = int(box[1])
        ymax = int(box[3])
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(0,255,0), thickness=1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, lbl, (xmin + 5, ymax - 5), font, fontScale=0.5, color=(0,255,0), thickness=1)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    return img

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2. - 0.2, 1.03 * height, '%s' % int(height))

def delete_img():
    files = os.listdir(train_labels_dir)
    files = sorted(files)
    index = 0
    pbar = tqdm(files)
    for file_name in pbar:
        pbar.set_description("Processing %s" % file_name)
        image_name = re.findall(r'(.+?)\.', file_name)[0] + '.jpg'
        image_dir_content = 'my_data/images/train/' + image_name + '\n'
        image_dir_delete = '/home/wuxj/PycharmProjects/yolov5_demo/my_data/images/train/' + image_name
        label_dir_delete = '/home/wuxj/PycharmProjects/yolov5_demo/my_data/labels/train/' + file_name
        with open(os.path.join(train_labels_dir, file_name)) as f:
            data = f.readlines()
            if len(data) == 0:
                index += 1
                if os.path.exists(image_dir_delete):
                    os.remove(image_dir_delete)
                if os.path.exists(label_dir_delete):
                    os.remove(label_dir_delete)
            else:
                with open('/home/wuxj/PycharmProjects/yolov5_demo/train.txt','a') as f:
                    f.write(image_dir_content)

    print(index)

def find_locate():

    for image_dir, label_dir in zip(images_dir, labels_dir):

        files = os.listdir(label_dir)
        files = sorted(files)
        index = 0
        count = 0
        pbar = tqdm(files)
        for file_name in pbar:
            pbar.set_description("Processing %s" % file_name)
            image_name = re.findall(r'(.+?)\.', file_name)[0] + '.jpg'
            img = Image.open(os.path.join(image_dir, image_name))
            with open(os.path.join(label_dir, file_name)) as f:
                data = f.readlines()
                if len(data) == 0:
                    index += 1
                else:
                    labels = []
                    boxes = []
                    for obj in data:
                        message = obj.split(' ')
                        # labels.append(name_list[int(message[0])])
                        boxes.append(xywh2xyxy(img.size, np.array(message[1:5], dtype=float)))
                    boxes = np.array(boxes, dtype=float)
                    # labels = np.array(labels, dtype=str)
                    ratios = []
                    for box in boxes:
                        ratio = round(box[1] / img.size[1],3)
                        ratios.append(ratio)
                    for ratio in ratios:
                        if ratio <= 0.5:
                            # print(os.path.join(image_dir, image_name))
                            os.remove(os.path.join(image_dir, image_name))
                            os.remove(os.path.join(label_dir, file_name))
                            count += 1
                            break
        print(count)
        # ratios = np.array(ratios)
        # print(np.sum(ratios < 0.5))



                    # img = draw_bbox(img, boxes, labels)

                # img = np.asarray(img)
        #         cv2.imwrite(os.path.join(draw_dir, image_name), img)
        # print(index)
        # autolabel(plt.bar(range(len(name_list)), sum_classes, tick_label=name_list))
        # plt.show()

# my_data/images/val/Czech_000030.jpg
# my_data/images/train/Japan_012820.jpg




if __name__ == '__main__':
    # delete_img()
    sum_class()
    # find_locate()