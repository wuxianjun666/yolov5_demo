import json

import numpy as np
import matplotlib.patches as patches
from matplotlib.image import imread
import math
import random
import os
import xml.etree.ElementTree as ET

OBJ_NAMES = ['mobilephone','cup']

def get_insect_names():
    insect_category2id = {}
    for i, item in enumerate(OBJ_NAMES):
        insect_category2id[item] = i
    return insect_category2id


def get_annotations():
    # filenames = os.listdir(os.path.join(datadir, 'annotations', 'xmls'))
    # records = []
    # ct = 0
    # for fname in filenames:
    #     fid = fname.split('.')[0]
    #     fpath = os.path.join(datadir, 'annotations', 'xmls', fname)
    #     img_file = os.path.join(datadir, 'images', fid + '.jpeg')
    #     tree = ET.parse(fpath)  # 解析每一个 xml文件
    #     if tree.find('id') is None:
    #         im_id = np.array([ct])
    #     else:
    #         im_id = np.array([int(tree.find('id').text)])
    #     objs = tree.findall('object')  # 拿到所有obj的内容
    #     im_w = float(tree.find('size').find('width').text)
    #     im_h = float(tree.find('size').find('height').text)
    #     gt_bbox = np.zeros((len(objs), 4), dtype=np.float32)
    #     gt_class = np.zeros((len(objs), ), dtype=np.int32)
    #     is_crowd = np.zeros((len(objs), ), dtype=np.int32)
    #     difficult = np.zeros((len(objs), ), dtype=np.int32)
    #     for i, obj in enumerate(objs):  #具体拿每个obj的内容
    #         cname = obj.find('name').text
    #         gt_class[i] = cname2cid[cname]
    #         _difficult = int(obj.find('difficult').text)
    #         x1 = float(obj.find('bndbox').find('xmin').text)
    #         y1 = float(obj.find('bndbox').find('ymin').text)
    #         x2 = float(obj.find('bndbox').find('xmax').text)
    #         y2 = float(obj.find('bndbox').find('ymax').text)
    #         x1 = max(0, x1)
    #         y1 = max(0, y1)
    #         x2 = min(im_w - 1, x2)
    #         y2 = min(im_h - 1, y2)
    #         gt_bbox[i] = [(x1+x2)/2.0 , (y1+y2)/2.0, x2-x1+1., y2-y1+1.]
    #         is_crowd[i] = 0
    #         difficult[i] = _difficul
    #
    #     voc_rec = {
    #         'im_file': img_file,
    #         'im_id': im_id,
    #         'h': im_h,
    #         'w': im_w,
    #         'is_crowd': is_crowd,
    #         'gt_class': gt_class,
    #         'gt_bbox': gt_bbox,
    #         'gt_poly': [],
    #         'difficult': difficult
    #         }
    #     if len(objs) != 0:
    #         records.append(voc_rec)
    #     ct += 1
    records = []
    labels_path = 'train.json'
    keys = []
    with open(labels_path) as f:
        lbls = json.load(f)
        for key in lbls:
            keys.append(key)
            targets = lbls[key]
            _boxes = targets['bbox']
            if 'category_name' in targets:
                _labels = targets['category_name']
            boxes = []
            labels = []
            for b, l in zip(_boxes, _labels):
                if l in OBJ_NAMES:
                    boxes.append(b)
                    labels.append(OBJ_NAMES.index(l))
            im_w = 960
            im_h = 540
            x1 = _boxes[0][0]
            y1 = _boxes[0][1]
            x2 = _boxes[0][2]
            y2 = _boxes[0][3]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(im_w - 1, x2)
            y2 = min(im_h - 1, y2)
            gt_bbox = [(x1+x2)/2.0 , (y1+y2)/2.0, x2-x1+1., y2-y1+1.]
            rec = {
                        'im_file': key,
                        # 'im_id': im_id,
                        'h': im_h,
                        'w': im_w,
                        # 'is_crowd': is_crowd,
                        'gt_class': labels,
                        'gt_bbox': gt_bbox,
                        # 'gt_poly': [],
                        # 'difficult': difficult
                        }
            records.append(rec)

    return records



# sum_records = get_annotations()

def get_allbox(sum_records):
    gt_bboxes = sum_records[0]['gt_bbox']
    # 利用numpy的合并方法，获得所有真实框构成的数组 gt_bboxes 是一个（12203，4）的数组
    for i in range(len(sum_records)-1):
        gt_bbox = sum_records[i+1]['gt_bbox']
        gt_bboxes = np.vstack((gt_bboxes, gt_bbox))
    return  gt_bboxes

# bboxes = get_allbox(sum_records)
# print(bboxes.shape)

def init_anchors(gt_bboxes,seed):
    '''
    gt_bboxes 是一个（N，4）的数组，N对应个数，4对应（x,y,w,h）
    这个值通过函数get_allbox(sum_records)获得
    '''
    gtbox_num = gt_bboxes.shape[0]
    index_list = range(gtbox_num)
    # 随机选取9个框，为便于程序调试和结果观察，设置为随机数种子，使得随机产生的9个数始终相同。
    # 而且再所有程序设计完成后，改变这个随机数种子，也就是改变9个ancho的初始值
    # 可以观察最终结果有所波动，不过波动范围不是很大
    random.seed(seed)
    random_num = random.sample(index_list,9)
    anchors = []  # 装了9个数组（框的坐标即初始化的anchors）的列表
    for i in random_num:
        anchor = gt_bboxes[i]
        anchors.append(anchor)
    return anchors

# anchors = init_anchors(bboxes,1)

#  -------------------------第二步：使用IOU度量，将每个box分配给与其距离最近的anchor；-----------------------------------
#  只关心w,h，默认两者的x,y相同，以此计算iou
def box_iou_wh(box1, box2):
    w1,h1 = box1[2:4]
    w2,h2 = box2[2:4]
    s1 = w1*h1
    s2 = w2*h2
    intersection = min(h1,h2) * min(w1,w2)
    if ((w1 < w2) and (h1 < h2)) or ((w1 > w2) and (h1 > h2)):
        union = max(w1,w2) * max(h1,h2)
    else:
        union = s1 + s2 - intersection
    iou = intersection / union
    return iou

# 开始分簇，求均值，更新anchors
def kmeans(anchors, boxes, anchors_num):
    loss = 0
    groups = []
    new_anchors = []
    # 创建9个聚类
    for i in range(anchors_num):
            groups.append([])
    # 遍历每个框
    for box in boxes:
        ious = []
        # 遍历每个初始聚类中心anchor，计算当前box与每个中心的iou，找出最大的IOU后将当前box归为对应的类
        for anchor in anchors:
            iou = box_iou_wh(box, anchor)
            ious.append(iou)
        index_of_maxiou = ious.index(max(ious))
        groups[index_of_maxiou].append(box)

    # 求每个聚类中，框的w, h 的均值
    for group in groups:
        w_sum = 0
        h_sum = 0
        length = len(group)
        for box_in_group in group:
            w_sum += box_in_group[2]
            h_sum += box_in_group[3]
        w_mean = w_sum / length
        h_mean = h_sum / length
        # 计算iou时并不关心xy， 所以这里xy设置为默认0
        anchor = np.array([0,0,w_mean,h_mean])
        new_anchors.append(anchor)
    return new_anchors


#  -------------第三步：重复调用kmean函数，直到满足要求：Ⅰ循环次数结束，或者Ⅱ平均值不再变化（代表找到了该类的中心）--------------
def do_kmeans(anchors, boxes, anchors_num, cycle_num):
    cycle = 0
    new_anchors = kmeans(anchors, boxes, anchors_num)
    while True:
        final_anchors = new_anchors
        new_anchors = kmeans(new_anchors, boxes, anchors_num)
        # for anchor in new_anchors:
        # loss = final_anchors -
        cycle += 1
        # if cycle % 10 == 0:
        #     print('循环了%d次'%(cycle))
        flag = np.zeros((9))
        for i in range(len(final_anchors)):
            equal = (new_anchors[i] == final_anchors[i]).all()
            flag[i] = equal
        if flag.all():
            print('循环了', cycle, '次，终于找到了中心anchors')
            break
        if cycle == cycle_num:
            print('循环次数使用完毕')
            break
    # 截取 w ，h
    final_anchors = [anchor[2:4].astype('int32') for anchor in final_anchors]
    # 由小到大排序
    final_anchors = sorted(final_anchors, key=lambda anchor: anchor[0])
    # 换成YOLOV3算法中需要的形式，即变成一个列表[w,h,w,h...w,h,w,h]
    true_final_anchors = []
    for anchor in final_anchors:
        true_final_anchors.append(anchor[0])
        true_final_anchors.append(anchor[1])
    return true_final_anchors


#--------------------------------------------------验证结果----------------------------------------
def test(seed=1):
    # TRAINDIR = '/home/aistudio/work/insects/train'
    # TESTDIR = '/home/aistudio/work/insects/test'
    # VALIDDIR = '/home/aistudio/work/insects/val'
    cname2cid = get_insect_names()
    sum_records = get_annotations()
    # 最大化利用已知数据，因此验证集上的信息我们也要统计
    # valid_records = get_annotations(cname2cid, VALIDDIR)
    gt_bboxes = get_allbox(sum_records)
    # 随机初始化anchors
    anchors = init_anchors(gt_bboxes,seed=seed)
    # 设置聚类个数K，这里指要生成的锚框大小个数
    anchors_num = 9
    # 设定迭代次数
    cycle_num = 10000
    # 进行kmeans算法迭代
    final_anchors = do_kmeans(anchors,gt_bboxes,anchors_num,cycle_num)
    print(final_anchors)
    return final_anchors

test(1)
def get_aver_anchors(num_random):
    # num_random 为随机次数，可以手动设定
    seed_list = range(num_random)
    # 存放所有anchor 方便求平均值
    all_anchors = []
    for seed in seed_list:
        print('种子号是%d'%seed)
        anchors = test(seed)  # 由于要打印的东西太多，所以我们把do_kmeans中打印循环次数的代码注释掉
        # 列表转换为数组方便计算平均值
        anchors = np.array(anchors) # anchors里有9个anchor， 9*2=18 共18个数据
        all_anchors.append(anchors)
    # 同样，把大列表转换成数组
    all_anchors = np.array(all_anchors)
    aver_anchors = np.mean(all_anchors, axis=0).astype('int32')
    return(aver_anchors)


aver_anchors = get_aver_anchors(20)
print('平均anchors：\n',aver_anchors)