import os
import json
import random
import shutil
from PIL import Image
from tqdm import tqdm
import re

dirs = ['train','val','test']
classes =['mobilephone','cup']
# img_path = 'datasets/image'
lbl_path = 'datasets/train.json'


def convert(size,box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[2]) / 2.
    y = (box[1] + box[3]) / 2.
    w = box[2] - box [0]
    h = box[3] - box[1]
    x = round(x * dw, 4)
    w = round(w * dw, 4)
    y = round(y * dh, 4)
    h = round(h * dh, 4)

    return (x,y,w,h)

def bbd2yolo():
    with open(lbl_path) as f:
        labels = json.load(f)
    keys = []
    for key in labels:
        keys.append(key)
    all = len(keys)
    random.shuffle(keys)
    image_num = {'train':keys[0:int(all * 0.85)],
                 'val':keys[int(all * 0.85):int(all * 0.85)+int(all*0.1)],
                 'test':keys[int(all * 0.85)+int(all*0.1):all]}
    for dir in dirs:
        if not os.path.exists(os.path.join('dsm_data/images',dir)):
            # os.mkdir()
            os.makedirs(os.path.join('dsm_data/images',dir))
        if not os.path.exists(os.path.join('dsm_data/labels',dir)):
            os.makedirs(os.path.join('dsm_data/labels',dir))
        pbar = tqdm(image_num[dir])
        for image in pbar:
            image = image.split('/')[1]
            pbar.set_description("Processing %s" % image)
            if os.path.exists(os.path.join('datasets','image/' + image)):
                img = Image.open(os.path.join('datasets','image/' + image)).convert('RGB')
                size = img.size
                shutil.move(os.path.join('datasets','image/' + image),os.path.join('dsm_data/images',dir))   #move iamge
                with open(os.path.join('dsm_data',dir + '.txt'),'a') as f:
                    # print(os.path.join('dsm_data',dir + '/images/' + image))
                    # if os.path.exists(os.path.join('dsm_data',dir + '/images/' + image)):
                    f.write(os.path.join('dsm_data',dir + '/' + image) + '\n')

                targets = labels['image/'+ image]
                bbox = targets['bbox']
                convert_box = convert(size,bbox[0])
                category_name = targets['category_name'][0]
                cls_id = classes.index(category_name)
                out_content = str(cls_id) + " " + " ".join([str(a) for a in convert_box]) + '\n'

                image = re.findall(r'(.+?)\.', image)[0]
                # print(os.path.join('dsm_data/labels', dir + '/' + image + '.txt'))
                with open(os.path.join('dsm_data/labels',dir + '/' + image + '.txt'),'w') as f:
                    f.write(out_content)


if __name__ == '__main__':
    # bbd2yolo()
    print()





