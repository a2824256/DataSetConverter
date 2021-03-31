# encoding=utf-8
import os
from collections import defaultdict
import csv
import cv2
import misc_utils as utils
from PIL import Image
import numpy as np
################ 需要修改的部分-start ###################
# csv文件路径
csv_path = 'train.csv'
# 图片通道，单通道1, rgb 3
channel = 1
# 验证集占比, 范围0~1, 只有训练集设置为0
val_rate = 0
# 图片格式
format = '.jpg'
# 你的数据集图片路径
training_set_path = os.path.join('dataset', 'train')
# 修改为自己的分类
classes = [
    'Aortic_enlargement',
    'Atelectasis',
    'Calcification',
    'Cardiomegaly',
    'Consolidation',
    'ILD',
    'Infiltration',
    'Lung_Opacity',
    'Nodule_Mass',
    'Other_lesion',
    'Pleural_effusion',
    'Pleural_thickening',
    'Pneumothorax',
    'Pulmonary_fibrosis',
]

# 根据自己实际的csv文件的行排布修改该函数
def line_extractor(line):
    image_id, class_name, class_id, rad_id, x_min, y_min, x_max, y_max = line
    filename = image_id + format
    image_path = os.path.join(training_set_path, filename)
    image = Image.open(image_path)
    nd_image = np.array(image)

    if len(nd_image.shape) == 2:
        height, width = nd_image.shape
    else:
        height, width, _ = nd_image.shape

    x_min, y_min, x_max, y_max = int(float(x_min)), int(float(y_min)), int(float(x_max)), int(float(y_max))
    return filename, image_id, class_name, class_id, rad_id, x_min, y_min, x_max, y_max, height, width

################ 需要修改的部分-end ###################
def dir_generator():
    os.makedirs('Annotations', exist_ok=True)
    print('Annotations目录已生成')
    os.makedirs('ImageSets/Main', exist_ok=True)
    print('ImageSets/Main目录已生成')

def csv_loader():
    files = os.listdir(training_set_path)
    files.sort()

    mem = defaultdict(list)

    with open(csv_path, 'r') as f:
        csv_file = csv.reader(f)

        for i, line in enumerate(csv_file):
            if i == 0:
                continue
            # 有需要可以调整该部分，该部分会自动将同一张图片不同的bbox加进去
            filename, image_id, class_name, class_id, rad_id, x_min, y_min, x_max, y_max, height, width = line_extractor(
                line)

            if x_min == '' or y_min == '' or x_max == '' or y_max == '':
                continue
            mem[image_id].append([x_min, y_min, x_max, y_max, class_id, height, width])
    print("csv数据加载结束")
    return mem

def annotation_generator(mem):
    for i, image_id in enumerate(mem):
        utils.progress_bar(i, len(mem), 'handling...')
        with open(os.path.join('Annotations', image_id + '.xml', 'w')) as f:
            f.write(f"""<annotation>
        <folder>train</folder>
        <filename>{image_id}.jpg</filename>
        <size>
            <width>{str(mem[image_id][0][6])}</width>
            <height>{str(mem[image_id][0][5])}</height>
            <depth>{channel}</depth>
        </size>
        <segmented>0</segmented>\n""")
            for x1, y1, x2, y2, _, _, _ in mem[image_id]:
                f.write(f"""    <object>
            <name>{classes[int(mem[image_id][0][4])]}</name>
            <pose>Unspecified</pose>
            <truncated>0</truncated>
            <difficult>0</difficult>
            <bndbox>
                <xmin>{x1}</xmin>
                <ymin>{y1}</ymin>
                <xmax>{x2}</xmax>
                <ymax>{y2}</ymax>
            </bndbox>
        </object>\n""")
            f.write("</annotation>")

    print("annotation生成结束")

def txt_generator(mem):
    files = list(mem.keys())
    files.sort()
    f1 = open('ImageSets/Main/train.txt', 'w')
    f2 = open('ImageSets/Main/val.txt', 'w')
    train_count = 0
    val_count = 0

    with open('ImageSets/all.txt', 'w') as f:
        for image_id in files:
            f.writelines(image_id + '\n')

            if utils.gambling(val_rate):
                f2.writelines(image_id + '\n')
                val_count += 1
            else:
                f1.writelines(image_id + '\n')
                train_count += 1

    f1.close()
    f2.close()
    print(f'随机划分 训练集: {train_count}张图，测试集：{val_count}张图')


if __name__ == "__main__":
    dir_generator()
    mem = csv_loader()
    annotation_generator(mem)
    txt_generator(mem)