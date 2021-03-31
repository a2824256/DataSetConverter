# encoding=utf-8
import os
from collections import defaultdict
import csv
import cv2
import misc_utils as utils
from PIL import Image
import numpy as np

training_set_path = os.path.join('dataset', 'train')
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

os.makedirs('Annotations', exist_ok=True)
print('Annotations目录已生成')
os.makedirs('ImageSets/Main', exist_ok=True)
print('ImageSets/Main目录已生成')

files = os.listdir(training_set_path)
files.sort()

mem = defaultdict(list)

with open('train.csv', 'r') as f:
    csv_file = csv.reader(f)

    for i, line in enumerate(csv_file):
        if i == 0:
            continue
        image_id, class_name, class_id, rad_id, x_min, y_min, x_max, y_max = line
        if x_min == '' or y_min == '' or x_max == '' or y_max == '':
            continue
        filename = image_id + ".jpg"
        image_path = os.path.join(training_set_path, filename)
        image = Image.open(image_path)
        nd_image = np.array(image)
        # width, height
        if len(nd_image.shape) == 3:
            height, width, channel = nd_image.shape
        elif len(nd_image.shape) == 2:
            height, width = nd_image.shape

        x1, y1, x2, y2 = int(float(x_min)), int(float(y_min)), int(float(x_max)), int(float(y_max))

        mem[image_id].append([x1, y1, x2, y2, class_id, height, width])
        print(filename)
print("csv加载结束")

for i, filename in enumerate(mem):
    utils.progress_bar(i, len(mem), 'handling...')
    img = cv2.imread(os.path.join('train', filename))
    with open(os.path.join('Annotations', filename.rstrip('.jpg')) + '.xml', 'w') as f:
        f.write(f"""<annotation>
    <folder>train</folder>
    <filename>{filename}.jpg</filename>
    <size>
        <width>{str(mem[filename][0][6])}</width>
        <height>{str(mem[filename][0][5])}</height>
        <depth>1</depth>
    </size>
    <segmented>0</segmented>\n""")
        for x1, y1, x2, y2, _, _, _ in mem[filename]:
            f.write(f"""    <object>
        <name>{classes[int(mem[filename][0][4])]}</name>
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

files = list(mem.keys())
files.sort()
f1 = open('ImageSets/Main/train.txt', 'w')
f2 = open('ImageSets/Main/val.txt', 'w')
train_count = 0
val_count = 0

with open('ImageSets/all.txt', 'w') as f:
    for filename in files:
        filename = filename.rstrip('.jpg')
        f.writelines(filename + '\n')

        if utils.gambling(0):  # 10%的验证集
            f2.writelines(filename + '\n')
            val_count += 1
        else:
            f1.writelines(filename + '\n')
            train_count += 1

f1.close()
f2.close()

print(f'随机划分 训练集: {train_count}张图，测试集：{val_count}张图')


# if __name__ == "__main__":
