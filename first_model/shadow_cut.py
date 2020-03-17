import cv2
import os
import numpy as np


def shadow_cut(img, labels, num):  # 投影切割算法
    label = []
    for i in (labels):
        temp = i
        if i =='*':
            temp = '='  # 这里因为文件保存不能有‘*’，所以用‘_’代替
        label.append(temp)
    print(label)
    result = []
    h, w = img.shape[:2]
    count = [0 for i in range(w)]
    for x in range(0, w):
        for y in range(0, h):
            # print((x, y))
            if img[y][x] == 0:  # cv2中的图像第一个维度是高，而不是宽
                count[x] += 1
    start = 0
    end = 0
    while (end < w):
        if (count[end] == 0):
            start = end
            end = end + 1
        else:
            while (end < w and count[end] != 0):
                end = end + 1
            if (end < w and end - start > 25):
                print((start, end))
                result.append(img[0:80, start:end])
                start = end
    # print(result)
    print(len(label), len(result))
    for idex, image in enumerate(result):
        image = cv2.resize(image, (50, 50), interpolation=cv2.INTER_CUBIC)
        save_name = './progress-train1-picture2/' + label[idex] + '_' + str(idex) + '_' + str(num) + '.jpg'
        cv2.imwrite(save_name, image)


if __name__ == '__main__':
    path = './progress-train1-picture/'
    img_names = os.listdir(path)
    label_names = []
    f = open('mappings1.txt')
    for i in f:
        label_names.append(i.split(",")[1].split("=")[0].strip())
    f.close()
    for idex, name in enumerate(img_names):
        img_path = os.path.join(path, name)
        img = cv2.imread(img_path)
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        label = label_names[idex]
        print("step:%d" % (idex))
        shadow_cut(img, label, idex)
