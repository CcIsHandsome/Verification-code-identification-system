# -*-coding:utf-8 -*-
import cv2
import os


def cut_image(image, num, img_name):
    # image = cv2.imread('./img/8.jpg')
    im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # im_cut_real = im[8:47, 28:128]

    im_cut_1 = im[0:80, 5:55]
    im_cut_2 = im[0:80, 50:100]
    im_cut_3 = im[0:80, 100:150]
    im_cut_4 = im[0:80, 150:200]
    index = 0
    im_cut = [im_cut_1, im_cut_2, im_cut_3, im_cut_4]
    for i in range(4):
        im_temp = im_cut[i]
        cv2.imwrite('./progress-train3-picture2/' + img_name[i] + '_' + str(num) + '_' + str(index) + '.jpg', im_temp)
        index = index + 1


if __name__ == '__main__':
    img_dir = './progress-train3-picture/'
    label_dir = './mappings3.txt'
    img_name = os.listdir(img_dir)  # 列出文件夹下所有的目录与文件
    label_txt = []
    f = open(label_dir)
    for i in f:
        label_txt.append(i.split(",")[1].split("=")[0].strip())
    print(label_txt)
    f.close()
    for i in range(len(img_name)):
        path = os.path.join(img_dir, img_name[i])
        image = cv2.imread(path)
        print(image.shape)
        name_list = list(label_txt[i])[:5]
        cut_image(image, i, name_list)
        print("step:", i)

    print(u'*****图片分割预处理完成！*****')
