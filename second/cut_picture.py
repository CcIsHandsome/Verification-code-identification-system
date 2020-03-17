#-*-coding:utf-8 -*-
import cv2
import os

def cut_image(image, num, img_name):
    # image = cv2.imread('./img/8.jpg')
    im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # im_cut_real = im[8:47, 28:128]

    im_cut_1 = im[0:60, 0:44]
    im_cut_2 = im[0:60, 38:82]
    im_cut_3 = im[0:60, 80:124]
    im_cut_4 = im[0:60, 120:164]
    im_cut_5 = im[0:60, 156:200]
    index=0
    im_cut = [im_cut_1, im_cut_2, im_cut_3, im_cut_4, im_cut_5]
    for i in range(5):
        im_temp = im_cut[i]
        cv2.imwrite('./train2/progress-train2-picture2/'+img_name[i]+'_'+str(num)+'_'+str(index)+'.jpg', im_temp)
        index=index+1


if __name__ == '__main__':
    img_dir = './train2/progress-train2-picture/'
    label_dir='./train2/mappings2.txt'
    img_name = os.listdir(img_dir)  # 列出文件夹下所有的目录与文件
    label_txt=[]
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
        if i %2000==0:
            print('图片%s分割完成' % (i))

    print(u'*****图片分割预处理完成！*****')

