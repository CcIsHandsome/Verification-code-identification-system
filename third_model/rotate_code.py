import cv2
from math import *
import os

width = 50
height = 80


# 旋转后的尺寸
def rotate(img, degree, name, index):
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))  # 这个公式参考之前内容
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2  # 因为旋转之后,坐标系原点是新图像的左上角,所以需要根据原图做转化
    matRotation[1, 2] += (heightNew - height) / 2

    imgRotation = cv2.warpAffine(img, matRotation, (width, height), borderValue=(255, 255, 255))
    img_name = name.split('_')[0] + '_' + str(index) + '_' + str(degree)
    save_path = 'C:\\GraduationProject\\train3\\progress-train3-picture2\\' + img_name + '.jpg'
    cv2.imwrite(save_path, imgRotation)


if __name__ == '__main__':
    img_name = os.listdir('./progress-train3-picture2')
    for i in range(len(img_name)):
        img = cv2.imread(os.path.join('./progress-train3-picture2', img_name[i]))
        rotate(img, -15, img_name[i], i)
        rotate(img, 15, img_name[i], i)
        print("step:",i)
