import cv2
import numpy as np
import os

if __name__ == '__main__':
    M_r = np.float32([[1, 0, 10], [0, 1, 0]])
    M_l = np.float32([[1, 0, -10], [0, 1, 0]])
    M_u = np.float32([[1, 0, 0], [0, 1, 10]])
    M_d = np.float32([[1, 0, 0], [0, 1, -10]])
    path = './train2/progress-train2-picture2/'
    #spath = './train2/translate_pic/'
    image_name = os.listdir(path)
    for i in range(len(image_name)):
        name1 = image_name[i].split('_')[0] + '_' + str(i) + '_' + 'r' + '.jpg'
        name2 = image_name[i].split('_')[0] + '_' + str(i) + '_' + 'l' + '.jpg'
        name3 = image_name[i].split('_')[0] + '_' + str(i) + '_' + 'u' + '.jpg'
        name4 = image_name[i].split('_')[0] + '_' + str(i) + '_' + 'd' + '.jpg'
        img_path = os.path.join(path, image_name[i])
        save_path1 = path + name1
        save_path2 = path + name2
        save_path3 = path + name3
        save_path4 = path + name4
        temp_img = cv2.imread(img_path)
        dst1 = cv2.warpAffine(temp_img, M_r, (44, 60), borderValue=(255, 255, 255))
        dst2 = cv2.warpAffine(temp_img, M_l, (44, 60), borderValue=(255, 255, 255))
        dst3 = cv2.warpAffine(temp_img, M_u, (44, 60), borderValue=(255, 255, 255))
        dst4 = cv2.warpAffine(temp_img, M_d, (44, 60), borderValue=(255, 255, 255))
        cv2.imwrite(save_path1, dst1)
        cv2.imwrite(save_path2, dst2)
        cv2.imwrite(save_path3, dst3)
        cv2.imwrite(save_path4, dst4)
        print("step:",i)
