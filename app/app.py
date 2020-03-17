import sys
import forward1, test1, forward2, test2, forward3, test3
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import QWidget, QPushButton, QApplication, QToolTip, QLabel, QLineEdit, QGridLayout
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import os
import numpy as np
import copy


class Example(QWidget):

    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        QToolTip.setFont(QFont('SansSerif', 12))
        # self.setToolTip('This is a <b>QWidget</b> widget')
        self.setGeometry(300, 300, 650, 450)
        self.setWindowTitle('验证码识别app')

        self.step = 0

        self.pic_path = ''
        self.model_path = ''
        self.pic_path_pro = ''
        self.img = None
        self.img_name = None

        choose_model = QLabel('选择模型')
        choose_pic_label = QLabel('选择图片')
        show_process_pic_label = QLabel('处理图片')
        show_res_label = QLabel('预测结果')
        self.show_model_path = QLineEdit()
        self.show_model_path.setStyleSheet("border:2px solid black;")

        self.show_pic = QLabel('')
        self.show_pic.setStyleSheet("border:2px solid black;")

        self.show_pro_pic = QLabel('')
        self.show_pro_pic.setStyleSheet("border:2px solid black;")

        self.show_res_text = QLineEdit()
        self.show_res_text.setStyleSheet("border:2px solid black;")
        grid = QGridLayout()
        grid.setSpacing(30)

        grid.addWidget(choose_model, 1, 0)
        grid.addWidget(self.show_model_path, 1, 1)
        grid.addWidget(choose_pic_label, 2, 0)
        grid.addWidget(self.show_pic, 2, 1)
        grid.addWidget(show_process_pic_label, 3, 0)
        grid.addWidget(self.show_pro_pic, 3, 1)
        grid.addWidget(show_res_label, 4, 0)
        grid.addWidget(self.show_res_text, 4, 1)

        self.choose_model_btn = QPushButton('打开', self)
        self.choose_model_btn.setObjectName("pushButton")
        self.choose_model_btn.clicked.connect(self.openfile_model)
        grid.addWidget(self.choose_model_btn, 1, 2)

        self.choose_pic_btn = QPushButton('打开', self)
        self.choose_pic_btn.setObjectName("pushButton")
        self.choose_pic_btn.clicked.connect(self.openfile_pic)
        grid.addWidget(self.choose_pic_btn, 2, 2)

        self.choose_pic_btn = QPushButton('处理', self)
        self.choose_pic_btn.setObjectName("pushButton")
        self.choose_pic_btn.clicked.connect(self.process_pic)
        self.choose_pic_btn.clicked.connect(self.show_pic_after_pro)
        grid.addWidget(self.choose_pic_btn, 3, 2)

        self.show_res_btn = QPushButton('预测', self)
        self.choose_pic_btn.setObjectName("pushButton")
        self.choose_pic_btn.clicked.connect(self.show_pre_res)
        grid.addWidget(self.show_res_btn, 4, 2)

        self.setLayout(grid)
        self.show()

    def openfile_pic(self):
        openfile_name, _ = QFileDialog.getOpenFileName(self, '选择文件', 'e:/pythoncode/game_1/',
                                                       'Image files(*.jpg , *.jpeg , *.png)')
        self.pic_path = openfile_name
        self.img = cv2.imread(self.pic_path)
        self.img_name = str(self.pic_path).split('/')[-1]
        self.show_pic.setPixmap(QPixmap(openfile_name))

    def show_pic_after_pro(self):
        self.show_pro_pic.setPixmap(QPixmap(self.pic_path_pro))

    def show_pre_res(self):
        res = ''
        if 'train1' in self.model_path:
            res = test1.test(self.pic_path_pro, self.model_path)
        elif 'train2' in self.model_path:
            res = test2.test(self.pic_path_pro, self.model_path)
        elif 'train3' in self.model_path:
            res = test3.test(self.pic_path_pro, self.model_path)
        self.show_res_text.setText(res)

    def del_noise(self, img, number):
        height = img.shape[0]
        width = img.shape[1]

        img_new = copy.deepcopy(img)
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                point = [[], [], []]
                count = 0
                point[0].append(img[i - 1][j - 1])
                point[0].append(img[i - 1][j])
                point[0].append(img[i - 1][j + 1])
                point[1].append(img[i][j - 1])
                point[1].append(img[i][j])
                point[1].append(img[i][j + 1])
                point[2].append(img[i + 1][j - 1])
                point[2].append(img[i + 1][j])
                point[2].append(img[i + 1][j + 1])
                for k in range(3):
                    for z in range(3):
                        if point[k][z] == 0:
                            count += 1
                if count <= number:
                    img_new[i, j] = 255
        return img_new

    def process_pic(self):
        img = self.img
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 二值化
        result = cv2.adaptiveThreshold(grayImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21,
                                       1)  # 自适应阈值的选取
        # 去噪声
        img = self.del_noise(result, 6)
        img = self.del_noise(img, 4)
        img = self.del_noise(img, 3)
        # 加滤波去噪
        im_temp = cv2.bilateralFilter(src=img, d=15, sigmaColor=130, sigmaSpace=150)
        im_temp = im_temp[1:-1, 1:-1]
        im_temp = cv2.copyMakeBorder(im_temp, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[255])
        cv2.imwrite('./example_after_process/%s' % (self.img_name), im_temp)
        self.pic_path_pro = './example_after_process/' + str(self.img_name)

    def openfile_model(self):
        openfile_name, _ = QFileDialog.getOpenFileName(self, '选择文件', 'e:/pythoncode/game_1/', 'Image files(*.meta)')
        temp = str(openfile_name).split('/')
        path = ''
        for c in range(len(temp) - 1):
            path += temp[c]
            path += '/'
        self.model_path = path
        self.show_model_path.setText(path)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
