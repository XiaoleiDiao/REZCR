'''
predict.py有几个注意点
1、无法进行批量预测，如果想要批量预测，可以利用os.listdir()遍历文件夹，利用Image.open打开图片文件进行预测。
2、如果想要保存，利用r_image.save("img.jpg")即可保存。
3、如果想要获得框的坐标，可以进入detect_image函数，读取top,left,bottom,right这四个值。
4、如果想要截取下目标，可以利用获取到的top,left,bottom,right这四个值在原图上利用矩阵的方式进行截取。
'''
import pandas
import os
import cv2
import numpy as np
from skimage import morphology
from PIL import Image

from yolo_r import YOLO


def readFiles(tpath):
    txtLists = os.listdir(tpath)
    List = []
    for t in txtLists:
        t = tpath + "/" + t
        List.append(t)
    return List

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")

    else:
        print("---  There is this folder!  ---")




# Oracle_path = "D:\Learn Python\yolo3-pytorch-radical\oracle_img_779\img"
# store_path  = "D:\Learn Python\yolo3-pytorch-radical\oracle_img_779/results"

def DerectionRadical(PATH_TO_TEST_IMAGES_DIR):

    OraclePath = readFiles(PATH_TO_TEST_IMAGES_DIR)
    yolo = YOLO()

    df = pandas.DataFrame(columns=["Name","Y0","X0","Y1","X1","Confidence","Class"])

    error_list = []
    for op in OraclePath:
        name = op.split('/')[-1].split('.')[0]
        OracleImage = Image.open(op)
        # print(OracleImage)

        try:
            rect, r_image = yolo.detect_image(OracleImage)
            print(name, rect)

            for r in rect:
                new = pandas.DataFrame({'Name': name,
                                        'Y0': r[1],
                                        'X0': r[2],
                                        'Y1': r[3],
                                        'X1': r[4],
                                        'Confidence': str(rect[0]).split(' ')[1].split("'")[0],
                                        'Class':str(r[0]).split(' ')[0].split("'")[1]},index=[1])
                df = df.append(new,ignore_index=True)
            # pp = store_path + '/' + name + '.jpg'
            # r_image.save(pp)
        except:
            print("error")
            print(name)
            # error_list.append(name)

    # df.to_csv("D:\Learn Python\yolo3-pytorch-radical\oracle_img_779/rect_Oracles.csv")
    # print("error_list", error_list)

    return df


# yolo = YOLO()
# while True:
#     img = input('Input image filename:')
#     try:
#         image = Image.open(img)
#     except:
#         print('Open Error! Try again!')
#         continue
#     else:
#
#         r_image = yolo.detect_image(image)
#         r_image.show()
