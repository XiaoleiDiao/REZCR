# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 16:55:43 2021
@author: Xiaolei
"""


# Imports
import time
# start = time.time()
import numpy as np
import sys
# import tensorflow as tf
import cv2
from PIL import Image
import rdflib
import pandas as pd
import collections
from rdflib import Namespace
import os, os.path
import pandas
import os
import cv2
import numpy as np
from skimage import morphology
from PIL import Image
import collections
from yolo_r import YOLO
from predict_oracle import DerectionRadical


# Reasoning in KG

#更新df，同一位置的候选框只保留置信度最高的class
def UpdateOutputRadical(df):
    # print(df)
    point = []
    try:
        for index, row in df.iterrows():
            img_id, y1, x1, radical_acc, radical_id = row[0], row[1], row[2], row[5], row[6]

            for p in point:
                x = p[0]
                y = p[1]

                if float(x1)-1 <= float(x) <= float(x1)+1:
                    if float(y1)-1 <= float(y) <= float(y1)+1:
                        # print("delete current line!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        df = df.drop(index=index)
                        # print("new df", df)

            point.append((x1,y1))
    except:
        print("删除同一位置候选框的多个radical报错！")

    return df




# 对应class_id 和 部件名
def getRadicalName(excelradicalname):
    # excelradicalname = './component test/radical_id.xls'
    xls_radical = pd.read_excel(excelradicalname)
    dic_radical = collections.OrderedDict(zip(xls_radical.iloc[:, 0], xls_radical.iloc[:, 1]))
    print(dic_radical)
    return dic_radical




# 读取甲骨文-部件对应表
def gerCharacterRadical(excelcharactername):
    # excelcharactername = './excel_data/oracle_radical_779.xls'
    xls_character = pd.read_excel(excelcharactername)
    character_name = []
    for x in xls_character.iloc[:, 1]:
        x = str(x)
        character_name.append(x)
    dic_radical_num = collections.OrderedDict(zip(character_name, xls_character.iloc[:, 3]))
    print(dic_radical_num)
    dic_character_name = collections.OrderedDict(zip(character_name, xls_character.iloc[:, 2]))
    print(dic_character_name)
    dic_contain_radical = collections.OrderedDict(zip(character_name, xls_character.iloc[:, 5]))
    print(dic_contain_radical)
    return dic_radical_num, dic_character_name, dic_contain_radical




#读取知识图谱
def readKG(owl_path):
    # owl_path = "./component test/CR_KG.owl"
    g1 = rdflib.Graph()
    g1.parse(owl_path, format="xml")
    ns = Namespace('http://www.jlu.edu.cn/CR/ontology#')  # 命名空间

    return(g1, ns)




# find Single character
def findSingleCharacter(df):
    for index, row in df.iterrows():
        img_id, x1,y1,x2,y2, radical_acc, radical_id = row[0], row[1], row[2], row[3], row[4], row[5], row[6]
        print("radical_id",radical_id)
        radical_id = 'r_' + radical_id
        radical_name = dic_radical.get(radical_id)
        character_value = radical_name + "*1"
        character = list(dic_contain_radical.keys())[list(dic_contain_radical.values()).index(character_value)]
        print("character",character)

    return character


# 判断两个list是否完全一样
def equal_ignore_order(a, b):
    """ Use only when elements are neither hashable nor sortable! """
    unmatched = list(b)
    for element in a:
        try:
            unmatched.remove(element)
        except ValueError:
            return False
    return not unmatched



# find combination character
def findnRC(df):
    character = []     # 记录含有识别到的全部部件的 所有甲骨文字符
    character4 = []     #记录仅由当前部件组成的所有甲骨文
    character2 = []    # 记录有几个甲骨文由当前这些部件组成
    character3 = []
    # radical_acc = []
    radical = []       # 记录该图片内所有识别到的部件
    radical_test = []  # 统计所有候选甲骨文字符分别包含哪些部件
    character_can = list(dic_contain_radical.keys())
    # get_result = False
    character_can_2 = []   # 仅由当前部件组成的所有甲骨文汇总

    # print("character_can_ori", character_can)

    for index, row in df.iterrows():
        img_id, r_acc, radical_id = row[0], row[5], row[6]
        print('current line', img_id, r_acc, radical_id)

        # radical_acc.append(r_acc)
        radical_id = "r_" + radical_id
        ra_name = dic_radical.get(radical_id)
        radical.append(ra_name)
        print("radical", radical)


    # 在知识图谱中找到对应的character
    for i in range(len(radical)):
        print("i", i)
        for s in g1.subjects(ns['Contain'], ns[radical[i]]):
            c = s[34:]
            # print("c", c)
            character.append(c)
            # print("character", character)

        character_can = list(set(character_can).intersection(set(character)))
        character = []
    print("character_can", character_can)

    if len(character_can) == 1:
        # get_result = True
        print("character_can_final", character_can)
        return character_can

    # elif len(character_can) != 1:
    #     print("进入第二种情况的判断")
    #     for j in range(len(character_can)):
    #         # print("j", j)
    #         for o in g1.objects(ns[character_can[j]], ns['Contain']):
    #             # print("o", o)
    #             r = o[34:]
    #             radical_test.append(r)
    #
    #         # print("radical_test", sorted(radical_test))
    #         # print("radical now", sorted(radical))
    #
    #         # collections.Counter(radical_test) == collections.Counter(radical)
    #         if equal_ignore_order(radical_test, radical):
    #             print("找到正好对应的甲骨文了！！！")
    #             # get_result = True
    #             print("character41", character4)
    #             character4 = character_can[j]
    #             print("character42", character4)
    #
    #             # return character_can
    #             character_can_2.append(character4)
    #             print("character_can_2", character_can_2)
    #             character4 = []
    #         radical_test = []
    #     if len(character_can_2) == 1:
    #         print("character_can_2", character_can_2)
    #         return character_can_2


    else:
        print("进入第三种情况的判断")

        # print("detemine radical num")
        for k in range(len(character_can)):
            print("k", k)
            print("character_can[k]", character_can[k])
            # for r_num in g1.objects((ns[character_can[j]], ns['Radical_all_num'])):
            for s in g1.objects(ns[character_can[k]], ns['Radical_all_num']):
                r_num = s[:-2]
            # for s in g1.subjects(ns['Radical_all_num'], len(radical)):
                print("r_num",r_num)
                print("len(radical)",str(len(radical)))

                if r_num == str(len(radical)):
                    print("部件数量match！！！！！！！！！！！！！！")
                    character2.append(character_can[k])
                    print("character2", character2)


        if len(character2) == 1:
            print("只有一个甲骨文由当前这些部件组成")
            return character2

        else:

            # 统计当前radical的个数及组成
            print("统计当前radical的个数及组成")
            r_s = set()
            # print("s   what is this s:", r_s)
            # radical_out_num = ""
            pre = ""
            pre2 = ""
            radical_out_contain = []
            radical_contain_num2 = []

            print("radical", radical)
            for r_name in radical:  # 循环列表中的元素
                if r_name != pre:
                    if r_name not in r_s:  # 如果i不在集合中
                        r_s.add(i)  # 将i添加到集合中

                        radical_out_num = r_name + "*"+ str(radical.count(r_name))
                        radical_out_contain.append(radical_out_num)
                        print("radical_out_num", radical_out_num)
                        print("radical_out_contain", radical_out_contain)
                        pre = r_name

            # 读取character实际的部件组成及每个部件的个数
            print("character2", character2)
            for c1 in character2:
                print("c1", c1)
                radical_contain_num = dic_contain_radical.get(c1)
                radical_contain_num = radical_contain_num.split(",")
                # radical_contain_num2 = str(radical_contain_num2).split(",")
                print("radical_contain_num2", radical_contain_num)
                print("radical_out_contain", radical_out_contain)


                # for e in radical_out_contain:
                #     print("e", e)
                #     if e in radical_contain_num:
                #        character3.append(c1)

                if equal_ignore_order(radical_out_contain, radical_contain_num):
                    print("yes!!")
                    character3.append(c1)

            character3 = set(character3)
            print("character3", character3)

            return character3




def main():
    df = UpdateOutputRadical(data)
    radical_num = df.shape[0]
    # print(radical_num)
    character_out = []
    character_final = []

    if radical_num == 1:
        r_name = findSingleCharacter(df)
        c_name = r_name
        print("c_name", c_name)
        character_out.append(c_name)
        print("Single Character", character_out)

    elif radical_num > 1:
        # character_out = set(findnRC(df))
        character_out = findnRC(df)

        print("Combination character", character_out)


    for character in character_out:
        c_name = dic_character_name.get(character)
        character_final.append(c_name)
    print("character_final", character_final)

    return character_final



if __name__ == '__main__':
    # os.chdir('D:\\Learn Python\\yolo3-pytorch-radical\\')
    # sys.path.append("..")
    # PATH_TO_CKPT = './component test/frozen_inference_graph.pb'
    # PATH_TO_LABELS = os.path.join('train_component', 'my_label_map.pbtxt')
    PATH_TO_TEST_IMAGES_DIR = 'D:\Learn Python\yolo3-pytorch-radical\img'
    owl_path = "./kg/oracle_779_KG.owl"
    excelradicalname = './excel_data/59_radical_id.xls'
    excelcharactername = './excel_data/oracle_radical_779.xls'
    # detection_graph, category_index = loadModel(PATH_TO_CKPT, PATH_TO_LABELS)
    dic_radical = getRadicalName(excelradicalname)
    print("dic_radical", dic_radical)
    dic_radical_num, dic_character_name, dic_contain_radical = gerCharacterRadical(excelcharactername)
    print("dic_radical_num", dic_radical_num, "dic_character_name", dic_character_name, "dic_contain_radical", dic_contain_radical)
    g1, ns = readKG(owl_path)
    print("g1", g1, "ns", ns)
    data = DerectionRadical(PATH_TO_TEST_IMAGES_DIR)
    print("data", data)

    main()

    # end = time.time()
    # print("Execution Time: ", end - start)
