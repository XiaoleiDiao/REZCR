#-------------------------------------#
#       创建YOLO类
#-------------------------------------#

import colorsys
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

from nets.RIE import RIEBody
from utils.config import Config
from utils.utils import (DecodeBox, bbox_iou, letterbox_image,
                         non_max_suppression, yolo_correct_boxes)


def accuracy(output, labels, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path和classes_path参数的修改
#--------------------------------------------#
class RIE(object):
    _defaults = {
        "model_path"        : 'logs/Epoch81-Total_Loss32.6410-Val_Loss32.4129.pth',
        "Radical_classes_path"      : 'model_data/Radical_classes.txt',
        "SR_classes_path": 'model_data/SR_classes.txt',
        "model_image_size"  : (416, 416, 3),
        "confidence"        : 0.5,
        "iou"               : 0.3,
        "cuda"              : True,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.r_class_names = self._get_R_class()
        self.s_class_names = self._get_S_class()
        self.config = Config
        self.generate()
    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_R_class(self):
        r_classes_path = os.path.expanduser(self.Radical_classes_path)
        with open(r_classes_path) as f:
            r_classes = f.readlines()
        r_class_names = [c.strip() for c in r_classes]
        print("r_class_names", r_class_names)
        return r_class_names

    def _get_S_class(self):
        s_classes_path = os.path.expanduser(self.SR_classes_path)
        with open(s_classes_path) as f:
            s_classes = f.readlines()
        s_class_names = [c.strip() for c in s_classes]
        print("s_class_names", s_class_names)
        return s_class_names

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self):
        self.config["RIE"]["classes"] = len(self.r_class_names)
        #---------------------------------------------------#
        #   建立RIE模型
        #---------------------------------------------------#
        self.net = RIEBody(self.config)
        # print("self.net", self.net)

        #---------------------------------------------------#
        #   载入RIE模型的权重
        #---------------------------------------------------#
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)

        self.net.load_state_dict(state_dict)
        self.net = self.net.eval()

        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

        #---------------------------------------------------#
        #   建立三个特征层解码用的工具
        #---------------------------------------------------#
        print(self.model_image_size[1],self.model_image_size[0])
        self.RIE_decodes = DecodeBox(self.config["RIE"]["anchors"][0], self.config["RIE"]["classes"], (self.model_image_size[1], self.model_image_size[0]))


        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.r_class_names), 1., 1.)
                      for x in range(len(self.r_class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        # print("image_shape",image_shape)
        # print("image", np.shape(image))

        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        if self.letterbox_image:
            crop_img = np.array(letterbox_image(image, (self.model_image_size[1],self.model_image_size[0])))
            # print("crop_img11111111111", type(crop_img), np.shape(crop_img))
        else:
            crop_img = image.convert('RGB')
            # print("crop_img222222222222", np.shape(crop_img))
            crop_img = crop_img.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
            # print("crop_img333333333333", np.shape(crop_img))
        photo = np.array(crop_img,dtype = np.float32) / 255.0
        photo = np.transpose(photo, (2, 0, 1))
        # print("photo", np.shape(photo))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        images = [photo]

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(images))
            if self.cuda:
                images = images.cuda()

            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            # outputs = self.net(images)
            output_r, output_s = self.net(images)
            # print("output_r",output_r.shape)
            # print("output_s",output_s.shape, type(output_s))
            output_r = self.RIE_decodes(output_r)
            # print("output_r",output_r.shape)

            #---------------------------------------------------------#
            #  predict SR
            #---------------------------------------------------------#

            prob_s = torch.nn.functional.softmax(output_s, dim=1)
            # print("prob_s", prob_s.shape)
            # pred_s = torch.argmax(prob_s, dim=1)
            # print("pred_s", pred_s.shape)
            prob_s = prob_s.cpu().numpy().reshape(-1)
            # print("prob_s", type(prob_s), prob_s)

            SP = []
            for i, c in enumerate(prob_s):
                # print(i,c)
                sp = []
                class_s = self.s_class_names[i]
                conf_s = c
                # print("class_s", class_s)
                # print("cof_s", conf_s)
                sp.append(class_s)
                sp.append(conf_s)
                sp_t = tuple(sp)
                # dic_s = dict(zip(label_s, score_s))
                SP.append(sp_t)
            print("SP", SP)



            #---------------------------------------------------------#
            #   predict RPs, 将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            # output = torch.cat(output_list, 1)
            # print("output_r",output_r.shape)
            # print("self.config", self.config["RIE"]["classes"])
            # print("self.confidence",self.confidence)
            # print("self.iou",self.iou)
            # batch_detections = non_max_suppression(output_r, self.config["RIE"]["classes"],
            #                                         conf_thres=self.confidence,
            #                                         nms_thres=self.iou)
            batch_detections, prob_r = non_max_suppression(output_r, self.config["RIE"]["classes"],
                                                    conf_thres=self.confidence,
                                                    nms_thres=self.iou)
            # print("batch_detections", type(batch_detections), batch_detections)
            # print("prob_r", type(prob_r), prob_r.shape)
            prob_r = prob_r.cpu().numpy()
            # print("prob_r", type(prob_r), prob_r.shape)

            RPs = []
            for i, cr in enumerate(prob_r):
                # print(i,cr)
                RP = []
                for j, r in enumerate(cr):
                    rp = []
                    class_r = self.r_class_names[j]
                    conf_r = r
                    # print("class_r", class_r)
                    # print("cof_r", conf_r)
                    rp.append(class_r)
                    rp.append(conf_r)
                    # print("rp", rp)
                    rp_t = tuple(rp)
                    RP.append(rp_t)
                    # print("RP", RP)
                # dic_r = dict(zip(label_r, score_r))
                RPs.append(RP)
            print("RPs", RPs)


            #---------------------------------------------------------#
            #   如果没有检测出物体，返回原图
            #---------------------------------------------------------#
            try :
                batch_detections = batch_detections[0].cpu().numpy()
            except:
                return image

            #---------------------------------------------------------#
            #   对预测框进行得分筛选
            #---------------------------------------------------------#
            top_index = batch_detections[:,4] * batch_detections[:,5] > self.confidence
            top_conf = batch_detections[top_index,4]*batch_detections[top_index,5]
            top_label = np.array(batch_detections[top_index,-1],np.int32)
            top_bboxes = np.array(batch_detections[top_index,:4])
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

            # print("top_index",type(top_index), top_index.shape, top_index)
            # print("top_conf", type(top_conf), top_conf.shape, top_conf)
            # print("top_label", type(top_label), top_label.shape, top_label)
            # print("top_bboxes", type(top_bboxes), top_bboxes.shape, top_bboxes)
            # print("batch_detections[:,4]", type(batch_detections[:,4]), batch_detections[:,4].shape, batch_detections[:,4])
            # print("batch_detections[:,5]", type(batch_detections[:,5]), batch_detections[:,5].shape, batch_detections[:,5])


            #-----------------------------------------------------------------#
            #   在图像传入网络预测前会进行letterbox_image给图像周围添加灰条
            #   因此生成的top_bboxes是相对于有灰条的图像的
            #   我们需要对其进行修改，去除灰条的部分。
            #-----------------------------------------------------------------#
            if self.letterbox_image:
                boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)
            else:
                top_xmin = top_xmin / self.model_image_size[1] * image_shape[1]
                top_ymin = top_ymin / self.model_image_size[0] * image_shape[0]
                top_xmax = top_xmax / self.model_image_size[1] * image_shape[1]
                top_ymax = top_ymax / self.model_image_size[0] * image_shape[0]
                boxes = np.concatenate([top_ymin,top_xmin,top_ymax,top_xmax], axis=-1)
                
        # font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        font = ImageFont.truetype(font='model_data/simhei.ttf',size=11)

        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0], 1)
        Rect = []
        for i, c in enumerate(top_label):
            predicted_class = self.r_class_names[c]
            score = top_conf[i]

            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            # print(label, top, left, bottom, right)
            rect = [label, top, left, bottom, right]
            Rect.append(rect)
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[self.r_class_names.index(predicted_class)], width=2)
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.r_class_names.index(predicted_class)])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        # print("Rect",Rect)
        # print("image",image)
        # return Rect, image

        # new output: RPs & SP
        # print("RPs",RPs)
        # print("dic_s",dic_s)
        return RPs, SP


def readFiles(tpath):
    txtLists = os.listdir(tpath)
    List = []
    for t in txtLists:
        t = tpath + "/" + t
        List.append(t)
    return List


if __name__ == '__main__':
    # input=torch.randn(8,3,416,416)
    input_path = "F:\oracle\REZCR-test\img/oc_02_1_0153_1_6.png"
    # pics = os.listdir(input_path)
    # for pic in pics:
    image = Image.open(input_path)
    print("input image", np.shape(image))

    model = RIE()
    output = model.detect_image(image)
    # print(output)

    # model = CBR(inplanes=64, planes=[32,64])
    # model = RAB([1,2])
    # model = RAB([1, 2, 8, 8, 4])
    # out=model(input).cuda
    # print(out.shape)