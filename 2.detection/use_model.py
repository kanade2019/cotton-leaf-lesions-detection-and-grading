# -*- coding: utf-8 -*-
import numpy as np
import cv2
from PIL import Image, ImageDraw

import hparams_config
import inference
import utils
import sys
import os

class ModelClass:
    def __init__(self) :
        # 初始化
        model_name='efficientdet-d2'
        saved_model_dir=r'F:\cotton\automl-master\efficientdet\model_save\efficientdet-d1-pb'
        ckpt_path = 'F:/code/cotton/final/QtCottonProject/x64/Release/model_save/efficientdet-d2-finetune/archive'
        hparams='F:/code/cotton/final/QtCottonProject/x64/Release/voc_config.yaml'
        
        model_config = hparams_config.get_detection_config(model_name)
        model_config.override(hparams)  # Add custom overrides
        model_config.is_training_bn = False
        model_config.image_size = utils.parse_image_size(model_config.image_size)
        
        
        # batch_size = 1
        # labels_shape = [batch_size, model_config.num_classes]
        height, width = model_config.image_size
        
        config_dict = {'line_thickness' : 5, 'max_boxes_to_draw' : 100, 'min_score_thresh' : 0.4}
        
        self.driver = inference.ServingDriver(model_name, ckpt_path, 
                                         batch_size=1, use_xla=False,
                                         model_params=model_config.as_dict(),
                                         **config_dict)
        self.driver.load(saved_model_dir)

    def detection(self, image):
        # 读取图像并分别进行检测
        # image = Image.open(image_path).rotate(180)
        # image.show()
        if __name__ != "__main__":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        # print("get in pycode", image.size)
        # image = arrayreset(image)
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        one_size = image.size[1] // 6
        self.one_size = one_size
        self.shift_size = one_size // 2
        # print(one_size)
        # image = image.resize((8*one_size, 6*one_size))
        cutImages = []
        res = []
        for col in range(16):
            # print(1)
            for row in range(12):
                tempImage = image.crop((self.shift_size*col, self.shift_size*row, self.shift_size*col+one_size, self.shift_size*row+one_size))
                cutImages.append(np.array(tempImage))
                detections = self.driver.serve_images([np.array(tempImage)])
                res.append(detections)
                # output_images.append(driver.visualize(np.array(tempImage), detections[0], **config_dict))
                # tempImage.show()
                # print(detections[0][:, 5])
        '''
        # 拼接图像
        dst = Image.new('RGB', (one_size*8, one_size*6))
        for col in range(8):
            for row in range(6):
                print(row+col*6)
                dst.paste(Image.fromarray(output_images[row+col*6]), (one_size*col, one_size*row))
                
        # 保存
        dst.save('test.png')
        '''

        # cv2.imshow("test", image)
        # cv2.waitKey(0)
        
        # 返回box 和 score
        self.res = res
        self.get_num()
        # print(self.res)

        # 绘制矩形框 检测结果
        # draw = ImageDraw.Draw(image)
        # for xyxys in self.res:
        #     draw.rectangle((xyxys[1], xyxys[0], xyxys[3], xyxys[2]), outline='red', width=5)
        # image.show()

        return self.res

    def get_num(self):
        tmp = []
        for col in range(16):
            for row in range(12):
                det = self.res[row + col*12]
                for i in det[0]:
                    if i[5] >= 0.5:
                        yxyxs = i[1:6]
                        yxyxs[0] += row*self.shift_size
                        yxyxs[2] += row*self.shift_size
                        yxyxs[1] += col*self.shift_size
                        yxyxs[3] += col*self.shift_size
                        # print(yxyxs)
                        tmp.append(yxyxs.tolist())

        num = len(tmp)
        self.res = tmp
        self.num = num

    def get_x1(self):
        return self.res[0][0]

    def get_y1(self):
        return self.res[0][1]

    def get_x2(self):
        return self.res[0][2]

    def get_y2(self):
        return self.res[0][3]

    def get_score(self):
        return self.res[0][4]

    def delete_first(self):
        self.res = self.res[1:]

    def testString(self, test):
        print(test)

    def testNone(self):
        print('success')

if __name__ == "__main__":
    a = ModelClass()
    image = Image.open(r'D:/study/lib/棉花/数据/20200704/5m80%80%/RGB/IMG_200704_024640_0001_RGB.JPG').rotate(180)
    # image.show()
    # np.array(image)
    a.detection(image)
    for i in range(a.num):
        print(a.get_x1(), a.get_x2(), a.get_y1(), a.get_y2(), a.get_score())
        a.delete_first()
