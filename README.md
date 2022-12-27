# 零.写在最前

该项目源于《[飞桨学习赛：钢铁缺陷检测挑战赛](https://aistudio.baidu.com/aistudio/competition/detail/114/0/introduction)》

基于飞桨目标检测开发套件PaddleDetection进行快速微调和预测，生成比赛结果文件



# 一.安装PaddleDetection

PaddleDetection为基于飞桨PaddlePaddle的端到端目标检测套件，内置30+模型算法及250+预训练模型，覆盖目标检测、实例分割、跟踪、关键点检测等方向，其中包括服务器端和移动端高精度、轻量级产业级SOTA模型、冠军方案和学术前沿算法，并提供配置化的网络模块组件、十余种数据增强策略和损失函数等高阶优化支持和多种部署方案，在打通数据处理、模型开发、训练、压缩、部署全流程的基础上，提供丰富的案例及教程，加速算法产业落地应用。

![](https://ai-studio-static-online.cdn.bcebos.com/2e768bbebe3743a7943a9dc216caf0cc55b30a49940f4f638e7768a4aa6271dc)


# 二.数据预处理

将数据集生成为符合PaddleDetection的格式,这里是VOC格式

格式如下：

![](https://ai-studio-static-online.cdn.bcebos.com/4f2d65643bfa4ef9a60a98b3c029fef2f6651fd0019e4f8eb195fa14a0bb7ac1)



```python
import os
# 切换到work目录
%cd /home/aistudio/work/

if not os.path.exists("PaddleDetection"):
    # 解压PadddleDetection
    !unzip -o PaddleDetection.zip
    # 安装其他依赖
    %cd PaddleDetection
    !pip install -r requirements.txt
```


```python
# 切换到数据集目录
%cd /home/aistudio/data/data169999

!rm -rf train test

# 解压数据集
!unzip -o train.zip
!unzip -o test.zip
```


```python
# 查看目录
! ls -all
```

    总用量 27654
    drwxrwxrwx 1 aistudio aistudio     4096 12月 14 13:34 .
    drwxrwxrwx 1 aistudio aistudio     4096 12月 14 13:31 ..
    drwxrwxr-x 4 aistudio aistudio     4096 12月 14 13:34 __MACOSX
    drwxr-xr-x 4 aistudio aistudio     4096 8月  26  2021 test
    -rwxrwxrwx 1 aistudio aistudio  6055677 12月 14 13:32 test.zip
    drwxr-xr-x 4 aistudio aistudio     4096 7月  23  2021 train
    -rwxrwxrwx 1 aistudio aistudio 22240365 12月 14 13:32 train.zip



```python
! ls -all train
! ls -all test

! mv train/ANNOTATIONS/ train/annotations
! mv train/IMAGES/ train/images

! mv test/IMAGES/ test/images

! ls -all train
! ls -all test
```


```python
import random
class Pre(object):
    def __init__(self):
        self.categorys = {
            'crazing':0,
            'inclusion':1,
            'pitted_surface':2,
            'scratches':3,
            'patches':4,
            'rolled-in_scale':5
        }
        self.trainDataRatio = 0.9

    def run(self):
        # 生成label_list.txt
        labelFile = open('train/label_list.txt', 'w')
        for key, value in self.categorys.items():
            labelFile.write(key)
            labelFile.write('\n')

        # 生成train.txt和valid.txt
        trainFile = open('train/train.txt', 'w')
        vaildFile = open('train/valid.txt', 'w')
        trainNums = int(1400*self.trainDataRatio)
        alls = range(0,1400)
        trainIndexs = random.sample(alls, trainNums)
        vaildIndexs = ret = list(set(alls) ^ set(trainIndexs))

        for k in trainIndexs:
            trainFile.write('./images/'+str(k)+'.jpg ./annotations/'+str(k)+'.xml')
            trainFile.write('\n')

        for k in vaildIndexs:
            vaildFile.write('./images/'+str(k)+'.jpg ./annotations/'+str(k)+'.xml')
            vaildFile.write('\n')


preObj = Pre()
preObj.run()     
```


```python
# 拷贝数据集到 PadddleDetection 对应目录下
!mkdir -p /home/aistudio/work/PaddleDetection/dataset/mine_voc/
!cp -r train/* /home/aistudio/work/PaddleDetection/dataset/mine_voc/
!cp -r test /home/aistudio/work/PaddleDetection/dataset/mine_voc/
```

# 三.训练

训练需要配置文件。

1.入口配置文件:

/work/PaddleDetection/configs/ppyoloe/ppyoloe_plus_crn_x_80e_coco.yml

```
_BASE_: [
  '../datasets/mine_voc.yml',
  '../runtime.yml',
  './_base_/optimizer_80e.yml',
  './_base_/ppyoloe_plus_crn.yml',
  './_base_/ppyoloe_plus_reader.yml',
]

log_iter: 5
snapshot_epoch: 1
weights: output/ppyoloe_plus_crn_x_80e_coco/model_final

pretrain_weights: https://bj.bcebos.com/v1/paddledet/models/pretrained/ppyoloe_crn_x_obj365_pretrained.pdparams
```

  
mine_voc.yml 主要说明了训练数据和验证数据的路径

runtime.yml 主要说明了公共的运行参数，比如说是否使用GPU、每多少个epoch存储checkpoint等

optimizer_80e.yml 主要说明了学习率和优化器的配置。

ppyoloe_plus_crn.yml 主要说明模型、和主干网络的情况。

ppyoloe_plus_reader.yml 主要说明数据读取器配置，如batch size，并发加载子进程数等，同时包含读取后预处理操作，如resize、数据增强等等



```python
# 单卡GPU 训练
%cd /home/aistudio/work/PaddleDetection/

!export CUDA_VISIBLE_DEVICES=0

# 训练，--eval 边训练边评估
!python tools/train.py \
        -c configs/ppyoloe/ppyoloe_plus_crn_x_80e_coco.yml \
        --use_vdl=true \
        --vdl_log_dir=vdl_dir/scalar \
        --eval


# [11/09 20:15:02] ppdet.engine INFO: Epoch: [53] [  0/157] learning_rate: 0.000043 loss: 1.640873 loss_cls: 0.768857 loss_iou: 0.159550 loss_dfl: 0.975133 loss_l1: 0.374083 eta: 0:33:40 batch_cost: 0.4564 data_cost: 0.0005 ips: 17.5284 images/s
# [11/09 20:15:14] ppdet.engine INFO: Epoch: [53] [ 20/157] learning_rate: 0.000042 loss: 1.623144 loss_cls: 0.784792 loss_iou: 0.148097 loss_dfl: 0.928553 loss_l1: 0.350931 eta: 0:33:30 batch_cost: 0.4790 data_cost: 0.0003 ips: 16.7022 images/s
# [11/09 20:15:27] ppdet.engine INFO: Epoch: [53] [ 40/157] learning_rate: 0.000042 loss: 1.667645 loss_cls: 0.778825 loss_iou: 0.157998 loss_dfl: 0.953017 loss_l1: 0.396211 eta: 0:33:21 batch_cost: 0.4920 data_cost: 0.0003 ips: 16.2617 images/s
# [11/09 20:15:39] ppdet.engine INFO: Epoch: [53] [ 60/157] learning_rate: 0.000042 loss: 1.702080 loss_cls: 0.819409 loss_iou: 0.166717 loss_dfl: 0.961283 loss_l1: 0.377798 eta: 0:33:11 batch_cost: 0.4605 data_cost: 0.0003 ips: 17.3721 images/s
# [11/09 20:15:50] ppdet.engine INFO: Epoch: [53] [ 80/157] learning_rate: 0.000042 loss: 1.685796 loss_cls: 0.820941 loss_iou: 0.159150 loss_dfl: 0.994252 loss_l1: 0.389399 eta: 0:33:01 batch_cost: 0.4379 data_cost: 0.0003 ips: 18.2674 images/s
# [11/09 20:16:02] ppdet.engine INFO: Epoch: [53] [100/157] learning_rate: 0.000042 loss: 1.617593 loss_cls: 0.744304 loss_iou: 0.155046 loss_dfl: 0.947161 loss_l1: 0.374160 eta: 0:32:52 batch_cost: 0.4626 data_cost: 0.0003 ips: 17.2947 images/s
# [11/09 20:16:14] ppdet.engine INFO: Epoch: [53] [120/157] learning_rate: 0.000041 loss: 1.675176 loss_cls: 0.807137 loss_iou: 0.160434 loss_dfl: 0.964645 loss_l1: 0.397004 eta: 0:32:42 batch_cost: 0.4787 data_cost: 0.0003 ips: 16.7128 images/s
# [11/09 20:16:26] ppdet.engine INFO: Epoch: [53] [140/157] learning_rate: 0.000041 loss: 1.648713 loss_cls: 0.788865 loss_iou: 0.151504 loss_dfl: 0.928811 loss_l1: 0.361151 eta: 0:32:33 batch_cost: 0.4757 data_cost: 0.0003 ips: 16.8159 images/s
# [11/09 20:16:41] ppdet.utils.checkpoint INFO: Save checkpoint: output/ppyoloe_plus_crn_x_80e_coco
# [11/09 20:16:41] ppdet.engine INFO: Eval iter: 0
# [11/09 20:16:54] ppdet.metrics.metrics INFO: Accumulating evaluatation results...
# [11/09 20:16:54] ppdet.metrics.metrics INFO: mAP(0.50, integral) = 79.64%
# [11/09 20:16:54] ppdet.engine INFO: Total sample number: 140, averge FPS: 10.8917303840318
# [11/09 20:16:54] ppdet.engine INFO: Best test bbox ap is 0.796.
# [11/09 20:17:06] ppdet.utils.checkpoint INFO: Save checkpoint: output/ppyoloe_plus_crn_x_80e_coco
```

# 四.评估模型

可以在output/ppyoloe_plus_crn_x_80e_coco/目录下指定某个epoch的参数，或者直接用最终的参数,或者指定最优权重




```python
%cd /home/aistudio/work/PaddleDetection/
!python tools/eval.py \
        -c configs/ppyoloe/ppyoloe_plus_crn_x_80e_coco.yml \
        -o weights=/home/aistudio/work/PaddleDetection/output/ppyoloe_plus_crn_x_80e_coco/best_model.pdparams
```

# 五.预测

--infer_dir 表示预测的图片文件夹
--output_dir 表示输出的图片文件夹
--draw_threshold 可视化阀值
-o weights 权重路径
--use_vdl 可视化
--save_results 保存结果，这里设置为True


```python
%cd /home/aistudio/work/PaddleDetection/
!python tools/infer.py -c configs/ppyoloe/ppyoloe_plus_crn_x_80e_coco.yml \
                    --infer_dir=dataset/mine_voc/test/images \
                    --output_dir=mine_output/ \
                    --draw_threshold=0.1 \
                    -o weights=output/ppyoloe_plus_crn_x_80e_coco/best_model.pdparams \
                    --use_vdl=True \
					--save_results=True
```

结果保存在mine_output目录下

特别注意。在aistudio 平台上运行infer.py，得到的测试集图片列表不一定是按照数字大小排序，所以修改了infer.py源代码，修改部分如下图

![](https://ai-studio-static-online.cdn.bcebos.com/932309aabd2645c792b74ca778b240c72ea885a2403d4d69b618fd2eb896b04f)


# 六.生成比赛结果


```python
import glob
import os
import json
import pandas as pd
class Result(object):
    def __init__(self):
        self.imagesPath = '/home/aistudio/work/PaddleDetection/mine_output/'
        self.bboxPath = '/home/aistudio/work/PaddleDetection/mine_output/bbox.json'
        self.submissionPath = '/home/aistudio/work/submission.csv'

    def run(self):
        images = self.get_image_ids()
        bbox = self.get_bbox()
        results = []
        for i in range(len(images)):
            image_id = images[i]
            for j in range(len(bbox['bbox'][i])):
                bbox_  = [int(i) for i in bbox['bbox'][i][j]]
                item = [
                    image_id,
                    bbox_,
                    int(bbox['label'][i][j]),
                    round(bbox['score'][i][j],2)
                    ]
                results.append(item)
        
        submit = pd.DataFrame(results, columns=['image_id', 'bbox','category_id','confidence'])
        submit[['image_id', 'bbox','category_id','confidence']].to_csv(self.submissionPath, index=False)

    def get_image_ids(self):
        images = set()
        exts = ['jpg']
        for ext in exts:
            images.update(glob.glob('{}/*.{}'.format(self.imagesPath, ext)))
        images = list(images)
        lists = []
        for item in images:
            item = item.replace(self.imagesPath,'')
            lists.append(item)
        lists.sort(key=lambda x:int(x.split('.')[0]))
        ids = []
        for fname in lists:
            ids.append(fname.replace('.jpg',''))
        return ids

    def get_bbox(self):
        with open(self.bboxPath, 'r', encoding='utf-8') as bbox:
            bbox = json.load(bbox)
        return bbox

resultObj = Result()
resultObj.run()
```

最终生成的结果保存在work/submission.csv 下，将其压缩后进行提交为11月榜5
![](https://ai-studio-static-online.cdn.bcebos.com/a1e0dda85961440b91af89738eb2e1c4c271a3c166194b2a963fef4b6f0499c4)



# 7.总结

通过PaddleDetection可以快速实现检测目标任务

优化方向

1.本项目用的是ppyoloe_x_80e模型, 可以尝试换下学术界霸榜的transformer系列模型

2.诸如学习率，数据增强等参数还有调优的可能
