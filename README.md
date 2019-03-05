# Lane_mark_semantic-segmentation
<br>

## １ 项目功能：
本项目是deeplab模型在百度apollo路面标线数据上训练来的．解决了样本训练样本不均衡以及训练数据标注结果无法直接使用的问题．在本项目中我提供了一个支持多线程的彩色标注转灰度图的脚本，并提供了一个将可视化结果合并为视频文件到的脚本．同时还提供已经打包好的ＴＦＲecord数据以及为解决样本不均衡问题设置的类别权重作为参考．<br>

下面的ＧＩＦ为网络预测结果
<br>

<div align=center><img width="426" height="240" src="https://github.com/ZGX010/Lane_Mark_semantic_segmentation/blob/master/doc/lane_mark.gif"/></div>
<br>

## 2 DeeplabV3 model and apollo-lane-mark-dataset
<br>

## 3 运行环境
* tensorflow >=1.6
```python
pip install tensorflow-gpu
```
* python-pil python-numpy jupyter matplotlib
```python
pip install python-pil python-numpy jupyter matplotlib
```
* opencv3 for python2.7
```python
pip install opencv3
```
<br>

## 4 检测环境
在research文件夹下运行检测命令并重启
```Python
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
测试mode_test.py能否正常运行
```python
python deeplab/model_test.py
```
<br>

## 5 处理训练数据为ＴＦＲecord格式
### 5.1 彩色标注数据转换为灰度数据
由于训练数据大小约150ＧＢ超出上传尺寸．所以我将训练数据单独放置，你可以按照/dataset/apollo文件夹中的readme操作获得训练数据．
从apollo上下载的label数据无法直接作为训练数据，虽然对方申明自己是按照cityspace数据建立的．label图片采用了彩色数据而没有使用灰度图来表示图像像素的类别，因此我提供了一个转换脚本color2TrainId.py将彩色图片按＇/datasets/apollo/lane_segmentation/helpers/laneMarkDetection.py
＇中的定义转换为三通道一致的图片，像素数值与训练ＩＤ相对应 
<br>
```python
# from /datasets/apollo/lane_segmentation/
python color2TrainIdLabelImgs.py
```
color2TrainIdLabelImgs.py中使用了多线程默认线程数是10，你可以在脚本中修改线程数以适用于你的计算机．脚本运行结束后会在Ｌabeimg文件夹下生成训练用的标注数据．
<br>
### 5.2 将apollo数据打包为ＴＦＲecord
这个脚本修改自build_cityscapes.py文件,它将从colorimg/labelimg文件夹中读取数据并打包成ＴＦＲecord．需要注意的是只有经过上一步转换后的图像才能进行打包不然会出现错误．
```python
# from /datasets/apollo/
python build_apollo_data.py
```
<br>

## 6 训练模型
### 6.1 下载cityspaces的预训练模型
cityspaecs数据集虽然没有标线这个类别，但是它却是在城市场景中训练的模型，与本项目有一定的交集，因此我使用了它提供的预训练模型进行训练，并在这个基础上得到了较好的结果．
download.tensorflow.org/models/deeplabv3_cityscapes_train_2018_02_06.tar.gz
### 6.2 下载本项目提供的预训练模型
如果你的时间比较紧张，或者你需要在较短的时间内看到结果，你可以使用我提供的预训练模型，直接在这个基础上进行训练，相信这样模型会很快收敛．
<br>
请按照'/datasets/apollo/exp'文件中的readme获取下载链接. 
<br>
### 6.3 非均衡样本的设置
在'/utils/train_utils.py'脚本中对loss的计算进行了定义，你可以根据自己的需要对权重进行编辑，将比较重要的对象设置较高权重．　<br>
如果你希望能够通过计算得到一个确切的数字，那么你可以参考E-net中的计算方法，但是这种方法仅仅只考虑了数据的分布并没考虑对象间的差异． <br>
```python
scaled_labels = tf.reshape(scaled_labels, shape=[-1])
    loss_weight0 = 1.5
    loss_weight1 = 2.3
    loss_weight2 = 2.5
   ....
    loss_weight32 = 4
    loss_weight33 = 12
    loss_weight34 = 4
    loss_weight35 = 4
    loss_weight_ignore = 0

    not_ignore_mask =   tf.to_float(tf.equal(scaled_labels, 0)) * loss_weight0 + \
                        tf.to_float(tf.equal(scaled_labels, 1)) * loss_weight1 + \
                        ....
                        tf.to_float(tf.equal(scaled_labels, 35)) * loss_weight35 + \
                        tf.to_float(tf.equal(scaled_labels, ignore_label)) * loss_weight_ignore
```

### 6.4 从头开始进行训练
> * 从头开始训练时，你需要设置较大的学习率，来保证在训练初期网络参数能以较快的速度进行调整．虽然deeplab建议的学习率为.007但是考虑到我们使用的预训练模型与本项目有交叉所以我建议将学习率设置在.005以此来减少学习时间． <br>

> * 其中＇model_variant＇除了xception_65外还有许多其他的模型可以选择． <br>

> * ＇atrous_rates＇设定了空洞捐积大小，如果你有更大的ＧＰＵ内存可以设定为＇8/16/32＇,＇out_put_stride=8＇，这样做可以使网络获得更大的感受野. <br>

> * 'train_crop_size'需要设置为4的倍数加１，并且如果想要获得较好的结果，至少需要保证该参数在325以上． <br>
```python
CUDA_VISIBLE_DEVICES=0 \
python deeplab/train.py \
--logtostderr \
--num_clones=1 \
--task=0 \
--learing_policy=poly \
--base_learning_rate=.005 \
--learing_rate_decay_factor=0.1\
--learing_rate_decay_step=2000 \
--training_bumber_of_steps=200000 \
--train_spilt="train" \
--model_variant="xception_65" \
--atrous_rates=6 \
--atrous_rates=12 \
--atrous_rates=18 \
--output_stride=16 \
--decoder_output_stride=4 \
--train_crop_size=341 \
--train_crop_size=341 \
--train_bath_size=2 \
--dataset="apollo" \
--tf_initial_checkpoint='/home/zgx010/TensorflowModels/models/research/deeplab/backbone/deeplabv3_cityscapes_train/model.ckpt' \
--train_logdir='/home/zgx010/TensorflowModels/models/research/deeplab/datasets/apollo/exp/train_on_train_set/train_200000' \
--dataset_dir='/home/zgx010/TensorflowModels/models/research/deeplab/datasets/apollo/tfrecord'
```
<br>

> 训练程中可以使用tensorboard观察和对比． 
```python
#form /datasets/apollo/exp/train_on_train_set/
tensorboar --log_dir=./datasets/apollo/exp/train_on_train_set
```
<br>

### 6.５ 在提供的预训练模型上fur-training
如果你下载了我提供的预训练模型，那么你只需要将＇tf_initial_checkpoint＇的位置改为，已经下载的模型的地址．将base_learning＿rate设置为.001并将训练步数改小．
```python
CUDA_VISIBLE_DEVICES=0 \
python deeplab/train.py \
--logtostderr \
--num_clones=1 \
--task=0 \
--learing_policy=poly \
--base_learning_rate=.001 \
--learing_rate_decay_factor=0.1\
--learing_rate_decay_step=2000 \
--training_bumber_of_steps=10000 \
--train_spilt="train" \
--model_variant="xception_65" \
--atrous_rates=6 \
--atrous_rates=12 \
--atrous_rates=18 \
--output_stride=16 \
--decoder_output_stride=4 \
--train_crop_size=341 \
--train_crop_size=341 \
--train_bath_size=2 \
--dataset="apollo" \
--tf_initial_checkpoint='/home/zgx010/TensorflowModels/models/research/deeplab/backbone/deeplabv3_cityscapes_train/model.ckpt' \
--train_logdir='/home/zgx010/TensorflowModels/models/research/deeplab/datasets/apollo/exp/train_on_train_set/train_200000' \
--dataset_dir='/home/zgx010/TensorflowModels/models/research/deeplab/datasets/apollo/tfrecord'
```
<br>

## 7 训练结果可视化
```python
CUDA_VISIBLE_DEVICES=1 \
python deeplab/vis.py \
--logtostderr \
--vis_split="val" \
--model_variant="xception_65" \
--atrous_rates=6 \
--atrous_rates=12 \
--atrous_rates=18 \
--output_stride=16 \
--decoder_output_stride=4 \
--vis_crop_size=2710 \
--vis_crop_size=3384 \
--dataset="apollo" \
--colormap_type="apollo" \
--checkpoint_dir='/home/zgx010/TensorflowModels/models/research/deeplab/datasets/apollo/exp/train_on_train_set/train_200000' \
--vis_logdir='/home/zgx010/TensorflowModels/models/research/deeplab/datasets/apollo/exp/train_on_train_set/vis_train_200000' \
--dataset_dir='/home/zgx010/TensorflowModels/models/research/deeplab/datasets/apollo/tfrecord'
```
<br>
<br>

## 8 评估模型
```python
CUDA_VISIBLE_DEVICES=0 \
python deeplab/eval.py \
--logtostderr \
--eval_split="val" \
--model_variant="xception_65" \
--atrous_rates=6 \
--atrous_rates=12 \
--atrous_rates=18 \
--output_stride=16 \
--decoder_output_stride=4 \
--eval_crop_size=2710 \
--eval_crop_size=3384 \
--dataset="apollo" \
--checkpoint_dir='/home/zgx010/TensorflowModels/models/research/deeplab/datasets/apollo/exp/train_on_train_set/train_110000' \
--eval_logdir='/home/zgx010/TensorflowModels/models/research/deeplab/datasets/apollo/exp/train_on_train_set/eval_110000' \
--dataset_dir='/home/zgx010/TensorflowModels/models/research/deeplab/datasets/apollo/tfrecord'
```
<br>
<br>

## 9 导出模型
```python
python deeplab/export_model.py \
  --logtostderr \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --crop_size= \
  --crop_size= \
  --checkpoint_path='/home/zgx010/TensorflowModels/models/research/deeplab/datasets/apollo/exp/train_on_train_set/train_110000' \
  --export_dir='/home/zgx010/TensorflowModels/models/research/deeplab/datasets/apollo/exp/train_on_train_set/export_train_110000' \
```
