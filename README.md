# Lane_mark_semantic-segmentation
## 项目功能：
本项目是deeplab模型在百度apollo路面标线数据上训练来的．解决了样本训练样本不均衡以及训练数据标注结果无法直接使用的问题．在本项目中我提供了一个支持多线程的彩色标注转灰度图的脚本，并提供了一个将可视化结果合并为视频文件到的脚本．同时还提供已经打包好的ＴＦＲecord数据以及为解决样本不均衡问题设置的类别权重作为参考．
下面的ＧＩＦ为网络预测结果
<div align=center><img width="426" height="240" src="https://github.com/ZGX010/Lane_Mark_semantic_segmentation/blob/master/doc/lane_mark.gif"/></div>
<br>

## DeeplabV3 model and apollo-lane-mark-dataset
<br>

## 检测训练环境

## 处理训练数据为ＴＦＲecord格式
<br>

## 训练模型
<br>

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

## 训练结果可视化
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


## 评估模型
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

## 导出模型
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
