# Lane_mark_semantic-segmentation
<br>

## １ Project Function：
This project is the result of training the deeplab_V3 model on Baidu apollo road marking data. This project solves the problem of unbalanced training data by setting weights, and uses numpy and opencv to solve the problem that the annotation data cannot be directly used. In the project I provided a script that merges the visualization results into a video file. The project also provides the already packed TFRecord data and the weights set to resolve the imbalance of training data. <br>

The following GIF is the network prediction result
<br>

<div align=center><img width="400" height="220" src="https://github.com/ZGX010/Lane_Mark_semantic_segmentation/blob/master/doc/lane_mark.gif"/></div>
<br>

## 2 DeeplabV3 model and apollo-lane-mark-dataset
<br>

## 3 Operating Environment
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

## 4 Detecting the operating environment
Run the test command in the research folder and restart <br>
```Python
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```
Test mode_test for normal operation. <br>
```python
python deeplab/model_test.py
```
<br>

## 5 Processing training images
### 5.1 Convert color image to grayscale image
I put about 150GB of training data separately, you can get the training data in the way of ‘readme’ in the '/dataset/apollo folder'．<br>

The label_image data downloaded directly from apollo cannot be used as training data, although Baidu declares that these training data are created according to cityspace data. The label image uses RGB three channels to distinguish categories instead of grayscale images, so I provided a script ‘color2TrainId.py’ to convert RGB images into grayscale images as defined by the training ID in 'laneMarkDetection.py'. <br>

```python
# from /datasets/apollo/lane_segmentation/
python color2TrainIdLabelImgs.py
```
The default number of threads for multithreading in the 'color2TrainIdLabelImgs.py' is 10. You can modify the number of threads in the script according to your cpu. After the script finishes running, the annotation image for training will be generated under the labeimg folder.
<br>

### 5.2 Package data as TFRecord
The script 'build_apollo.py' is modified from 'build_cityscapes.py'. The script will read the images from the '/colorimg' and ‘/labelimg' folders and package them into TFRecord. It should be noted that only the image after the 'color2trainID.py' conversion can be packaged. <br>

```python
# from /datasets/apollo/
python build_apollo_data.py
```
<br>

## 6 Training model
### 6.1 Download the pre-training model from cityspaces
The cityspaecs dataset does not have road markings, but it is trained in a similar urban scene to the project, so I used the pre-training model provided by it to get better results. <br>
download link：download.tensorflow.org/models/deeplabv3_cityscapes_train_2018_02_06.tar.gz
### 6.2 Download the pre-training model provided by this project
If you need to see the results in a short period of time, you can use the pre-training model I provided to speed up model convergence.
<br>
Please download the pre-training model by following the method in 'readme' in the '/datasets/apollo/exp' folder.  
<br>
### 6.3 Unbalanced data settings
The calculation of loss is defined in the '/utils/train_utils.py' script. You can edit the weights according to your needs and set higher weights for more important objects.　
<br>
If you want to get an exact number by calculation, then you can refer to the calculation method in E-net, although that method only considers the distribution of the data and does not consider the difficulty of learning objects. <br>
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

### 6.4 Training network
> * If you want to start training from the beginning, then the learning rate needs to be set larger to ensure that the network parameters can be adjusted at a faster speed in the early stages of training. Since the pre-training model we used is similar to this project scenario, it is recommended to set the learning rate to be '.005'. 
<br>

> * The 'model_variant' parameter can also choose many other models.
<br>

> * Parameter ＇atrous_rates＇ sets the atrous convolution size. If you have larger GPU memory, you can set it to '8/16/32' and 'out_put_stride=8'. This will make the network get more receptive field. 
<br>

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
### 6.5 在提供的预训练模型上fur-training
> * 如果你下载了我提供的预训练模型，那么你需要将＇tf_initial_checkpoint＇的位置改为已经下载的模型的地址． <br>
> * 将base_learning＿rate设置为.001并将'training_bumber_of_steps'改为10000．
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
