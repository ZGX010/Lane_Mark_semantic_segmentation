import cv2
import numpy as np
import time
from collections import namedtuple
import os
import glob
import os.path
import re
import tensorflow as tf

Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label.
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

labels = [
    #           name     id trainId      category  catId hasInstances ignoreInEval            color
    Label(     'void' ,   0 ,     0,        'void' ,   0 ,      False ,      False , (  0,   0,   0) ),
    Label(    's_w_d' , 200 ,     1 ,   'dividing' ,   1 ,      False ,      False , ( 70, 130, 180) ),
    Label(    's_y_d' , 204 ,     2 ,   'dividing' ,   1 ,      False ,      False , (220,  20,  60) ),
    Label(  'ds_w_dn' , 213 ,     3 ,   'dividing' ,   1 ,      False ,       True , (128,   0, 128) ),
    Label(  'ds_y_dn' , 209 ,     4 ,   'dividing' ,   1 ,      False ,      False , (255,   0,   0) ),
    Label(  'sb_w_do' , 206 ,     5 ,   'dividing' ,   1 ,      False ,       True , (  0,   0,  60) ),
    Label(  'sb_y_do' , 207 ,     6 ,   'dividing' ,   1 ,      False ,       True , (  0,  60, 100) ),
    Label(    'b_w_g' , 201 ,     7 ,    'guiding' ,   2 ,      False ,      False , (  0,   0, 142) ),
    Label(    'b_y_g' , 203 ,     8 ,    'guiding' ,   2 ,      False ,      False , (119,  11,  32) ),
    Label(   'db_w_g' , 211 ,     9 ,    'guiding' ,   2 ,      False ,       True , (244,  35, 232) ),
    Label(   'db_y_g' , 208 ,    10 ,    'guiding' ,   2 ,      False ,       True , (  0,   0, 160) ),
    Label(   'db_w_s' , 216 ,    11 ,   'stopping' ,   3 ,      False ,       True , (153, 153, 153) ),
    Label(    's_w_s' , 217 ,    12 ,   'stopping' ,   3 ,      False ,      False , (220, 220,   0) ),
    Label(   'ds_w_s' , 215 ,    13 ,   'stopping' ,   3 ,      False ,       True , (250, 170,  30) ),
    Label(    's_w_c' , 218 ,    14 ,    'chevron' ,   4 ,      False ,       True , (102, 102, 156) ),
    Label(    's_y_c' , 219 ,    15 ,    'chevron' ,   4 ,      False ,       True , (128,   0,   0) ),
    Label(    's_w_p' , 210 ,    16 ,    'parking' ,   5 ,      False ,      False , (128,  64, 128) ),
    Label(    's_n_p' , 232 ,    17 ,    'parking' ,   5 ,      False ,       True , (238, 232, 170) ),
    Label(   'c_wy_z' , 214 ,    18 ,      'zebra' ,   6 ,      False ,      False , (190, 153, 153) ),
    Label(    'a_w_u' , 202 ,    19 ,  'thru/turn' ,   7 ,      False ,       True , (  0,   0, 230) ),
    Label(    'a_w_t' , 220 ,    20 ,  'thru/turn' ,   7 ,      False ,      False , (128, 128,   0) ),
    Label(   'a_w_tl' , 221 ,    21 ,  'thru/turn' ,   7 ,      False ,      False , (128,  78, 160) ),
    Label(   'a_w_tr' , 222 ,    22 ,  'thru/turn' ,   7 ,      False ,      False , (150, 100, 100) ),
    Label(  'a_w_tlr' , 231 ,    23 ,  'thru/turn' ,   7 ,      False ,       True , (255, 165,   0) ),
    Label(    'a_w_l' , 224 ,    24 ,  'thru/turn' ,   7 ,      False ,      False , (180, 165, 180) ),
    Label(    'a_w_r' , 225 ,    25 ,  'thru/turn' ,   7 ,      False ,      False , (107, 142,  35) ),
    Label(   'a_w_lr' , 226 ,    26 ,  'thru/turn' ,   7 ,      False ,      False , (201, 255, 229) ),
    Label(   'a_n_lu' , 230 ,    27 ,  'thru/turn' ,   7 ,      False ,       True , (0,   191, 255) ),
    Label(   'a_w_tu' , 228 ,    28 ,  'thru/turn' ,   7 ,      False ,       True , ( 51, 255,  51) ),
    Label(    'a_w_m' , 229 ,    29 ,  'thru/turn' ,   7 ,      False ,       True , (250, 128, 114) ),
    Label(    'a_y_t' , 233 ,    30 ,  'thru/turn' ,   7 ,      False ,       True , (127, 255,   0) ),
    Label(   'b_n_sr' , 205 ,    31 ,  'reduction' ,   8 ,      False ,      False , (255, 128,   0) ),
    Label(  'd_wy_za' , 212 ,    32 ,  'attention' ,   9 ,      False ,       True , (  0, 255, 255) ),
    Label(  'r_wy_np' , 227 ,    33 , 'no parking' ,  10 ,      False ,      False , (178, 132, 190) ),
    Label( 'vom_wy_n' , 223 ,    34 ,     'others' ,  11 ,      False ,       True , (128, 128,  64) ),
    Label(   'om_n_n' , 250 ,    35 ,     'others' ,  11 ,      False ,      False , (102,   0, 204) ),
    Label(    'noise' , 249 ,   255 ,    'ignored' , 255 ,      False ,       True , (  0, 153, 153) ),
    Label(  'ignored' , 255 ,   255 ,    'ignored' , 255 ,      False ,       True , (255, 255, 255) ),
]

# print(labels[7].color) #(0, 0, 142)
#print(labels[37].trainId) #7
#print(len(labels))
# name to label object
name2label      = { label.name    : label for label in labels           }
# id to label object
id2label        = { label.id      : label for label in labels           }
# trainId to label object
trainId2label   = { label.trainId : label for label in reversed(labels) }
#color to label object
color2label      = { label.color    : label for label in labels           }
# category to list of label objects
category2labels = {}
for label in labels:
    category = label.category
    if category in category2labels:
        category2labels[category].append(label)
    else:
        category2labels[category] = [label]


# print(name2label['d_wy_za'].color) #(0, 255, 255)
#
# print(color2label[(178,132,190)].name) #r_wy_np
#
# print(trainId2label[1].color)
#
# print(len(trainId2label))
#
#
# print(labels[10].trainId)



def createLabelImage(colorimg, outline=None):

    # the size of the TrainID Labe image
    colorimage = cv2.imread(colorimg)                        #colorimg :the path to colorimg
    colorimage_shape = colorimage.shape                      #(2710, 3384, 3)
    colorimage = cv2.cvtColor(colorimage, cv2.COLOR_BGR2RGB) #change color spaces to rgb
    #size = ( colorimage_shape[1], colorimage_shape[0])      #size = ( annotation.imgWidth , annotation.imgHeight )

    labelImg = np.zeros((colorimage_shape[0],colorimage_shape[1],3),np.uint8)
    labelImg.fill(255)


    # loop over all class to one hot map
    semantic_map=[]
    #semantic_map = np.ones((3, colorimage_shape[0], colorimage_shape[1]), dtype=np.uint8)
    for class_dic in range(len(labels)) : #len(trainId2label) = 37
        equality = np.equal(colorimage, labels[class_dic].color)
        class_map = np.all(equality, axis = -1) #out put a shape[0]*shape[1] array ,and any pix be showe as Ture or False
        semantic_map.append(class_map) #out put a shape[0]*shape[1] array with the number is shape[2] ,and any pix be showa as Ture or False form 0 until the shape[2] be a class number

    labelImg = np.stack(semantic_map, axis=-1) #out put a shape[0]*shape[1]*shape[2] array and any pix be showa as Ture or False.
    labelImg = np.argmax(labelImg, axis=-1)

    #change the hot map to TrainID
    # for class_dic in range(len(labels)) :
    #     labelImg[labelImg == class_dic] = labels[class_dic].trainId # make the
    labelImg[labelImg >= 36] = 255

    # return the TrainID map
    return labelImg

startime = time.time()
labelImg = createLabelImage('/home/zgx010/TensorflowModels/models/research/deeplab/datasets/apollo/lane_segmentation/helpers/170927_063811892_Camera_5_bin.png'  )
cv2.imwrite('/home/zgx010/TensorflowModels/models/research/deeplab/datasets/apollo/lane_segmentation/helpers/170927_063811892_Camera_5_test.png',labelImg)
endtime = time.time()
print(endtime - startime)

# =========================================================================================================

# apolloPath = os.path.join(os.path.dirname(os.path.realpath(__file__)),'..','..')
# print(apolloPath)
# searchFine   = os.path.join( apolloPath , "labelimg","*","*","*","*","*","*Camera_*_bin.png")
# print(searchFine)
#
# # how to search for all ground truth
# # searchFine = os.path.join(cityscapesPath, "gtFine", "*", "*", "*_gt*_polygons.json")
# # searchCoarse = os.path.join(cityscapesPath, "gtCoarse", "*", "*", "*_gt*_polygons.json")
#
# # search files
# filesFine = glob.glob(searchFine)
# filesFine.sort()
# # filesCoarse = glob.glob(searchCoarse)
# # filesCoarse.sort()
#
# # concatenate fine and coarse
# # files = filesFine + filesCoarse
# files = filesFine # use this line if fine is enough for now.
#
# # quit if we did not find anything
# if not files:
#     printError("Did not find any files. Please consult the README.")
#
# for f in files:
#      # create the output filename
#     dst = f.replace( "_.jpg" , "_labelTrainIds.png" )
#
#     # do the conversion
#     try:
#         color2labelImg( f , dst , "trainIds" ) #old:jeson2labeImg
#     except:
#         print("Failed to convert: {}".format(f))
#         raise
#
#     # status
#     progress += 1
#     #print("\rProgress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
#     sys.stdout.flush()
# # a bit verbose
# print("Processing {} annotation files".format(len(files)))



#==========================================================================================

# FLAGS = tf.app.flags.FLAGS
#
# tf.app.flags.DEFINE_string('apollo_root',
#                            './apollo',
#                            'apollo dataset root folder.') #apollo root :./apollo
# print(FLAGS.apollo_root)
#
# tf.app.flags.DEFINE_string(
#     'output_dir',
#     './tfrecord',
#     'Path to save converted SSTable of TensorFlow examples.') #output_dir : ./tfrecord
#
#
# _NUM_SHARDS = 10
#
# # A map from data type to folder name that saves the data.
# _FOLDERS_MAP = {
#     'image': 'colorimg',
#     'label': 'labelimg',
# }
#
# # A map from data type to filename postfix.
# _POSTFIX_MAP = {
#     'image': '_Camera_*',  # old:_leftImg8bit filename:aachen_000000_000019_leftImg8bit.png  new: "_Camera_" filename:170927_063811892_Camera_5.jpg
#     'label': '_bin_labelTrainIds', # newfilename:170927_063811892_Camera_5_bin_labelTrainIds.png
# }
#
# # A map from data type to data format.
# _DATA_FORMAT_MAP = {
#     'image': 'jpg',
#     'label': 'png',
# }
#
# # Image file pattern.
# _IMAGE_FILENAME_RE = re.compile('(.+)' + _POSTFIX_MAP['image'])
#
#
# def _get_files(data, dataset_split):
#   """Gets files for the specified data type and dataset split.
#
#   Args:
#     data: String, desired data ('image' or 'label').
#     dataset_split: String, dataset split ('train', 'val', 'test')
#
#   Returns:
#     A list of sorted file names or None when getting label for
#       test set.
#   """
#   if data == 'label' and dataset_split == 'test':
#     return None
#   pattern = '*%s.%s' % (_POSTFIX_MAP[data], _DATA_FORMAT_MAP[data]) # _Camera_*.jpg
#   search_files = os.path.join(
#       FLAGS.apollo_root, _FOLDERS_MAP[data], dataset_split, '*','*','*','*', pattern)  # /apollo/colorimg/train/*
#   # print(search_files)
#   filenames = glob.glob(search_files)
#   #print(filenames)
#   return filenames
#   # sortedfilenames = sorted(filenames)
#   # return sorted(filenames)


