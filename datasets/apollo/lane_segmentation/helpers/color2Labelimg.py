#!/usr/bin/python

# python imports
from __future__ import print_function, absolute_import, division
import os, sys, getopt
import cv2
import numpy as np
from PIL import Image
from PIL import ImageFile

from collections import namedtuple

sys.path.append('/home/zgx010/TensorflowModels/models/research/deeplab/datasets/apollo/lane_segmentation/')
# cityscapes imports
from laneMarkDetection import color2label
from ..helpers.laneMarkDetection import color2label


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



# Convert the given colorimg to a TrainID label image
def createLabelImage(colorimg):
    # the size of the TrainID Labe image
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.open(colorimg).convert("RGB").save(colorimg)

    #colorimage = cv2.imread(colorimg)  # colorimg :the path to colorimg
    colorimage = np.array(Image.open(colorimg))
    #colorimage = np.array(colorimage)

    colorimage_shape = colorimage.shape # (2710, 3384, 3)
    #colorimage = cv2.cvtColor(colorimage, cv2.COLOR_BGR2RGB)  # change color spaces to rgb

    labelImg = np.zeros((colorimage_shape[0], colorimage_shape[1], 1), np.uint8)
    labelImg.fill(255)

    # loop over all class to one hot map
    semantic_map = []
    #semantic_map = np.ones((3, colorimage_shape[0], colorimage_shape[1]), dtype=np.uint8)
    for class_dic in range(len(labels)):  # len(trainId2label) = 37
        equality = np.equal(colorimage, labels[class_dic].color)
        class_map = np.all(equality,axis=-1)  # out put a shape[0]*shape[1] array ,and any pix be showe as Ture or False
        #print(class_map)
        semantic_map.append(class_map)  # out put a shape[0]*shape[1] array with the number is shape[2] ,and any pix be showa as Ture or False form 0 until the shape[2] be a class number

    labelImg = np.stack(semantic_map, axis=-1)  # out put a shape[0]*shape[1]*shape[2] array and any pix be showa as Ture or False.
    labelImg = np.argmax(labelImg, axis=-1)

     # change the hot map to TrainID
    for class_dic in range(len(labels)):
         labelImg[labelImg == class_dic] = labels[class_dic].trainId  # make the class_dic to be the trainId

    # return the TrainID map

    return labelImg


def color2labelImg(colorimg,outImg):
    labelImg  = createLabelImage( colorimg )
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    cv2.imwrite(outImg, labelImg)
    Image.open(outImg).convert("L").save(outImg)



# The main method, if you execute this script directly
# Reads the command line arguments and calls the method 'json2labelImg'
def main(argv):
    trainIds = False
    try:
        opts, args = getopt.getopt(argv,"ht")
    except getopt.GetoptError:
        printError( 'Invalid arguments' )
    for opt, arg in opts:
        if opt == '-h':
            printHelp()
            sys.exit(0)
        elif opt == '-t':
            trainIds = True
        else:
            printError( "Handling of argument '{}' not implementend".format(opt) )

    if trainIds:
        color2labelImg( colorimg , outImg )
    else:
        json2labelImg( inJson , outImg )

# call the main method
if __name__ == "__main__":
    main(sys.argv[1:])