#!/usr/bin/python
#
# Converts the polygonal annotations of the Cityscapes dataset
# to images, where pixel values encode ground truth classes.
#
# The Cityscapes downloads already include such images
#   a) *color.png             : the class is encoded by its color
#
# With this tool, you can generate option
#   b) *labelTrainIds.png     : the class is encoded by its training ID
# This encoding might come handy for training purposes. You can use
# the file labes.py to define the training IDs that suit your needs.
# Note however, that once you submit or evaluate results, the regular
# IDs are needed.
#
# Uses the converter tool in 'color2labelImg.py'
# Uses the mapping defined in 'labels.py'
#

# python imports
from __future__ import print_function, absolute_import, division
import os, glob, sys
import time
import threading
from color2Labelimg import color2labelImg

class myThread (threading.Thread):
    def __init__(self, threadID, name ,counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.files = name
        self.counter = counter
    def run(self):
        print("Starting "+ self.name)
        cotoTrainid(self.counter)
        print("Exiting " + self.name)


def cotoTrainid(files):
    progress = 0
    for f in files:
        # create the output filename
        dst = f.replace( ".png" , "_labelTrainIds.png" )
        # print(dst)

        # do the conversion
        try:
            color2labelImg( f , dst ) #old:jeson2labeImg
        except:
            print("Failed to convert: {}".format(f))
            raise

        # status
        progress += 1
        print("\rProgress: {:>3} %".format( progress * 100 / len(files) ), end=' ')
        sys.stdout.flush()


# The main method
def main():
    # Where to look for Apollo
    #startime = time.time()
    if 'APOLLO_DATASET' in os.environ:
        apolloPath = os.environ['APOLLO_DATASET']
    else:
        apolloPath = os.path.join(os.path.dirname(os.path.realpath(__file__))) # /home/zgx010/TensorflowModels/models/research/deeplab/datasets/apollo/lane_segmentation/helpers
        #print(apolloPath)

    # how to search for all ground truth
    searchFine   = os.path.join( apolloPath , "test"   , "*" , "*" , "*Camera_*_bin.png" ) #/deeplab/dataset/cityscapes/gtfine/train/bochum/bochum_00000_000313_gtfine_polygons.json


    # search files
    filesFine = glob.glob( searchFine )
    filesFine.sort()

    # concatenate fine and coarse
    files = filesFine # use this line if fine is enough for now.

    # quit if we did not find anything
    if not files:
        print( "Did not find any files. Please consult the README." )

    # a bit verbose
    print("Processing {} labelimg files".format(len(files)))
    # print(files)

    #device the files to four list
    j = 10                                    # set thread number in there
    r = len(files) % j                        # 528%10= 8
    dclass = (len(files) - r) / j             # (520)/10 =52
    i = int(dclass)                         # 52+1

    fileslist1 = files[0: i]  # zuo bi you kai
    fileslist2 = files[1 * i:2 * i]  #
    fileslist3 = files[2 * i:3 * i]  # 425:636
    fileslist4 = files[3 * i:4 * i]
    fileslist5 = files[4 * i:5 * i]
    fileslist6 = files[5 * i:6 * i]
    fileslist7 = files[6 * i:7 * i]
    fileslist8 = files[7 * i:8 * i]
    fileslist9 = files[8 * i:9 * i]
    fileslist10 = files[9 * i:len(files)]  # 637:852


    #start four thread
    thread1 = myThread(1,"Thread-1",fileslist1)
    thread2 = myThread(2,"Thread-2",fileslist2)
    thread3 = myThread(3,"Thread-3",fileslist3)
    thread4 = myThread(4,"Thread-4",fileslist4)
    thread5 = myThread(5, "Thread-5", fileslist5)
    thread6 = myThread(6, "Thread-6", fileslist6)
    thread7 = myThread(7, "Thread-7", fileslist7)
    thread8 = myThread(8, "Thread-8", fileslist8)
    thread9 = myThread(9, "Thread-9", fileslist9)
    thread10 = myThread(10, "Thread-10", fileslist10)

    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()
    thread6.start()
    thread7.start()
    thread8.start()
    thread9.start()
    thread10.start()


# call the main
if __name__ == "__main__":
    main()