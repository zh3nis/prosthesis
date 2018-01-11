from __future__ import print_function
import io
import os
import imageio
from os import listdir
from os.path import isfile, join
import numpy as np
import math
import PIL.Image as Image
import cv2
import re
import sys
import random

class MyData:
  def __init__(self, path, set_indices, config, data_type):
    self.mystep = config.skipstep
    self.path = path
    self.set_indices = set_indices
    self.batch_size = config.batch_size
    self.num_frames = config.num_frames_per_clip
    self.data_type = data_type
    self.crop_size = config.crop_size
    self.doRandom = False
    if (config.batch_sampling == 'random'):
        self.doRandom = True 
    self.mydb = []
    self.mylabels = []
    self.epoch_size = 0
    self.str_to_label = {
        "run": 0,
        "walk": 1,
        "stand": 2,
        "stairs up": 3,
        "stairs down": 4
    }
    self.read_data()

  def get_batch(self, myind):
    # 1st: from batch_size*0+ind -> to -> batch_size*0 + ind+num_frames-1
    # 2nd: from batch_size*1+ind -> to -> batch_size*1 + ind+num_frames-1
    # ith: from batch_size*i+ind -> to -> batch_size*i + ind+num_frames-1
    # nth: from batch_size*(div_num-1)+ind -> to -> batch_size*(div_num-1) + ind+num_frames-1
    result = []
    labels = []
    ind = self.random_ind[myind]
    for i in range(0, self.batch_size): # by batches 
      # by num of frames
      part1 = self.mydb[	xrange(self.div_num * i + ind, self.div_num * i + ind + self.num_frames, self.mystep), : , : ] 
      lbls = self.mylabels[	xrange(self.div_num * i + ind, self.div_num * i + ind + self.num_frames, self.mystep)]
      labels.append(lbls)
      result.append(part1)
    return (np.array(result), np.array(labels))
          
  def read_data(self):
    first = True
    print("Reading %s trials: " % self.data_type, end='')
    sys.stdout.flush()
    for i in self.set_indices:
      print(i, end=' .. ')
      sys.stdout.flush()
      buf = 'database/set%d/depthsense/' % i
      buf = os.path.join(self.path, buf)
      annf = os.path.join(buf, 'ann.txt')
      depthpath = os.path.join(buf, 'depth/')
      confpath = os.path.join(buf, 'conf/')

      with io.open(annf, encoding="utf-8") as file:
        x = [l.strip() for l in file]

      for j in range(0, len(x)):
        str_action, str_startInd, str_endInd = re.split('\t+', x[j])
        label, startInd, endInd = self.str_to_label[str_action], int(str_startInd), int(str_endInd)
        # read frames from start to end indexes
        for k in range(startInd, endInd + 1):
          pngfilename = "depth%05d.png" % k
          im = imageio.imread(os.path.join(depthpath,pngfilename))
          im = np.array(im)
          img = Image.fromarray(im, 'L')
          if(img.width > img.height):
            scale = float(self.crop_size) / float(img.height)
            img = np.array(
                cv2.resize(np.array(img),(int(img.width * scale + 1), self.crop_size))
            ).astype(np.float32)
          else:
            scale = float(self.crop_size) / float(img.width)
            img = np.array(
                cv2.resize(np.array(img), (self.crop_size, int(img.height * scale + 1)))
            ).astype(np.float32)
          crop_x = int((img.shape[0] - self.crop_size)/2)
          crop_y = int((img.shape[1] - self.crop_size)/2)
          img = img[crop_x : crop_x + self.crop_size, crop_y : crop_y + self.crop_size]# - np_mean[j]
          self.mydb.append(img)
        llab = np.full((1, endInd - startInd + 1), label)
        if first:
          self.mylabels = llab
          first = False
        else:
          self.mylabels = np.append(self.mylabels, llab)
      x[:] = []
  
    self.mydb = np.expand_dims(np.array(self.mydb), axis=3)
    self.div_num = int(math.floor(self.mydb.shape[0]/self.batch_size))
    self.epoch_size = self.div_num - self.num_frames + 1        
    self.random_ind = range(self.epoch_size)
    if (self.doRandom):
        random.shuffle(self.random_ind)
    print("Done. Shape: " + str(self.mydb.shape) + ", Epoch size: " + str(self.epoch_size))
