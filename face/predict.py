# -*- coding: utf-8 -*-
import numpy as np
import os
# caffe_root = '/home/hzzone/caffe'  # this file is expected to be in {caffe_root}/examples/siamese
caffe_root = '/Users/HZzone/caffe'  # this file is expected to be in {caffe_root}/examples/siamese
import sys
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import pylab
import matplotlib.pyplot as plt
import distance
import cv2
import random

'''
plot accuracy map
from features.txt
'''
def plot_accuracy(features_source, totals=6000):
    '''
    :param features_source: the features txt
    :return: None
    '''
    '''
    read features from features.txt
    '''
    with open(features_source) as f:
        features = [line.strip("\n").split(" ") for line in f.readlines()]
    '''
    Temporary generate sequence
    '''
    ###################
    _same = {}
    _diff = {}
    _same_distance = []
    _diff_distance = []
    for i in range(int(totals/2)):
        while True:
            x1 = random.randint(0, len(features)-1)
            x2 = random.randint(0, len(features)-1)
            if not (x1, x2) in _same and features[x1][0] == features[x2][0]:
                _same[(x1, x2)] = ''
                break
        while True:
            x1 = random.randint(0, len(features)-1)
            x2 = random.randint(0, len(features)-1)
            if not (x1, x2) in _diff and features[x1][0] != features[x2][0]:
                _diff[(x1, x2)] = ''
                break
        print(i)
    ###################
    #### get the distances
    for x in _same.keys():
        s1, s2 = x
        d = distance.cosine_distnace(np.array(list(map(float, features[s1][2:]))),
                                     np.array(list(map(float, features[s2][2:]))))
        _same_distance.append(d)
    for x in _diff.keys():
        s1, s2 = x
        d = distance.cosine_distnace(np.array(list(map(float, features[s1][2:]))),
                                     np.array(list(map(float, features[s2][2:]))))
        _diff_distance.append(d)
    print("get the distances complete!")
    ####################
    x_values = pylab.arange(-1.0, 1.0, 0.0001)
    y_values = []
    _same_distance = np.array(_same_distance)
    _diff_distance = np.array(_diff_distance)
    for threshold in x_values:
    	correct = np.sum(_same_distance>=threshold) + np.sum(_diff_distance<threshold)
        y_values.append(float(correct)/totals)
    max_index = np.argmax(y_values)
    plt.title("threshold-accuracy curve")
    plt.xlabel("threshold")
    plt.ylabel("accuracy")
    plt.plot(x_values, y_values)
    plt.plot(x_values[max_index], y_values[max_index], '*', color='red', label="(%s, %s)"%(x_values[max_index], y_values[max_index]))
    plt.legend()
    plt.show()

def output_features(data_source, caffemodel, deploy):
    caffe.set_mode_cpu()
    net = caffe.Net(deploy, caffemodel, caffe.TEST)
    samples = []
    for dir_name in os.listdir(data_source):
        one_person_dir = os.path.join(data_source, dir_name)
        for file_name in os.listdir(one_person_dir):
            one_person_pic_path = os.path.join(one_person_dir, file_name)
            samples.append((dir_name, file_name, one_person_pic_path))
    data = np.zeros((1, 3, 112, 96))
    with open("features.txt", "w") as f:
        for index, sample in enumerate(samples):
            data[0] = np.transpose(cv2.imread(sample[-1]), (2, 0, 1))
            net.blobs['data'].data[...] = data
            output = net.forward()
            features = output["fc5"][0]
            line = "%s %s %s\n" % (samples[index][0], samples[index][1], " ".join(map(str, features.tolist())))
            f.writelines(line)
            print index, samples[index][2]

def predict(d1, d2, threshold=0.9396):
    caffe.set_mode_cpu()
    net = caffe.Net("./face_deploy.prototxt", "./face_model.caffemodel", caffe.TEST)
    data = np.zeros((2, 3, 112, 96))
    data[0] = np.transpose(cv2.imread(d1), (2, 0, 1))
    data[1] = np.transpose(cv2.imread(d2), (2, 0, 1))
    net.blobs['data'].data[...] = data
    output = net.forward()
    features = output["fc5"]
    d = distance.cosine_distnace(features[0], features[1])
    if d>=threshold:
        print "same person"
    else:
        print "diff person"

