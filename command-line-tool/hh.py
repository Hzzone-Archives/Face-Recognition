# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "/Users/HZzone/caffe/python")
import caffe
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt  



### 具体流程
'''
（1）构建金字塔输入到网络。
（2）对金字塔的每一层，通过CNN自动滑动窗口，这是全卷积神经网络，没有全连接层，有stride，然后保存输出的人脸的窗口位置和概率。
（3）将人脸窗口映射到原图img中的人脸位置，概率不变。
（4）NMS。
 (5) 2，3网络没有第1，2操作，输出的bounding box 都要对齐、补全成一个正方形
（6）在原图画出人脸位置。
'''
def bbreg(boundingbox, reg):
    reg = reg.T 
    
    # calibrate bouding boxes
    if reg.shape[1] == 1:
        print "reshape of reg"
        pass # reshape of reg
    w = boundingbox[:,2] - boundingbox[:,0] + 1
    h = boundingbox[:,3] - boundingbox[:,1] + 1

    bb0 = boundingbox[:,0] + reg[:,0]*w
    bb1 = boundingbox[:,1] + reg[:,1]*h
    bb2 = boundingbox[:,2] + reg[:,2]*w
    bb3 = boundingbox[:,3] + reg[:,3]*h
    
    boundingbox[:,0:4] = np.array([bb0, bb1, bb2, bb3]).T
    #print "bb", boundingbox
    return boundingbox


def pad(boxesA, w, h):
    boxes = boxesA.copy() # shit, value parameter!!!
    #print '#################'
    #print 'boxes', boxes
    #print 'w,h', w, h
    
    tmph = boxes[:,3] - boxes[:,1] + 1
    tmpw = boxes[:,2] - boxes[:,0] + 1
    numbox = boxes.shape[0]

    #print 'tmph', tmph
    #print 'tmpw', tmpw

    dx = np.ones(numbox)
    dy = np.ones(numbox)
    edx = tmpw 
    edy = tmph

    x = boxes[:,0:1][:,0]
    y = boxes[:,1:2][:,0]
    ex = boxes[:,2:3][:,0]
    ey = boxes[:,3:4][:,0]
   
   
    tmp = np.where(ex > w)[0]
    if tmp.shape[0] != 0:
        edx[tmp] = -ex[tmp] + w-1 + tmpw[tmp]
        ex[tmp] = w-1

    tmp = np.where(ey > h)[0]
    if tmp.shape[0] != 0:
        edy[tmp] = -ey[tmp] + h-1 + tmph[tmp]
        ey[tmp] = h-1

    tmp = np.where(x < 1)[0]
    if tmp.shape[0] != 0:
        dx[tmp] = 2 - x[tmp]
        x[tmp] = np.ones_like(x[tmp])

    tmp = np.where(y < 1)[0]
    if tmp.shape[0] != 0:
        dy[tmp] = 2 - y[tmp]
        y[tmp] = np.ones_like(y[tmp])
    
    # for python index from 0, while matlab from 1
    dy = np.maximum(0, dy-1)
    dx = np.maximum(0, dx-1)
    y = np.maximum(0, y-1)
    x = np.maximum(0, x-1)
    edy = np.maximum(0, edy-1)
    edx = np.maximum(0, edx-1)
    ey = np.maximum(0, ey-1)
    ex = np.maximum(0, ex-1)

    return [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]



def rerec(bboxA):
    # convert bboxA to square
    w = bboxA[:,2] - bboxA[:,0]
    h = bboxA[:,3] - bboxA[:,1]
    l = np.maximum(w,h).T
    
    #print 'bboxA', bboxA
    #print 'w', w
    #print 'h', h
    #print 'l', l
    bboxA[:,0] = bboxA[:,0] + w*0.5 - l*0.5
    bboxA[:,1] = bboxA[:,1] + h*0.5 - l*0.5 
    bboxA[:,2:4] = bboxA[:,0:2] + np.repeat([l], 2, axis = 0).T 
    return bboxA


### 传入得分再加上scale
###
def nms(boxes, threshold, type):
    """nms
    :boxes: [:,0:5]
    :threshold: 0.5 like
    :type: 'Min' or others
    :returns: TODO
    """
    if boxes.shape[0] == 0:
        return np.array([])
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = np.array(s.argsort()) # read s using I
    
    pick = [];
    ### 循环直到没有重复的框
    while len(I) > 0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if type == 'Min':
            o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
        else:
            o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where( o <= threshold)[0]]
    return pick


### out['prob1'][0,1,:,:], out['conv4-2'][0], scale, threshold[0]
def generateBoundingBox(map, reg, scale, t):
    stride = 2
    cellsize = 12
    map = map.T
    dx1 = reg[0,:,:].T
    dy1 = reg[1,:,:].T
    dx2 = reg[2,:,:].T
    dy2 = reg[3,:,:].T
    ### 选出得分大于阈值的候选框
    (x, y) = np.where(map >= t)

    score = map[x,y]
    print "generateBoundingBox: "
    print "x", x
    print "y", y
    print "score", score
    ### 所有的候选框, 你看输出有很多个
    reg = np.array([dx1[x,y], dy1[x,y], dx2[x,y], dy2[x,y]])

    if reg.shape[0] == 0:
        pass
    boundingbox = np.array([y, x]).T

    ### 窗体滑动
    bb1 = np.fix((stride * (boundingbox) + 1) / scale).T # matlab index from 1, so with "boundingbox-1"
    bb2 = np.fix((stride * (boundingbox) + cellsize - 1 + 1) / scale).T # while python don't have to
    score = np.array([score])

    ### 就把所有的候选框再加上得分返回，窗体滑动的位置
    boundingbox_out = np.concatenate((bb1, bb2, score, reg), axis=0)


    return boundingbox_out.T



### 最后绘图
def drawBoxes(im, boxes):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    for i in range(x1.shape[0]):
        cv2.rectangle(im, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), (0,0,255), 3)
    return im



def detect_face(img, minsize, PNet, RNet, ONet, threshold, fastresize, factor):
    
    img2 = img.copy()

    factor_count = 0
    total_boxes = np.zeros((0,9), np.float)
    points = []
    h = img.shape[0]
    w = img.shape[1]
    minl = min(h, w)
    img = img.astype(float)
    m = 12.0/minsize
    minl = minl*m
    


    
    # create scale pyramid
    scales = []
    while minl >= 12:
        scales.append(m * pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    
    # first stage
    for scale in scales:
        hs = int(np.ceil(h*scale))
        ws = int(np.ceil(w*scale))

        if fastresize:
            im_data = (img-127.5)*0.0078125 # [0,255] -> [-1,1]
            im_data = cv2.resize(im_data, (ws,hs)) # default is bilinear
        else: 
            im_data = cv2.resize(img, (ws,hs)) # default is bilinear
            im_data = (im_data-127.5)*0.0078125 # [0,255] -> [-1,1]


        im_data = np.swapaxes(im_data, 0, 2)
        im_data = np.array([im_data], dtype = np.float)
        PNet.blobs['data'].reshape(1, 3, ws, hs)
        PNet.blobs['data'].data[...] = im_data
        out = PNet.forward()
        print "scale: ", scale
        print "prob: ", out['prob1'][0,1,:,:].shape
        print "boundingbox: ", out['prob1'][0,1,:,:].shape
        boxes = generateBoundingBox(out['prob1'][0,1,:,:], out['conv4-2'][0], scale, threshold[0])
        ### 这里就用bounding box， 判断一下有没有脸再做
        if boxes.shape[0] != 0:

            pick = nms(boxes, 0.5, 'Union')

            if len(pick) > 0 :
                boxes = boxes[pick, :]

        if boxes.shape[0] != 0:
            ### 这里就把所有scale的候选框都保存到total_boxes里面
            total_boxes = np.concatenate((total_boxes, boxes), axis=0)
         

    numbox = total_boxes.shape[0]
    if numbox > 0:
        # nms
        pick = nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        
        # revise and convert to square
        ### 因为后两个网络有全连接层，所以必须弄到正方形
        regh = total_boxes[:,3] - total_boxes[:,1]
        regw = total_boxes[:,2] - total_boxes[:,0]
        t1 = total_boxes[:,0] + total_boxes[:,5]*regw
        t2 = total_boxes[:,1] + total_boxes[:,6]*regh
        t3 = total_boxes[:,2] + total_boxes[:,7]*regw
        t4 = total_boxes[:,3] + total_boxes[:,8]*regh
        t5 = total_boxes[:,4]
        total_boxes = np.array([t1,t2,t3,t4,t5]).T
        total_boxes = rerec(total_boxes) # convert box to square
        
        total_boxes[:,0:4] = np.fix(total_boxes[:,0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)


    numbox = total_boxes.shape[0]
    ### 把候选框输入到第二个网络 这里就和第一个网络同样的操作，只不过没有构造图像金字塔
    if numbox > 0:
        # second stage


        # construct input for RNet
        tempimg = np.zeros((numbox, 24, 24, 3)) # (24, 24, 3, numbox)
        for k in range(numbox):
            tmp = np.zeros((int(tmph[k]) +1, int(tmpw[k]) + 1,3))
          

            tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]

            tempimg[k,:,:,:] = cv2.resize(tmp, (24, 24))

        tempimg = (tempimg-127.5)*0.0078125 # done in imResample function wrapped by python


        # RNet

        tempimg = np.swapaxes(tempimg, 1, 3)
        
        RNet.blobs['data'].reshape(numbox, 3, 24, 24)
        RNet.blobs['data'].data[...] = tempimg
        out = RNet.forward()


        score = out['prob1'][:,1]
        pass_t = np.where(score>threshold[1])[0]
        
        score =  np.array([score[pass_t]]).T
        total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis = 1)

        
        mv = out['conv5-2'][pass_t, :].T
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, 0.7, 'Union')
            #print 'pick', pick
            if len(pick) > 0 :
                total_boxes = total_boxes[pick, :]
                total_boxes = bbreg(total_boxes, mv[:, pick])
                total_boxes = rerec(total_boxes)
            
        #####
        # 2 #
        #####

        ### 第三步就输出关键点，同时对应到每个bounding box
        numbox = total_boxes.shape[0]
        if numbox > 0:
            # third stage
            
            total_boxes = np.fix(total_boxes)
            [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(total_boxes, w, h)

        

            tempimg = np.zeros((numbox, 48, 48, 3))
            for k in range(numbox):
                tmp = np.zeros((int(tmph[k]), int(tmpw[k]),3))
                tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
                tempimg[k,:,:,:] = cv2.resize(tmp, (48, 48))
            tempimg = (tempimg-127.5)*0.0078125 # [0,255] -> [-1,1]
                
            # ONet
            tempimg = np.swapaxes(tempimg, 1, 3)
            ONet.blobs['data'].reshape(numbox, 3, 48, 48)
            ONet.blobs['data'].data[...] = tempimg
            out = ONet.forward()
            
            score = out['prob1'][:,1]
            points = out['conv6-3']
            pass_t = np.where(score>threshold[2])[0]
            points = points[pass_t, :]
            score = np.array([score[pass_t]]).T
            total_boxes = np.concatenate( (total_boxes[pass_t, 0:4], score), axis=1)
            
            mv = out['conv6-2'][pass_t, :].T
            w = total_boxes[:,3] - total_boxes[:,1] + 1
            h = total_boxes[:,2] - total_boxes[:,0] + 1

            points[:, 0:5] = np.tile(w, (5,1)).T * points[:, 0:5] + np.tile(total_boxes[:,0], (5,1)).T - 1 
            points[:, 5:10] = np.tile(h, (5,1)).T * points[:, 5:10] + np.tile(total_boxes[:,1], (5,1)).T -1

            if total_boxes.shape[0] > 0:
                total_boxes = bbreg(total_boxes, mv[:,:])
                pick = nms(total_boxes, 0.7, 'Min')
                
                #print pick
                if len(pick) > 0 :
                    total_boxes = total_boxes[pick, :]
                    points = points[pick, :]

    return total_boxes, points


def aligment(img, imgpath):
    ### 人脸最小的大小
    minsize = 20


    threshold = [0.6, 0.7, 0.7]
    factor = 0.709

    img_matlab = img.copy()
    tmp = img_matlab[:,:,2].copy()
    img_matlab[:,:,2] = img_matlab[:,:,0]
    img_matlab[:,:,0] = tmp

    # check rgb position
    #tic()
    boundingboxes, points = detect_face(img_matlab, minsize, PNet, RNet, ONet, threshold, False, factor)
    #toc()
    print points.shape
    print boundingboxes.shape
    area = np.array(abs(boundingboxes[:,2]-boundingboxes[:,0])*(boundingboxes[:,3]-boundingboxes[:,1]))
    max_index = area.argmax()
    initial_img = img.copy()
    drawBoxes(img, boundingboxes[max_index].reshape(1, -1))
    # cv2.imshow('xiaoxiongwoaini', img)
    # cv2.waitKey(0)
    x1 = boundingboxes[max_index,0]
    y1 = boundingboxes[max_index,1]
    x2 = boundingboxes[max_index,2]
    y2 = boundingboxes[max_index,3]
    w = x2-x1
    h = y2-y1
    max_rect_points = points[max_index].reshape((2,5))
    print max_rect_points

    from oct2py import octave
    octave.eval("pkg load image")
    # imgSize = [112, 96] 
    # octave.push('imgSize', imgSize)
    # coord5points = [[30.2946, 65.5318, 48.0252, 33.5493, 62.7299],[51.6963, 51.5014, 71.7366, 92.3655, 92.2041]] 
    # octave.push('coord5points', coord5points)
    # facial5points = [[347.32406616, 535.37561035, 415.44692993, 345.96670532, 498.96899414],[406.24920654,435.16409302, 539.11853027,592.89459229 , 617.25360107]] 
    # octave.push('facial5points', facial5points)
    # octave.eval("img = imread('%s');"%imgpath) 

    # octave.eval('''Tfm = cp2tform(facial5points', coord5points', 'similarity');''') 
    # octave.eval('''cropImg = imtransform(img, Tfm, 'XData', [1 imgSize(2)], 'YData', [1 imgSize(1)], 'Size', imgSize);''') 
    # cropImg = octave.pull('cropImg')
    cropImg = octave.align(imgpath, max_rect_points)
    print cropImg.shape
    cropImg = cv2.cvtColor(cropImg, cv2.COLOR_BGR2RGB);

    # cv2.imshow("小熊好可爱", cropImg)
    # cv2.waitKey(0)
    return cropImg

def cosine_distnace(v1, v2):
    cos = np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
    return cos


def predict(im1, im2, threshold):
    # cv2.imshow("", im1)
    # cv2.waitKey(0)
    # cv2.imshow("", im2)
    # cv2.waitKey(0)
    data = np.zeros((2, 3, 112, 96))
    net.blobs['concat_data'].reshape(2, 3, 112, 96)
    data[0] = np.transpose(im1, (2, 0, 1))
    data[1] = np.transpose(im2, (2, 0, 1))
    net.blobs['concat_data'].data[...] = data - 128
    features = net.forward()["fc5"]
    print features.shape
    print features
    d = cosine_distnace(features[0], features[1])
    is_same_person = False
    print d
    if d >= threshold:
        is_same_person = True
    return is_same_person

if __name__ == "__main__":
    # if len(sys.argv)!=3:
    #     print "请输入两张图片的路径"
    #     exit(1)
    # imgpath1 = sys.argv[1]
    # imgpath2 = sys.argv[2]
    # if not (os.path.isfile(imgpath1) and os.path.isfile(imgpath2)):
    #     print "路径错误"
    #     exit(1)


    imgpath1 = "./Jen_Bice_0001.jpg"
    imgpath2 = "./Jennifer_Capriati_0001.jpg"
    ### initial net
    caffe_model_path = "./model"
    caffe.set_mode_cpu()
    PNet = caffe.Net(caffe_model_path+"/det1.prototxt", caffe_model_path+"/det1.caffemodel", caffe.TEST)
    RNet = caffe.Net(caffe_model_path+"/det2.prototxt", caffe_model_path+"/det2.caffemodel", caffe.TEST)
    ONet = caffe.Net(caffe_model_path+"/det3.prototxt", caffe_model_path+"/det3.caffemodel", caffe.TEST)
    net = caffe.Net("./model/center_loss_ms.prototxt", "./model/center_loss_ms.caffemodel", caffe.TEST)

    img1 = cv2.imread(imgpath1)
    img2 = cv2.imread(imgpath2)
    aligned_im1 = aligment(img1, imgpath1)
    aligned_im2 = aligment(img2, imgpath2)
    flag = predict(aligned_im1, aligned_im2, 0.3889)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB);
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB);
    aligned_im1 = cv2.cvtColor(aligned_im1, cv2.COLOR_BGR2RGB);
    aligned_im2 = cv2.cvtColor(aligned_im2, cv2.COLOR_BGR2RGB);
    plt.figure(figsize=(8,8),dpi=80)
    plt.subplot(321)
    plt.imshow(img1)
    plt.subplot(322)
    plt.imshow(img2)
    plt.subplot(323)
    plt.imshow(aligned_im1)
    plt.subplot(324)
    plt.imshow(aligned_im2)
    plt.subplot(313)
    plt.text(0.5, 0.5, u"same person" if flag else u"different person", size=50, rotation=0.,
         ha="center", va="center")
    plt.axis('off')
    plt.draw()
    plt.show()



