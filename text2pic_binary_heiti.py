# -*- coding: utf-8 -*-





import os
import pygame
from pygame.locals import *
import numpy as np
from PIL import Image
import cPickle


#datapath00=u"e:\\汉字\\word.txt"

datapath1=u"e:\\汉字\\words_1.txt" #####################
#datapath2=u"e:\\汉字\\words_2.txt"
def demo():
    pygame.init()


    text = u"我";print text
    #font = pygame.font.SysFont('SimHei', 32)
    font = pygame.font.SysFont('', 32)
    ftext = font.render(text, True, (0, 0, 0), (255, 255, 255))
 
    pygame.image.save(ftext, "e:\\t.jpg") #d:\t.jpg  not work


def loaddata(datapath):
    content=open(datapath,'r')
    dic=[]
    #line= content.readline().strip('\n').split(' ')
    line= [char for char in content.readline().strip('\n').split(' ') if len(char)>1];#print 'line1',line
    while line.__len__()>1:
        #line=line.split(' ');
        #print 'line2',line.__len__()
        for char in line:
            if char.__len__()>1:
                dic.append(str(char))
        #print 'dic',dic
        line= [char for char in content.readline().strip('\n').split(' ') if len(char)>1];#print 'line3',line

    print dic.__len__(),unicode(dic[-1], "cp936") #print pass variable to system system use cp936
    dic_d={}
    for i in range(dic.__len__()):
        dic_d[i]=dic[i]
    return dic_d
    ###########

def single_word2pic(word,i):
    global pic_arr
    pygame.init()

    text = unicode(word, "cp936");print text
    #font = pygame.font.SysFont('SimHei', 32) KaiTi
    font = pygame.font.SysFont('SimHei', 32)
    ftext = font.render(text, True, (0, 0, 0), (255, 255, 255))

    pygame.image.save(ftext, "e:\\charactor_1\\"+str(i)+".jpg") #d:\t.jpg  not work
    pic_list.append(np.array(Image.open("e:\\charactor_1\\"+str(i)+".jpg").convert('L'))[:32,:32])
        
def save_feats(featlist,name):#[w_floor1,b1,w2,b2]
    featpath="e:\\charactor_1\\"+str(name)
    write_file=open(featpath,'wb')
    for feat in featlist:
        cPickle.dump(feat,write_file,-1)
    write_file.close()



def load_pickle(path):
    f=open(path,'rb')
    dic1=cPickle.load(f)
    pic_arr1=cPickle.load(f)
    print unicode(dic1.values()[0],"cp936")
    print pic_arr1.shape
#####
if __name__=="__main__":
    '''
    demo()
    '''


    ####load txt
    dic_d=loaddata(datapath1);print 'len total',dic_d.__len__()
    global pic_list;pic_list=[]
    ##txt->pic->array
    for k,v in dic_d.items():
        single_word2pic(v,k)
     
    pic_arr= np.array(pic_list)
    print 'arr',pic_arr.shape
    ###############
    save_feats([dic_d],'dic')
    save_feats([pic_arr],'pic_arr_heiti')


    '''
    #####load pickle
    featpath="e:\\charactor_1\\dic_pic"
    load_pickle(featpath)
    '''
