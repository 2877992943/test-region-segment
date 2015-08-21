#!/usr/bin/env python
# encoding=utf-8



'''
only erosion and dilation, not svm classify
use erzhi a_dd as feat 
dd01/10 or dd01/5
'''

__docformat__='restructedtext en'

import os,sys,time,theano,cPickle
import theano.tensor as T
import numpy as np

impath='/home/yr/work_result/text_spotting/24.jpg'
 
import pylab as plt;from PIL import Image

from scipy.cluster.vq import*
from scipy.misc import imresize
import pywt
from scipy.ndimage import measurements,morphology

def resize2(dd,imarr): ##wave_dd ,half of imarr size  
	from scipy.misc import imresize
	scale=2. #2 not work
	#########resize
	dd_1=imresize(dd,scale,interp='nearest');print 'resize',dd.shape,dd_1.shape
	#####
	#plt.figure();plt.subplot(1,2,1);plt.imshow(dd)
	#plt.subplot(1,2,2);plt.imshow(dd_1);
	#plt.show()
	###if twice sized dd sz!=imarr.shape
	dd_2=np.zeros(imarr.shape)
	if dd_1.shape>imarr.shape:
		dd_2=dd_1[:imarr.shape[0],:imarr.shape[1]]
	if dd_1.shape<imarr.shape:
		dd_2[dd_1.shape[0],dd_1.shape[1]]=dd_1
	if dd_1.shape==imarr.shape:dd_2=dd_1
	print '==?',dd_2.shape,imarr.shape#531x945  ,set(dd_2.flatten())  #set([0, 1, 2, 3, 4, 5, 6, 7, 250, 251, 252, ]) 
	####### binary_dd01 <-uint8_dd #set([0, 1, 2, 3, 4, 5, 6, 7, 250, 251, 252, ])  
	dd01=np.zeros((dd_2.shape));dd01[np.where(dd_2>255*0.4)]=1
	#####
	'''
	plt.figure();plt.subplot(1,2,1);plt.imshow(dd01*255.);plt.title('dd01') #dd01--binary01
	plt.subplot(1,2,2);plt.imshow(dd);plt.title('dd')  #dd--uint8
	plt.show()
	'''
	return dd01



def save(imarr01,dd01):
	f=open('/home/yr/work_result/text_spotting/imarr01_ddErosDil01','wb')
	cPickle.dump(imarr01,f,-1)
	cPickle.dump(dd01,f,-1)
	f.close()

	
		
if __name__=="__main__":
	imarr=np.array(Image.open(impath).convert('L'))
	print imarr.shape
	imarr1=imarr.copy()
	wp = pywt.WaveletPacket2D(data=imarr, wavelet='db1', mode='sym')
	print 'wp data',wp.data.shape
	print 'a',wp['a'].data.shape
	print 'h',wp['h'].data.shape
	print 'v',wp['v'].data.shape
	print 'd',wp['d'].data.shape
	 
	####draw wave 4 piece :  average, diffhoriz, diffvertic, diffd 
	 
	plt.figure();plt.gray()
	plt.subplot(2,3,1);plt.imshow(np.uint8(wp['a'].data/10));plt.title('np.uint8(wp[a].data/10)')
	plt.subplot(2,3,2);plt.imshow(np.uint8(wp['h'].data/10));plt.title('np.uint8(wp[h].data/10)')
	plt.subplot(2,3,3);plt.imshow(np.uint8(wp['v'].data/10));plt.title('np.uint8(wp[v].data/10)')
	plt.subplot(2,3,4);plt.imshow(np.uint8(wp['d'].data/10));plt.title('np.uint8(wp[d].data/10)')

	#dd=wp['d'].data*wp['d'].data # +wp['h'].data*wp['h'].data+wp['v'].data*wp['v'].data
	#im1=(dd-np.min(dd) ) /(np.max(dd)-np.min(dd) );print 'im1',im1.max(),im1.min()
	 
	#plt.subplot(2,3,5);plt.imshow(im1*255.) ;plt.title('dd**2 normalize')
	 
	#print 'dd',np.max(np.uint8(wp['d'].data.flatten()/10)),np.max(0.1*wp['d'].data.flatten())###255 6.45
	
	#plt.show()
	 
	 
	###########################wave dd->resize->binary->calculate
	dd=np.uint8(wp['d'].data/10); #print 'dd binary',dd.max(),dd.min(),wp['d'].data.max()#255,0 77
	#####dd resize x2 ->dd01 binary
	dd01=resize2(dd,imarr)#dd_2=expanded dd ,dd_1 should be same size with imarr
	#####a01 binary ,a is half sized imarr, so use imarr01
	#a=np.uint8(wp['a'].data/10); #print 'a',a.max(),a.min(),wp['a'].data.max()#51 0 510
	#a01=np.zeros(a.shape);a01[np.where(a>a.max()*0.4)]=1
	imarr01=np.zeros(imarr.shape);imarr01[np.where(imarr>imarr.max()*0.6)]=1##if black word white paper, 0.6,otherwise0.4
	
	
	 


	 
	###########morphology erosion and dilation
	kernel=np.array([[0,1,0],[1,1,1],[0,1,0]])
	im_dil=morphology.binary_dilation(dd01,structure=kernel,iterations=2) 
	im_ero=morphology.binary_erosion(im_dil, structure=np.ones((2,5)),iterations=2)
	im_dil2=morphology.binary_dilation(im_ero,structure=np.ones((5,2)),iterations=2) 
	print 'dil2',set(im_dil2.flatten()),im_dil2.flatten()[:10] ###true false pic
	####im_dil2 from false_true pic->01
	dd01_ero=np.zeros(im_dil2.shape);dd01_ero[np.where(im_dil2==True)]=1
	########
	
	
	plt.figure();plt.gray();
	plt.subplot(2,2,1);plt.imshow(im_dil);plt.title('first dil')
	plt.subplot(2,2,2);plt.imshow(im_ero);plt.title('then ero')
	plt.subplot(2,2,3);plt.imshow(im_dil2);plt.title('and then dil')
	plt.subplot(2,2,4);plt.imshow(imarr01);plt.title('imarr01')
	#plt.show()
	###########
	 
	#imarr01[np.where(im_dil2==0)]=0
	plt.figure();plt.gray();
	plt.imshow(dd01_ero)
	####plt.imshow(imarr01)
	###plt.imshow(dd) 
	plt.show()
	#####
	print 'imarr01, ddero01',imarr01.shape,dd01_ero.shape
	save(imarr01,dd01_ero) 
	 
	
	
	
			


 
'''
imarr  imarr01  dd01 -- size of imarr
dd dh dv a  -- size of half imarr
1)why not use alone CNN /svm window: cannot erase edge
2)why not use wave_dd dila, erosion alone, noise cannot be erase
3)why not use dd directly, must dila+erosion before cnn,reduce noise,get connected entity, see effect picture
'''
		


