#!/usr/bin/env python
# encoding=utf-8
 
'''
this read imarr01 dd_ero_01
based on each patch 50x50 of dd_ero_01, decide on whether not go into cnn
use rectangular to get region of text region instead of only dila+eros
'''
#####__docformat__='restructedtext en'

from PIL import Image
import numpy as np
import os,cPickle,pylab
import pylab as plt

datapath='/home/yr/work_result/text_spotting/imarr01_ddErosDil01'
 
 
featpath='/home/yr/work_result/text_spotting/cnn_textsegm_para'


def load_para():
	global p1,p2,p3,p4,p5,p6,p7,p8	
	f=open(featpath,'rb')
	p1=cPickle.load(f) #[50,2]
	p2=cPickle.load(f) #[2,]
	p3=cPickle.load(f) #[567,50]
	p4=cPickle.load(f) #[50,]
	p5=cPickle.load(f) #[7 3 6 6]
	p6=cPickle.load(f) #[7,]
	p7=cPickle.load(f) #[3 1 5 5]
	p8=cPickle.load(f) #[3,]
   	f.close()
	print p1.shape#[0].shape,p1[1].shape
	print p2.shape,p3.shape,p4.shape,p5.shape,p6.shape,p7.shape,p8.shape
def load_data():
	global imarr01,dd01
	f=open(datapath,'rb')
	imarr01=cPickle.load(f)
	dd01=cPickle.load(f) #this dd is binary01 ,size of imarr, after dila+eros
	f.close()
	'''####
	plt.figure();plt.gray()
	plt.subplot(1,2,1);plt.imshow(imarr01);
	plt.subplot(1,2,2);plt.imshow(dd01)
	plt.show()
	#####'''
	print 'imarr01, dd01',imarr01.shape,dd01.shape
	
def try_cnn(patch):
	global p1,p2,p3,p4,p5,p6,p7,p8	
	#imarr=np.array(Image.open(datapath).convert('L'));#print 'imarr',imarr.shape
	imarr=patch#50x50
	####conv1 50x50->3x46x46   kernel5x5
	conv_out1=np.zeros((3,46,46))
	for i in range(3)[:]: #3kernels
		kernel=p7[i,0,:,:];#print kernel.shape
		kernel=rot90(rot90(kernel))
		conv_out1[i,:,:]=conv_(kernel,imarr) +p8[i] #5x5  50x50
		
		
	###maxpool1
	pool_out1=pool_(conv_out1) #3x23x23
	#print 'pool1',pool_out1.shape 
	####tanh1
	out1=np.tanh(pool_out1) #3x23x23
	#####conv2   3x23x23->7x18x18 kernel 6x6
	conv_out2=np.zeros((7,18,18))
	for k in range(7)[:]:
		###for each kernel [3x6x6]
		map3=np.zeros((3,18,18))
		for i in range(3):
			kernel=p5[k,i,:,:]#3x6x6
			kernel=rot90(rot90(kernel))
			map3[i,:,:]=conv_(kernel,out1[i,:,:]) #6x6  23x23->18x18
		map2=map3.sum(axis=0);#print 'map2',map2.shape #3x18x18->18x18
		###############
		conv_out2[k,:,:]=map2+p6[k]
	######pool2 7x18x18->7x9x9
	pool_out2=pool_(conv_out2)
	#print 'pool2',pool_out2.shape
	####tanh2  7x9x9
	out2=np.tanh(pool_out2)
	#####hidden layer
	full=np.dot(out2.flatten(),p3) #567  567x50->1x50
	full=full+p4;#print 'hid',full.shape #[50,]
	full=np.tanh(full)
	#####log-regression
	full=np.dot(full,p1)+p2 ;#print 'log reg',full.shape #[2,]
	rst=np.argmax(full);#print 'result',m,full
	return rst
###############################################################	
def rot90(patch):#5x5
	patch=patch.T
	p1=np.zeros(patch.shape)
	m=p1.shape[0]
	for i in range(m): #m=5  [0,1,2,3,4]
		p1[:,i]=patch[:,m-i-1]
	return p1		


def conv_(kernel,imarr): #5x5  50x50
	k=kernel.shape[0];m=imarr.shape[0]
	mk=m-k+1
	out=np.zeros((mk,mk))
	for i in range(mk):
		for j in range(mk):
			  
			out[i,j]=np.sum(imarr[i:i+k,j:j+k]*kernel)
	return out
	
def pool_(conv_out):#3x46x46
	sz=2
	kk,m,n=conv_out.shape
	pool_out=np.zeros((kk,int(m/2),int(n/2)))
	for k in range(kk):
		for i in range(int(m/2)):
			for j in range(int(n/2)):
				pool_out[k,i,j]=np.max(conv_out[k,i*sz:i*sz+sz,j*sz:j*sz+sz])
	return pool_out
		
def tanh_(pool_out):   ###always overflow exp
	#t=np.exp(2*pool_out);#not overflow
	#t1=np.exp(-pool_out)#overflow
	#fenz=np.exp(pool_out)-np.exp(-pool_out)	
	#fenmu=np.exp(pool_out)+np.exp(-pool_out)
	fenz=np.exp(2*pool_out)-1
	fenmu=np.exp(2*pool_out)+1	
	return fenz/(fenmu+0.00001)


def save(arr):
	f=open('/home/yr/work_result/text_spotting/text_region_tobe_sliced','wb')
	cPickle.dump(arr,f,-1)
	f.close()

if __name__=='__main__':
	load_para()
	####load imarr01 dd01 
	load_data()
	 
	#######################cnn judge
	global imarr01,dd01
	stride=20.;patch=[50,50] 
	hang,lie=imarr01.shape#600,900
	num_h=int((hang-patch[0])/stride)+1;num_l=int((lie-patch[1])/stride+1)+1 
	###find density harris
	density_list=[]
	imarr01_pos=imarr01.copy()####see patch50x50  block
	for i in range(num_h)[:-1]:
		for j in range(num_l)[:-1]:
			#######filter patch 50x50
			density=dd01[i*stride:i*stride+patch[0],j*stride:j*stride+patch[1]].sum()
			if density<100 and density>=1:
			 	imarr01_pos[i*stride:i*stride+patch[0],j*stride:j*stride+patch[1]]=1#see which patches will go through cnn
			#########
	    			patch_arr=imarr01[i*stride:i*stride+patch[0],j*stride:j*stride+patch[1]]
				rst=try_cnn(patch_arr)  #result of classify the patch 1-textregion  0-nontext regin
				#print rst
				if rst==0: #reduce noise which cannot be erased by dila+eros
					dd01[i*stride:i*stride+patch[0],j*stride:j*stride+patch[1]]=0
						 
	#############
	pylab.figure();pylab.gray();
	 
	pylab.subplot(2,2,1);pylab.imshow(imarr01);pylab.title('imarr01')
	pylab.subplot(2,2,2);pylab.imshow(dd01);pylab.title('dd01')
	pylab.subplot(2,2,3);pylab.imshow(imarr01_pos);pylab.title('imarr01 position to go through cnn')
	  
	#####after erase the noise, dila the dd01 to single out only text region without edge of form
	######wavelet contribute to get rid of edge of form
	kernel=np.array([[0,1,0],[1,1,1],[0,1,0]])
	from scipy.ndimage import morphology
	dd01_=morphology.binary_dilation(dd01,structure=np.ones((5,20)),iterations=2)  #ones(5,2)
	#dd01__=morphology.binary_dilation(dd01_,structure=np.ones((5,10)),iterations=2)#fill the hole between lines
	#dd01___=morphology.binary_erosion(dd01__,structure=kernel,iterations=2)
	#dd01___=morphology.binary_erosion(dd01___,structure=np.ones((5,5)),iterations=2)
	#dd01__=np.zeros(dd01_.shape);dd01__[np.where(dd01_==True)]=1
	#dd01_=dd01__
	pylab.figure();pylab.gray()
	pylab.subplot(1,3,1);pylab.imshow(dd01_);pylab.title('dd01 dila ')
	#pylab.subplot(1,3,2);pylab.imshow(dd01__);pylab.title('dila')
	#pylab.subplot(1,3,3);pylab.imshow(dd01___);pylab.title('eros')
	####
	################### dd01___ rectangular
	dd01___=np.zeros(dd01_.shape)
	nz_x=np.nonzero(dd01_)[0];print 'x',nz_x.shape
	nz_y=np.nonzero(dd01_)[1];print 'y',nz_y.shape
	xmax=np.max(nz_x);xmin=np.min(nz_x);
	ymax=np.max(nz_y);ymin=np.min(nz_y);
	dd01___[xmin:xmax,ymin:ymax]=1
	pylab.subplot(1,3,2);pylab.imshow(dd01___);pylab.title('dd01 rectangular') 



	 
	#################dd01___  -> imarr01_
	imarr01_=imarr01.copy()
	imarr01_[np.where(dd01___==0)]=0####dd01___
	#imarr01_[np.where(dd01__==False)]=0
	#imarr01_=1.0-imarr01_##############
	###############print original and filtered pic
	pylab.figure();pylab.subplot(1,2,1);pylab.imshow(imarr01_);pylab.title('imarr01_text region')
	pylab.subplot(1,2,2);pylab.imshow(imarr01);pylab.title('imarr01')
	final=np.vstack((imarr01,np.ones((2,imarr01.shape[1])),imarr01_))
	pylab.figure();pylab.imshow(final);
	
	##################
	
	pylab.show()
	##################save text_region to be sliced
	#save(imarr01_)
	
	 
				
	 
	
	
	 


	

	
	
		
 			

		


