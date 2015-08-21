#!/usr/bin/env python
# encoding=utf-8


#####__docformat__='restructedtext en'
'''
this is to cut from 1000x1000 pic to generate trainset of patches 50x50 ,text region or non-text region
stride, kernel_size could be adjusted
'''

from PIL import Image
import numpy as np
import os


datapath='/home/yr/computer_vision/ppt_photo/'

def get_imlist(path):
	return [os.path.join(path,f) for f in os.listdir(path) if \
		f.endswith('.jpg')]


def imresize(imlist):
	for imname in imlist:
		im=Image.open(imname).convert('L')
		im1=im.resize((80,80))
		im1.save('/home/yr/computer_vision/data_resize/'+imname[31:])

 
def conv_cut(imname,count):
	stride=20.;kernel=50.# stride=2,kernel=3
	imarr=np.array(Image.open(imname).convert('L'))
	hang,lie=imarr.shape#600,900
	num_h=int((hang-kernel)/stride)+1;num_l=int((lie-kernel)/stride+1)+1# h8 l8 kernel3 stride2  (8-3)/2+1
	for i in range(num_h):
		for j in range(num_l):
			patch=imarr[i*stride:i*stride+kernel,j*stride:j*stride+kernel]
			im=Image.fromarray(np.uint8(patch))
			im.save('/home/yr/computer_vision/patch/'+str(count)+'_'+str(i)+str(j)+'.jpg')
	

	
		

if __name__=='__main__':
	print len(os.listdir(datapath))
	imlist=get_imlist(datapath)
	print len(imlist),imlist[0]
	#imresize(imlist)
	####resize done 64x64
	for imname in imlist[55:58]:
		ind=imlist.index(imname)
		conv_cut(imname,ind)
	
	 
	
	
	 


	

	
	
		

			
			

		


