#!/usr/bin/env python
# encoding=utf-8


__docformat__='restructedtext en'

import numpy as np
import os,cPickle
from PIL import Image
import pylab

threshold01=255.*0.3
sample=17  #9 not right  13right  17 14


impath='/home/yr/chinese_char/xihua/split_char'
#impath='/home/yr/chinese_char/xihua/char_std'
 
def get_imlist(path):
	return [os.path.join(path,f) for f in os.listdir(path) if \
		f.endswith('.jpg')]

def loaddata_32x32():#image to array
	imname_list=get_imlist(impath)
	arr_list=[np.array(Image.open(name).convert('L'))[:32,:32] for name in imname_list]
	arr=np.array(arr_list);print 'arr',arr.shape #77x32x32
	 
	 
	#####
	
	return arr #77x32x32
	 
def grayto01(arr):#77x32x32
	 
	
	arr01_list=[]
	for i in range(arr.shape[0]):
		
		imarr=(arr[i,:,:]<threshold01)*1;
		arr01_list.append(imarr)  #01array list  1=char 0=background
		#im=Image.fromarray(np.uint8(imarr*255))
		#im.save('/home/yr/chinese_char/show_effect_of01/'+str(i)+'.jpg') #show effect of 01picture
	arr01=np.array(arr01_list)
	print 'arr01',arr01.shape #10x32x32
	return arr01	
	

def try_xihua(imarr01):
	m,n=imarr01.shape #32x32
	
	##pading 0
	imarr01_=np.zeros((m+2,n+2))#34x34
	imarr01_[1:m+1,1:n+1]=imarr01
	imarr01=imarr01_ #34x34
	####
	#mask=np.zeros(imarr01.shape)????
	##################
	loop=True
	while loop:
		 
		#loop=False
		##################################
		### step1
		#####################################
		mask=np.zeros(imarr01.shape)
		mark_num=0
		for i in range(m+2)[1:m+1]:  #0,1....31
			for j in range(n+2)[1:n+1]:
				##########
				#p9 p2 p3
				#p8 p1 p4
				#p7 p6 p5
				p_nb={}
				p=imarr01[i,j]
				p_nb[2]=imarr01[i-1,j]
				p_nb[3]=imarr01[i-1,j+1]
				p_nb[4]=imarr01[i,j+1]
				p_nb[5]=imarr01[i+1,j+1]
				p_nb[6]=imarr01[i+1,j]
				p_nb[7]=imarr01[i+1,j-1]
				p_nb[8]=imarr01[i,j-1]
				p_nb[9]=imarr01[i-1,j-1]
				##########
				if p==1: 
					Np=sum(p_nb.values())
					if Np>=2 and Np<=6: 
						Sp=num_pattern_0to1(p_nb)
						if Sp==1: 
							if p_nb[2]*p_nb[4]*p_nb[6]==0: 
								if p_nb[4]*p_nb[8]*p_nb[6]==0:
									
									##################
									mask[i,j]=1
									mark_num+=1
									#loop=True
		##########################set marked point to 0
		if mark_num!=0:imarr01[np.where(mask==1)]=0
		if mark_num==0:break
		##################################
		###  step2
		########################################
		mask=np.zeros(imarr01.shape)
		mark_num=0
		for i in range(m+2)[1:m+1]:  #0,1....31
			for j in range(n+2)[1:n+1]:
				##########
				#p9 p2 p3
				#p8 p1 p4
				#p7 p6 p5
				p_nb={}
				p=imarr01[i,j]
				p_nb[2]=imarr01[i-1,j]
				p_nb[3]=imarr01[i-1,j+1]
				p_nb[4]=imarr01[i,j+1]
				p_nb[5]=imarr01[i+1,j+1]
				p_nb[6]=imarr01[i+1,j]
				p_nb[7]=imarr01[i+1,j-1]
				p_nb[8]=imarr01[i,j-1]
				p_nb[9]=imarr01[i-1,j-1]
				##########
				if p==1: 
					Np=sum(p_nb.values())
					if Np>=2 and Np<=6: 
						Sp=num_pattern_0to1(p_nb)
						if Sp==1: 
							if p_nb[2]*p_nb[4]*p_nb[8]==0: 
								if p_nb[2]*p_nb[8]*p_nb[6]==0:
									
									##################
									mask[i,j]=1
									mark_num+=1
									#loop=True
		##########################set marked point to 0
		if mark_num!=0:imarr01[np.where(mask==1)]=0
		if mark_num==0:break

	return imarr01[1:33,1:33]
		
		
				
		
	

			
def num_pattern_0to1(p_nb):
	sp=0
	'''
	for i in range(10)[3:]:#[0,...9]   [3,...9]
		if p_nb[i]-p_nb[i-1]==1:
			sp+=1
	'''
	if p_nb[3]==1 and p_nb[2]==0:sp+=1
	if p_nb[4]==1 and p_nb[3]==0:sp+=1
	if p_nb[5]==1 and p_nb[4]==0:sp+=1
	if p_nb[6]==1 and p_nb[5]==0:sp+=1
	if p_nb[7]==1 and p_nb[6]==0:sp+=1
	if p_nb[8]==1 and p_nb[7]==0:sp+=1
	if p_nb[9]==1 and p_nb[8]==0:sp+=1
	
	if p_nb[2]==1 and p_nb[9]==0:sp+=1
	return sp
					
				
					 
					
 
					
				
	 
	 
	 	 
				
def save_feats(featlist):#[w_floor1,b1,w2,b2]
	featpath='/home/yr/chinese_char/xihua'
	write_file=open(featpath,'wb')
	for feat in featlist:
		cPickle.dump(feat,write_file,-1)
    	write_file.close()		
	


if __name__=='__main__':
	arr=loaddata_32x32() #gray->array
	 
	arr01=grayto01(arr) #gray_array->01 pic  1=charactor 0=background
	 

	 
	 
	for i in range(arr01.shape[0]):
		imarr01=arr01[i,:,:]
		arr_xh=try_xihua(imarr01) 
		im=Image.fromarray(np.uint8(arr_xh*255.))
		im.save('/home/yr/chinese_char/xihua/xihua_zhang_splited/'+str(i)+'.jpg') #show effect of 01picture
	 
	 
	

 
		


