#!/usr/bin/env python
# encoding=utf-8

'''
load gray 50x50 patch ->binary ->train cnn classifier->model param into pickle
'''

__docformat__='restructedtext en'

import os,sys,time,numpy,theano,cPickle
import theano.tensor as T
from theano.tensor.signal import downsample
from c10_mlp import HiddenLayer
from c10_logReg_sgd import load_data,LogisticRegression
from theano.tensor.nnet import conv

datapath0='/home/yr/computer_vision/train_data_nontext'
datapath1='/home/yr/computer_vision/train_data_text'
datapath00='/home/yr/computer_vision/test_data_nontext'
datapath11='/home/yr/computer_vision/test_data_text'
def shared_data(data_x,data_y):
	#print data_x.shape,data_y.shape
	shared_x=theano.shared(numpy.asarray(data_x,dtype=theano.config.floatX),borrow=True)
    	shared_y=theano.shared(numpy.asarray(data_y,dtype=theano.config.floatX),borrow=True)
    	return shared_x,T.cast(shared_y,'int32')

def get_imlist(path):
	return [os.path.join(path,f) for f in os.listdir(path) if \
		f.endswith('.jpg')]
def im2arr(data_0,data_1):#return [2w,50x50][2w,]
	list_1=get_imlist(data_1);print list_1.__len__()
	list_0=get_imlist(data_0);print list_0.__len__()
	from PIL import Image
	#datasets=[]
	 
	x1=[numpy.array(Image.open(im).convert('L')).flatten() for im in list_1];
	x0=[numpy.array(Image.open(im).convert('L')).flatten() for im in list_0];
	arr_x1=numpy.array([arr for arr in x1 if arr.shape[0]==2500]);print arr_x1.shape
	arr_x0=numpy.array([arr for arr in x0 if arr.shape[0]==2500]);print arr_x0.shape
	y0=[0 for i in range(arr_x0.shape[0])]
	y1=[1 for i in range(arr_x1.shape[0])]
	arr_y=numpy.array(y1+y0);arr_x=numpy.vstack((arr_x1,arr_x0));print 'x y',arr_y.shape,arr_x.shape #[4w,32x32] [4w,]
	return arr_x,arr_y
	
def load_data(arr_x,arr_y):
	
	print 'arr',arr_x.shape,arr_y.shape 
	x1_0,y1_0=shared_data(arr_x,arr_y)
	print 'shared',x1_0.eval().shape,y1_0.eval().shape
	#######
	
	return [x1_0,y1_0]
class LeNetConvPoolLayer(object):
	def __init__(self,rng,input,filter_shape,image_shape,poolsize=(2,2)):
		assert image_shape[1]==filter_shape[1]#[20,1,28,28] [20,1,9,9]
		self.input=input
		fan_in=numpy.prod(filter_shape[1:])
		fan_out=(filter_shape[0]*numpy.prod(filter_shape[2:])/numpy.prod(poolsize))
		W_bound=numpy.sqrt(6./(fan_in+fan_out))
		self.W=theano.shared(
				numpy.asarray(rng.uniform(low=-W_bound,high=W_bound,size=filter_shape),
						dtype=theano.config.floatX),
				borrow=True)
		b_values=numpy.zeros((filter_shape[0],),dtype=theano.config.floatX)
		self.b=theano.shared(value=b_values,borrow=True)
		conv_out=conv.conv2d(input=input,filters=self.W,filter_shape=filter_shape,image_shape=image_shape)
		pooled_out=downsample.max_pool_2d(input=conv_out,ds=poolsize,ignore_border=True)
		self.output=T.tanh(pooled_out+self.b.dimshuffle('x',0,'x','x'))
		self.params=[self.W,self.b]

def evaluate_lenet5(datasets,learning_rate=0.1,n_epochs=1000,nkerns=[3,7],batch_size=200):
	rng=numpy.random.RandomState(23455)
	#datasets=load_data(dataset)

	train_set_x,train_set_y=datasets[0]#[5w,784] [5w,]
	valid_set_x,valid_set_y=datasets[1]
	test_set_x,test_set_y=datasets[2]
	
	n_train_batches=train_set_x.get_value(borrow=True).shape[0]/batch_size
	n_valid_batches=valid_set_x.get_value(borrow=True).shape[0]/batch_size
	n_test_batches=test_set_x.get_value(borrow=True).shape[0]/batch_size

	##symbolic variable
	index=T.lscalar()
	x=T.matrix('x')
	y=T.ivector('y')
	
	print 'build the model'
	###expression graph
	layer0_input=x.reshape((batch_size,1,50,50))#tensor4 not list/tuple
	layer0=LeNetConvPoolLayer(
		rng,
		input=layer0_input,#tensor[20obs,1,height28,width28]
		image_shape=(batch_size,1,50,50),#list or tuple[20,1,28,28]
		filter_shape=(nkerns[0],1,5,5),
		poolsize=(2,2))
	
	layer1=LeNetConvPoolLayer(
		rng,
		input=layer0.output,#tensor [20,1,12,12]
		image_shape=(batch_size,nkerns[0],(50-5+1)/2,(50-5+1)/2),#list[20,1,23,23]
		filter_shape=(nkerns[1],nkerns[0],6,6),
		poolsize=(2,2))
	
	layer2_input=layer1.output.flatten(2)
	layer2=HiddenLayer(
		rng,
		input=layer2_input,
		n_in=nkerns[1]*( ( (50-5+1)/2 -6+1 )/2)**2,#9x9x7kernel
		n_out=50,
		activation=T.tanh)
	layer3=LogisticRegression(input=layer2.output,n_in=50,n_out=2)
	cost=layer3.negative_log_likelihood(y)
	
	###function compile
	test_model=theano.function(
			[index],
			layer3.errors(y),
			givens={x:test_set_x[index*batch_size:(index+1)*batch_size],
				y:test_set_y[index*batch_size:(index+1)*batch_size]})
	validate_model=theano.function(
			[index],
			layer3.errors(y),
			givens={x:valid_set_x[index*batch_size:(index+1)*batch_size],#[20obs,784]
				y:valid_set_y[index*batch_size:(index+1)*batch_size]})#[20obs,]
	#params=layer3.params+layer2.params+layer1.params+layer0.params
	params=layer3.params+layer2.params+layer1.params+layer0.params
	grads=T.grad(cost,params)
	updates=[(param_i,param_i-learning_rate*grad_i)
			for param_i,grad_i in zip(params,grads)]
	train_model=theano.function(
			[index],
			cost,
			updates=updates,
			givens={x:train_set_x[index*batch_size:(index+1)*batch_size],
				y:train_set_y[index*batch_size:(index+1)*batch_size]})
	print 'train the model'
	patience=10000
	patience_increase=2
	improvement_threshold=0.995
	validation_frequency=min(n_train_batches,patience/2)
	best_validation_loss=numpy.inf
	best_iter=0
	test_score=0.0
	start_time=time.clock()

	epoch=0
	done_looping=False
	
	while (epoch<n_epochs) and (not done_looping):
		epoch=epoch+1
		for minibatch_index in xrange(n_train_batches):
			iter=(epoch-1)*n_train_batches+minibatch_index
			#if iter%100==0:
				#print 'training iter= ',iter
			cost_ij=train_model(minibatch_index)
			if (iter+1)%500==0:#validation_frequency==0:
				validation_losses=[validate_model(i) for i in xrange(n_valid_batches)]
				this_validation_loss=numpy.mean(validation_losses)
				print ('epoch %i minibatch %i/%i validation error %f %%'%
					(epoch,minibatch_index+1,n_train_batches,
					this_validation_loss*100.))
				if this_validation_loss<best_validation_loss:
					if this_validation_loss<best_validation_loss*improvement_threshold:
						patience=max(patience,iter*patience_increase)
					###added by m
					save_params(params)#[hid w,hid b,logreg w,logreg b]
					###
					best_validation_loss=this_validation_loss
					best_iter=iter
					test_losses=[test_model(i)
							for i in xrange(n_test_batches)]
					test_score=numpy.mean(test_losses)
					print ('epoch %i minibatch %i/%i test error of best model%f %%'%
						(epoch,minibatch_index+1,n_train_batches,test_score*100.))
			if patience<iter:
				done_looping=True
				break
	end_time=time.clock()
	print ('optimization complete,best validation score%f obtained at iteration %i with test performance%f %%'%
		(best_validation_loss*100.,best_iter+1,test_score*100.))
	###added by m
	save_params(params)#[hid w,hid b,logreg w,logreg b]
	###

####added 
def save_params(paramlist):#[w_floor1,b1,w2,b2]
    parapath='/home/yr/computer_vision/cnn_textsegm_para'
    write_file=open(parapath,'wb')
    for parai in paramlist:
        cPickle.dump(parai.get_value(borrow=True),write_file,-1)
    write_file.close()
### 
def show_params():
    parapath='/home/yr/...'
    f=open(parapath,'rb')
    w1=cPickle.load(f)
    b1=cPickle.load(f)
    w2=cPickle.load(f)
    b2=cPickle.load(f)

    print np.shape(w1),np.shape(b1)
    print np.shape(w2),np.shape(b2)
    f.close()
#####
if __name__=='__main__':
	arrx,arry=im2arr(datapath0,datapath1)
	train_set=load_data(arrx,arry)#generate dataset [x,y] 
	arrx,arry=im2arr(datapath00,datapath11)
	test_set=load_data(arrx,arry)
	####
	
	evaluate_lenet5([train_set,train_set,test_set])
	#show_params()


	
			
















	

	
	
		

			
			

		


