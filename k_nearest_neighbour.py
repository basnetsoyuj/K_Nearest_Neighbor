import numpy as np
import matplotlib.pyplot as plt 
import time

## Load the training set
train_data = np.load('MNIST/train_data.npy')
train_labels = np.load('MNIST/train_labels.npy')

## Load the testing set
test_data = np.load('MNIST/test_data.npy')
test_labels = np.load('MNIST/test_labels.npy')

def show_digit(x):
    plt.axis('off')
    plt.imshow(x.reshape((28,28)), cmap=plt.cm.gray)
    plt.show()
    return

## Define a function that takes an index into a particular data set ("train" or "test") and displays that image.
def vis_image(index, dataset="train"):
    if(dataset=="train"): 
        show_digit(train_data[index,])
    else:
        show_digit(test_data[index,])
    return

def distance(x,y):
	dtance=np.sum(np.square(x-y))
	return dtance

def NN_distance(x):
	d_list=[distance(x,train_data[i]) for i in range(len(train_labels))]
	return np.argmin(d_list)

def do_or():
    success=0
    fail=0
    for x in range(len(test_data)):
        vvalue=NN_distance(test_data[x])
        value=train_labels[vvalue]
        act_value=test_labels[x]
        if (value==act_value):
            print('Success')
            print('Actual digit and identified digts is :',test_labels[x])
            vis_image(x,'test')
            vis_image(vvalue,'train')
        else:
            print('Failure')
            print('Actual digit {} but identified digts as {}'.format(test_labels[x],train_labels[vvalue]))
            vis_image(x,'test')
            vis_image(vvalue,'train')
do_or()
