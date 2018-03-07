#some basic imports and setups
import os
import cv2
import numpy as np
import tensorflow as tf
from alexnet import AlexNet
from caffe_classes import class_names
from tensorflow.python.platform import gfile

from tensorflow.python.framework import graph_util


#mean of imagenet dataset in BGR
imagenet_mean = np.array([104., 117., 124.], dtype=np.float32)

current_dir = os.getcwd()
image_dir = os.path.join(current_dir, 'images')

#get list of all images
img_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpeg')]

#load all images
imgs = []
for f in img_files:
    imgs.append(cv2.imread(f))
    
#placeholder for input and dropout rate
x = tf.placeholder(tf.float32, [None, 227, 227, 3])
# keep_prob = tf.placeholder(tf.float32)
keep_prob = tf.constant(1.0)

#create model with default config ( == no skip_layer and 1000 units in the last layer)
model = AlexNet(x, keep_prob, 1000, [])

#define activation of last layer as score
score = model.fc8

#create op to calculate softmax 
softmax = tf.nn.softmax(score)

with tf.Session() as sess:
    # Initialize all variables (no need, will be done by load_op)
    # sess.run(tf.global_variables_initializer())

    # Load the pretrained weights into the model
    load_op = model.load_initial_weights(sess)

    # run the load operator
    sess.run(load_op)

    # Loop over all images
    allimgs = np.zeros(shape=(3,227,227,3), dtype=np.float)

    for i, image in enumerate(imgs):

        # Convert image to float32 and resize to (227x227)
        img = cv2.resize(image.astype(np.float32), (227,227))

        # Subtract the ImageNet mean
        img -= imagenet_mean

        # Reshape as needed to feed into model
        img = img.reshape((1,227,227,3))


        # Run the session and calculate the class probability
        # probs = sess.run(softmax, feed_dict={x: img, keep_prob: 1})
        probs = sess.run(softmax, feed_dict={x: img})

        # Get the class name of the class with the highest probability
        class_name = class_names[np.argmax(probs)]

        # print prob
        print("Class: " + class_name + ", probability: %.4f" %probs[0,np.argmax(probs)])

        allimgs[i] = img.reshape(227,227,3)

    # make sure batch prediction works
    probs = sess.run(softmax, feed_dict={x: allimgs})
    print(probs.shape)

    # dump .pb file
    print("graph def size:", sess.graph_def.ByteSize())
    with gfile.GFile("alexnet.pb", 'wb') as f:
        f.write(sess.graph_def.SerializeToString())
    # convert graph to constants
    output_graph_def = graph_util.convert_variables_to_constants(
      sess,
      sess.graph_def,
      ["Softmax"])
    with gfile.GFile("alexnet_frozen.pb", 'wb') as f:
        f.write(output_graph_def.SerializeToString())

    print("graph def written to alexnet.pb and alexnet_frozen.pb")
