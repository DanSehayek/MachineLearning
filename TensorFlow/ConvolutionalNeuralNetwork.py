import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

'''
Goal: To improve our classification accuracy of handwritten digits from the
MNIST dataset by using a convolutional neural network as opposed to a simple
multilayer perceptron
'''

n_classes = 10
batch_size = 128

x = tf.placeholder("float",[None,784])
y = tf.placeholder("float")

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x,W):
    '''
    Perform 2d convolution on a given set of input values.
    strides is set so that the feature window moves 1 pixel at a time
    '''
    return tf.nn.conv2d(x,W,strides = [1,1,1,1],padding = "SAME")

def maxpool2d(x):
    '''
    Perform 2d pooling on a given set of input values.
    ksize is set to make a 2 x 2 window
    strides is set so that the window moves 2 pixels at a time
    '''
    return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = "SAME")

def convolutional_neural_network(x):
    '''
    Convolutional Layer 1:
    - 5 x 5 convolution
    - Takes 1 input
    - Produces 32 feature maps
    Convolutional Layer 2:
    - 5 x 5 convolution
    - Takes 32 inputs
    - Produces 64 feature maps
    Fully Connected Layer:
    - Image at this point is 7 x 7 instead of 28 x 28
    - 64 feature maps of size 7 x 7
    - 1024 nodes

    Purpose:
    This function will return the activation values of the output layer
    for a given set of input values/pixel values
    '''

    weights = {"W_conv1":tf.Variable(tf.random_normal([5,5,1,32])),
               "W_conv2":tf.Variable(tf.random_normal([5,5,32,64])),
               "W_fc":tf.Variable(tf.random_normal([7*7*64,1024])),
               "out":tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {"b_conv1":tf.Variable(tf.random_normal([32])),
               "b_conv2":tf.Variable(tf.random_normal([64])),
               "b_fc":tf.Variable(tf.random_normal([1024])),
               "out":tf.Variable(tf.random_normal([n_classes]))}

    # We reshape a 784 pixel image to a 28 x 28 image
    x = tf.reshape(x, shape = [-1,28,28,1])

    conv1 = tf.nn.relu(conv2d(x,weights["W_conv1"]) + biases["b_conv1"])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1,weights["W_conv2"]) + biases["b_conv2"])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1,7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights["W_fc"]) + biases["b_fc"])
    fc = tf.nn.dropout(fc,keep_rate)

    output = tf.matmul(fc,weights["out"]) + biases["out"]
    return output

def train_neural_network(x):
    '''
    Only modification here is the first line where we call the CNN
    instead of our original multilayer perceptron
    '''
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(epochs):
            epoch_cost = 0
            for i in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                i,c = sess.run([optimizer,cost],feed_dict = {x: epoch_x, y: epoch_y})
                epoch_cost += c

            print ("Epoch {0} completed out of {1}. Cost: {2}".format(epoch,epochs,epoch_cost))

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,"float"))
        print("Accuracy:",accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)
