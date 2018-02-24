import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
Goal:
To write a simple multilayer perceptron/neural network that will classify
handwritten digits from the MNIST dataset with an accuracy of at least 95%

Structure:
input layer > weights > hidden layer 1 > weights > hidden layer 2 > weights
            > hidden layer 3 > weights > output layer

The input layer contains 784 nodes (one for each pixel)
The output layer contains 10 nodes (one for each number)
The activation value on each node of the input layer represents the greyscale
colour value of the corresponding pixel
'''

# Handwritten digits dataset
mnist = input_data.read_data_sets("/tmp/data/",one_hot = True)

# 3 hidden layers each with 500 nodes
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# 10 classes and 100 training examples per minibatch
n_classes = 10
batch_size = 100

# Define input tensor x and label y
x = tf.placeholder("float",[None,784])
y = tf.placeholder("float")

def neural_network_model(data):
    '''
    The activation values of the following layer are determined using a set
    of weights and biases that are initialized randomly:
    a2 = A(W * a1 + b)
    a1 is a vector containing the activation values of the preceding layer
    a2 is a vector containing the activation values of the current layer
    b is a vector containing the biases
    W is a matrix containing the weights
    A is an activation function that ensure that the activation values
    remain between 0 and 1. We will be using the ReLU activation function
    for our case.

    Purpose:
    This function will return the activation values of the output layer
    for a given set of input values/pixel values
    '''
    hidden_layer_1 = {"weights":tf.Variable(tf.random_normal([784,n_nodes_hl1])),
                      "biases":tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_layer_2 = {"weights":tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
                      "biases":tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_layer_3 = {"weights":tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
                      "biases":tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer = {"weights":tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
                    "biases":tf.Variable(tf.random_normal([n_classes]))}

    # (activations * weights) + biases
    l1 = tf.add(tf.matmul(data,hidden_layer_1["weights"]),hidden_layer_1["biases"])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_layer_2["weights"]),hidden_layer_2["biases"])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_layer_3["weights"]),hidden_layer_3["biases"])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer["weights"]) + output_layer["biases"]
    return output

def train_neural_network(x):
    '''
    Every time the neural network is given a new of set of input values,
    the cost or the sum of the squared differences between each of the
    expected values and actual output values will be computed.
    Backprogation and gradient descent will then be applied based on this
    cost in order to optimize the weights and biases so as to increase the
    network's confidence and correctness.

    Purpose:
    This function will compute the cost and apply gradient descent to determine
    the desired nudges over 100 training examples at a time. This process
    of applying the averaged desired changes over minibatchs as opposed to
    all of the training examples at once is known as stochastic gradient
    descent. While this descent is less careful it is also much faster.
    '''

    # x is input data
    # neural_network_model will return the corresponding activation
    # values of the output layer
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    epochs = 10

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for epoch in range(epochs):
            epoch_cost = 0
            for i in range(int(mnist.train.num_examples/batch_size)):
                # epoch_x is data
                # epoch_y is labels
                epoch_x,epoch_y = mnist.train.next_batch(batch_size)
                i,c = sess.run([optimizer,cost],feed_dict = {x:epoch_x,y:epoch_y})
                epoch_cost += c
            print ("Epoch {0} completed out of {1}. Cost: {2}".format(epoch,epochs,epoch_cost))

        # Now let us determine the accuracy of our current neural network
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,"float"))
        print ("Accuracy: {0}".format(accuracy.eval({x:mnist.test.images,y:mnist.test.labels})))

train_neural_network(x)
