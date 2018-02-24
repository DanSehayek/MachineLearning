import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
Goal: To write a neural network that will classify handwritten digits
      from the MNIST dataset with an accuracy of at least 95%

input layer > weights > hidden layer 1 > weights > hidden layer 2 > weights
            > hidden layer 3 > weights > output layer

The input layer contains 784 nodes (one for each pixel)
The output layer contains 10 nodes (one for each number)
'''

# Handwritten digits dataset
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# 3 hidden layers each with 500 nodes
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# 10 classes and 100 training examples per minibatch
n_classes = 10
batch_size = 100

# Define input tensor x and label y
# 28 pixels x 28 pixels = 784 pixels
x = tf.placeholder("float",[None,784])
y = tf.placeholder("float")

def neural_network_model(data):

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
    # x is input data
    # neural_network_model will return array of 10 elements
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))

    # We will use stochastic gradient descent to minimize the cost function
    # Optional parameter is learning rate (learning rate = 0.001)
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    epochs = 10

    # One cycle of forward propagation and back propagation is one epoch
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

        # How many times did the label equal the prediction?
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,"float"))
        print ("Accuracy: {0}".format(accuracy.eval({x:mnist.test.images,y:mnist.test.labels})))

train_neural_network(x)
