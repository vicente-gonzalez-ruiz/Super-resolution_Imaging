import tensorflow as tf

def weight(shape, name):
    w = tf.random.truncated_normal(shape,stddev=0.01)
    return tf.Variable(w, name=name)

def bias(shape , name):
    b = tf.compat.v1.constant(0.01, shape=shape)
    return tf.Variable(b, name=name)

def conv2d(x, w):
    return tf.nn.conv2d(x, w , strides=[1,1,1,1], padding= 'SAME')

def resnet(image):

    with tf.compat.v1.variable_scope("generator"):

        #convolutional l1
        
        W1 = weight([9, 9, 3, 64], name="W1");
        b1 = bias([64], name="b1");
        c1 = tf.nn.relu(conv2d(image, W1) + b1)

        #residual l2
        
        W2 = weight([3, 3, 64, 64], name="W2");
        b2 = bias([64], name="b2");
        c2 = tf.nn.relu(_inst_norm(conv2d(c1, W2) + b2))

        W3 = weight([3, 3, 64, 64], name="W3");
        b3 = bias([64], name="b3");
        c3 = tf.nn.relu(_inst_norm(conv2d(c2, W3) + b3)) + c1

        #residual l3
        
        W4 = weight([3, 3, 64, 64], name="W4");
        b4 = bias([64], name="b4");
        c4 = tf.nn.relu(_inst_norm(conv2d(c3, W4) + b4))

        W5 = weight([3, 3, 64, 64], name="W5");
        b5 = bias([64], name="b5");
        c5 = tf.nn.relu(_inst_norm(conv2d(c4, W5) + b5)) + c3

        #residual l4
        
        W6 = weight([3, 3, 64, 64], name="W6");
        b6 = bias([64], name="b6");
        c6 = tf.nn.relu(_inst_norm(conv2d(c5, W6) + b6))

        W7 = weight([3, 3, 64, 64], name="W7");
        b7 = bias([64], name="b7");
        c7 = tf.nn.relu(_inst_norm(conv2d(c6, W7) + b7)) + c5

        #residual l5
        
        W8 = weight([3, 3, 64, 64], name="W8");
        b8 = bias([64], name="b8");
        c8 = tf.nn.relu(_inst_norm(conv2d(c7, W8) + b8))
        #residual l6 
        W9 = weight([3, 3, 64, 64], name="W9");
        b9 = bias([64], name="b9");
        c9 = tf.nn.relu(_inst_norm(conv2d(c8, W9) + b9)) + c7

        #residual l7
        
        W10 = weight([3, 3, 64, 64], name="W10");
        b10 = bias([64], name="b10");
        c10 = tf.nn.relu(conv2d(c9, W10) + b10)

        #convolutional_l

        W11 = weight([3, 3, 64, 64], name="W11");
        b11 = bias([64], name="b11");
        c11 = tf.nn.relu(conv2d(c10, W11) + b11)
		

        #output_layer

        W12 = weight([9, 9, 64, 3], name="W12");
        b12 = bias([3], name="b12");
        prediction = tf.nn.tanh(conv2d(c11, W12) + b12) * 0.58 + 0.5

    return prediction

        
def _inst_norm(net):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)
    scale = tf.Variable(tf.ones(var_shape))
    shift = tf.Variable(tf.zeros(var_shape))
    
    epsilon = 1e-3
    normalized = (net - mu) / (sigma_sq + epsilon) ** (.5)
    return scale * normalized + shift
    
