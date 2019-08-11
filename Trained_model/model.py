import tensorflow as tf

def weight(shape, name):
	w = tf.truncated_normal(shape,stddev=0.01)
	return tf.Variable(w, name=name)

def bias(shape , name):
	b = tf.constant(0.01, shape=shape)
	return tf.Variable(b, name=name)

def conv_2_d(x, w):
	return tf.nn.conv_2_d(x, w , strides=[1,1,1,1], padding= 'SAME')

def resnet(image):
	with tf.variable_scope("generator"):

		#convolutional l1
		w1 = weight([9,9,3,64], name="w1")
		b1 = bias([64], name="b1")
		c1 = tf.nn.relu(conv_2_d(image, w1) + b1)

		#residual l2
		w2 = weight([3,3,64,64], name="w2")
		b2 = bias([64], name="b2")
		c2 = tf.nn.relu(_inst_norm(conv_2_d(c1, w2) + b2))

		w3 = weight([3,3,64,64], name="w3")
		b3 = bias([64], name="b3")
		c3 = tf.nn.relu(_inst_norm(conv_2_d(c2, w3) + b3)) + c1

		#residual l3
		w4 = weight([3,3,64,64], name="w4")
		b4 = bias([64], name="b4")
		c4 = tf.nn.relu(_inst_norm(conv_2_d(c3, w4) + b4))

		w5 = weight([3,3,64,64], name="w5")
		b5 = bias([64] ,name="b5")
		c5 = tf.nn.relu(_inst_norm(conv_2_d(c4, w5) + b5)) + c3

		#residual l4
		w6 = weight([3, 3, 64, 64], name="w6")
        b6 = bias([64], name="b6")
        c6 = tf.nn.relu(_inst_norm(conv_2_d(c5, w6) + b6))

        w7 = weight([3, 3, 64, 64], name="w7")
        b7 = bias([64], name="b7")
        c7 = tf.nn.relu(_inst_norm(conv_2_d(c6, w7) + b7)) + c5

        # residual l4

        w8 = weight([3, 3, 64, 64], name="w8");
        b8 = bias([64], name="b8");
        c8 = tf.nn.relu(_inst_norm(conv_2_d(c7, w8) + b8))

        w9 = weight([3, 3, 64, 64], name="w9");
        b9 = bias([64], name="b9");
        c9 = tf.nn.relu(_inst_norm(conv_2_d(c8, w9) + b9)) + c7

        # convolutional layer

        w10 = weight([3, 3, 64, 64], name="w10");
        b10 = bias([64], name="b10");
        c10 = tf.nn.relu(conv_2_d(c9, w10) + b10)

        # convolutional layer

        w11 = weight([3, 3, 64, 64], name="w11");
        b11 = bias([64], name="b11");
        c11 = tf.nn.relu(conv_2_d(c10, W11) + b11)

        # Output layer

        w12 = weight([9, 9, 64, 3], name="w12");
        b12 = bias([3], name="b12");
        prediction = tf.nn.tanh(conv_2_d(c11, w12) + b12) * 0.58 + 0.5

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
	
