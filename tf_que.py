#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

def customOps(n):
	global plc 
	plc = tf.placeholder(tf.float32, shape=[n, n])

	# Tranpose of bottom right triangle
	mask = np.tril(np.ones((n,n)))
	mask = mask[:,::-1]
	T = (1-mask)*plc + (mask)*tf.transpose(plc)

	# calculating v1 and v2
	v1 = tf.reduce_sum(T, axis=1)
	v2 = tf.reduce_sum(T, axis=0)

	# softmax of [v1;v2]
	v_con = tf.nn.softmax(tf.concat([v1,v2],axis=0))
	max_index = tf.argmax(v_con)

	# result is a tensorflow boolean
	result = tf.greater(tf.cast(max_index, tf.float32), tf.constant(n/3, tf.float32))
	finalVal_true = tf.subtract(v1, v2)
	finalVal_false = tf.add(v1,v2)

	# evaluating boolean using lambda function
	finalVal = tf.cond(result, lambda:finalVal_true, lambda:finalVal_false)
	
	# return norm of finalVal
	return tf.reduce_sum(tf.square(finalVal))




if __name__ == '__main__':
	A = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
	assert A.shape[0] == A.shape[1]
	finalVal = customOps(3)

	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	outVal = sess.run(finalVal, feed_dict={plc:A})
	print(outVal)
	sess.close()
