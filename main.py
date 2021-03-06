import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    # load model from disk
    tf.saved_model.loader.load( sess, [vgg_tag], vgg_path )

    # get tensor hooks and return them
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name( vgg_input_tensor_name )
    keep_prob = graph.get_tensor_by_name( vgg_keep_prob_tensor_name )
    layer3_out = graph.get_tensor_by_name( vgg_layer3_out_tensor_name )
    layer4_out = graph.get_tensor_by_name( vgg_layer4_out_tensor_name )
    layer7_out = graph.get_tensor_by_name( vgg_layer7_out_tensor_name )

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
    
# loading VGG
#tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    # we use an L2 regularizer to prevent weight overfitting. 
    # ...     kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3)) ...
    
    # 1x1 conv of vgg layer 7
    l7_out = tf.layers.conv2d( vgg_layer7_out, num_classes, 1, padding='same', 
            kernel_initializer = tf.random_normal_initializer(stddev=0.02),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3));

    # upsampling layer 7 output
    l4a_input_1 = tf.layers.conv2d_transpose( l7_out, num_classes, 4, strides=(2,2), padding='same',
            kernel_initializer = tf.random_normal_initializer(stddev=0.02),
            kernel_regularizer = tf.contrib.layers.l2_regularizer(1e-3));

    # 1x1 conv of vgg layer 4 from encoder
    l4a_input_2 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 
                                   padding= 'same', 
                                   kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    
    # connect layer 4 and previous layer 4 by adding them
    layer4dec_output = tf.add(l4a_input_1, l4a_input_2)

    

    # upsample layer4dec output
    l3dec_input_1 = tf.layers.conv2d_transpose(layer4dec_output, num_classes, 4, strides= (2, 2), padding= 'same', 
                                             kernel_initializer= tf.random_normal_initializer(stddev=0.02), 
                                             kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    
    # 1x1 conv of vgg layer 3 encoder output
    l3dec_input_2 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 
                                   padding= 'same', 
                                   kernel_initializer= tf.random_normal_initializer(stddev=0.02), 
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))


    # connect layer 3s by adding them
    layer3dec_output = tf.add(l3dec_input_1, l3dec_input_2)

    # upsample
    out_layer = tf.layers.conv2d_transpose(layer3dec_output, num_classes, 16, strides= (8, 8), padding= 'same', 
                                               kernel_initializer= tf.random_normal_initializer(stddev=0.02), 
                                               kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))

    return out_layer
#tests.test_layers(layers)

def lyft_score(y_true, y_pred, image_shape ):
    #true = tf.reshape(y_true, [-1, image_shape[0]* image_shape[1],3])
    truef = tf.cast(y_true, dtype=tf.float32)
    pred = y_pred; #tf.reshape(y_pred, [-1, image_shape[0]* image_shape[1],3])
    

    inte = tf.multiply(truef, pred)
    true_sum = tf.reduce_sum(truef,axis=0)
    pred_sum = tf.reduce_sum(pred,axis=0)
    inte_sum = tf.reduce_sum(inte,axis=0)

    precision = tf.divide( inte_sum, tf.add(pred_sum, 1))
    recall = tf.divide(inte_sum, tf.add(true_sum ,1))

    beta2_r = 0.5**2   #road 
    beta2_v = 2.0**2   #vehicle
    beta2 = tf.constant([beta2_r, beta2_v, 1.0])

    fscore_num = tf.multiply(tf.add(1.0, beta2), tf.multiply(precision, recall))
    fscore_den = tf.multiply(beta2, tf.add(precision, tf.add(recall, 1e-6)))

    fscore = tf.divide(fscore_num, fscore_den)

    avg_weights = tf.constant([0.5, 0.5, 0.0])

    favg = tf.reduce_sum(tf.multiply(avg_weights, fscore), axis=0)
    
    return favg

def lyft_score_loss(y_true, y_pred, image_shape ):
    return tf.subtract(1.0, lyft_score(y_true, y_pred, image_shape))

def optimize(nn_last_layer, correct_label, learning_rate, num_classes, adam_epsilon, image_shape ):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # TODO: Implement function

    
    # reshape the output and the ground truth labels
    logits = tf.reshape( nn_last_layer, (-1, num_classes), name="logits" )
    correct_label = tf.reshape( correct_label, (-1, num_classes) )
    
    # our loss function. cel = cross entropy loss
    #cel = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    cel = tf.reduce_mean(lyft_score_loss( correct_label, logits, image_shape ))
    
    # choose optimizer
    optimizer = tf.train.AdamOptimizer( learning_rate = learning_rate, epsilon = adam_epsilon )
    train_op = optimizer.minimize( cel )

    return logits, train_op, cel

#tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, adam_epsilon ):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    sess.run( tf.global_variables_initializer() )

    print("train start\n")
    for i in range(epochs):
        print("epoch {0}".format(i))
        c = 0
        for img, lab in get_batches_fn(batch_size):
           # writer = tf.summary.FileWriter('logs', sess.graph)
            tmp, loss = sess.run( [train_op, cross_entropy_loss], feed_dict={input_image: img, correct_label: lab, keep_prob: 0.5, learning_rate: 0.004, adam_epsilon: 0.1})
            print("{}:{}: loss {:.4f}".format(i, c, loss))
            c += 1
            #writer.close()
        
        
    

#tests.test_train_nn(train_nn)

image_shape = (128, 128)
data_dir = './data'
runs_dir = './runs'

def infer(dataset='kitty', overfit=False):
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('saved_models/vgg-model.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('saved_models'))
        
        graph = tf.get_default_graph()
        in_img = graph.get_tensor_by_name( 'image_input:0' )
        keep_prob = graph.get_tensor_by_name( 'keep_prob:0' )
        logits = graph.get_tensor_by_name( 'logits:0' )
        
        data_dir = 'data/lyft' if dataset=='lyft' else 'data/data_road'
        helper.save_inference_samples(dataset, runs_dir, data_dir, sess, image_shape, logits, keep_prob, in_img, overfit)


def train(dataset='kitty', overfit=False):
    num_classes = 2 if dataset=='kitty' else 3
    #tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches        
        if dataset=='lyft':
            get_batches_fn = helper.gen_batch_function_lyft('data/lyft', image_shape, overfit )
        else:
            get_batches_fn = helper.gen_batch_function('data/data_road/training', image_shape, overfit )
            

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        epochs, batch_size = (1, 5)

        correct_label = tf.placeholder( tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder( tf.float32, name='learning_rate')
        adam_epsilon  = tf.placeholder( tf.float32, name='adam_epsilon')

        #load vgg
        in_img, keep_prob, vgg_l3_out, vgg_l4_out, vgg_l7_out = load_vgg( sess, vgg_path )
        
        #build up decoder of FCN
        nn_last_layer = layers( vgg_l3_out, vgg_l4_out, vgg_l7_out, num_classes )

        # and optimize tensorflow
        logits, train_op, cel = optimize( nn_last_layer, correct_label, learning_rate, num_classes, adam_epsilon, image_shape )

        # TODO: Train NN using the train_nn function
        train_nn( sess, epochs, batch_size, get_batches_fn, train_op, cel, in_img, correct_label, keep_prob, learning_rate, adam_epsilon )


        # save model
        saver = tf.train.Saver()
        saver.save(sess, './saved_models/vgg-model')


        # TODO: Save inference data using helper.save_inference_samples
        #data_dir = 'data/lyft' if dataset=='lyft' else 'data/data_road'
        #helper.save_inference_samples(dataset, runs_dir, data_dir, sess, image_shape, logits, keep_prob, in_img)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    dataset = 'lyft'
    overfit = False
    train(dataset, overfit)
    infer(dataset, overfit)
    
    #f = helper.gen_batch_function_lyft('./data/lyft', image_shape)(5)
    #for a,b in f:
    #    continue
    
    
