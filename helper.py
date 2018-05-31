import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))
        
def preprocess_labels_lyft(label_image):
    # Create a new single channel label image to modify
    labels_new = np.copy(label_image[:,:,0])
    # Identify lane marking pixels (label is 6)
    lane_marking_pixels = (label_image[:,:,0] == 6).nonzero()
    # Set lane marking pixels to road (label is 7)
    labels_new[lane_marking_pixels] = 7

    # Identify all vehicle pixels
    vehicle_pixels = (label_image[:,:,0] == 10).nonzero()
    # Isolate vehicle pixels associated with the hood (y-position > 496)
    hood_indices = (vehicle_pixels[0] >= 496).nonzero()[0]
    hood_pixels = (vehicle_pixels[0][hood_indices], \
                   vehicle_pixels[1][hood_indices])
    # Set hood pixel labels to 0
    labels_new[hood_pixels] = 0
    # Return the preprocessed label image 
    return labels_new

def gen_batch_function_lyft(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        
        short = False
        
        training_path = 'CameraRGB' if not short else 'CameraRGB_short'
        gt_path = 'CameraSeg' if not short else 'CameraSeg_short'
        
        imcount = 1000 if not short else 10
        
        imidxs = [i for i in range(imcount)]
        
        random.shuffle(imidxs)
        for batch_i in range(0, len(imidxs), batch_size):
            images = []
            gt_images = []

            
            for idx in imidxs[batch_i:batch_i+batch_size]:
                gt_image_file = "{}/{}/{}.png".format(data_folder, gt_path, idx)
                image_file    = "{}/{}/{}.png".format(data_folder, training_path, idx)

                print('{} '.format(idx), end='')

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape )
                
                gt_image_orig = scipy.misc.imread(gt_image_file)
                gt_image = scipy.misc.imresize( preprocess_labels_lyft(gt_image_orig) , image_shape, interp='nearest')

                roadsgt = gt_image == 7
                carsgt = gt_image == 10
                backgt = np.logical_not(carsgt, roadsgt)
                
                #scipy.misc.imsave( './seg.png', np.where(roadsgt==True, np.full(image_shape, 250), np.full(image_shape, 0) ))
                if False:
                    scipy.misc.imsave( './orig.png', image)
                    scipy.misc.imsave( './seg0.png', gt_image)
                    scipy.misc.imsave( './seg.png', roadsgt )
                
                roadsgt__= roadsgt.reshape(*roadsgt.shape, 1)
                carsgt__ = carsgt.reshape(*carsgt.shape, 1)
                backgt__ = backgt.reshape(*backgt.shape, 1)
                
                gt_image = np.concatenate((roadsgt__, carsgt__, backgt__) , axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn

def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([255, 0, 0])

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            
            for image_file in image_paths[batch_i:batch_i+batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
                gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)

                gt_bg = np.all(gt_image == background_color, axis=2)
                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def gen_test_output_lyft(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    
    for image_file in glob(os.path.join(data_folder, 'CameraRGB', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        
        def paste_mask(class_number, im_softmax, color, image):
            sm = im_softmax[0][:, class_number].reshape(image_shape[0], image_shape[1])
            segmentation = (sm > 0.5).reshape(image_shape[0], image_shape[1], 1)
            mask = np.dot(segmentation, np.array([color]))
            mask = scipy.misc.toimage(mask, mode="RGBA")
            
            image.paste(mask, box=None, mask=mask)
        
        street_im = scipy.misc.toimage(image)
        paste_mask(0, im_softmax, [0,255,0,127], street_im)
        paste_mask(1, im_softmax, [255,0,0,127], street_im)
        paste_mask(2, im_softmax, [0,0,255,127], street_im)
        
        
        

        yield os.path.basename(image_file), np.array(street_im)


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)

def save_inference_samples(dataset, runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    data_dir = os.path.join(data_dir, 'testing') if dataset=='kitty' else data_dir
    
    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    if dataset=='kitty':
        image_outputs = gen_test_output( sess, logits, keep_prob, input_image, data_dir, image_shape)
    else:
        image_outputs = gen_test_output_lyft( sess, logits, keep_prob, input_image, data_dir, image_shape)
        
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)
