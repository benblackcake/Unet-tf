import numpy as np
import tensorflow as tf
import h5py
import cv2
import datetime
import os
import shutil


def build_log_dir(args, arguments):
    """Set up a timestamped directory for results and logs for this training session"""
    if args.name:
        log_path = args.name  # (name + '_') if name else ''
    else:
        log_path = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_path = os.path.join('results', log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    print('Logging results for this session in folder "%s".' % log_path)
    # Output csv header
    with open(log_path + '/loss.csv', 'a') as f:
        f.write(
            'iteration, ROC\n')
    # Copy this code to folder
    shutil.copy2('model.py', os.path.join(log_path, 'srresnet.py'))
    shutil.copy2('main.py', os.path.join(log_path, 'main.py'))
    shutil.copy2('utils.py', os.path.join(log_path, 'utils.py'))
    shutil.copy2('Benchmarks.py', os.path.join(log_path, 'Benchmarkss.py'))
    # Write command line arguments to file
    with open(log_path + '/args.txt', 'w+') as f:
        f.write(' '.join(arguments))
    return log_path


def get_data_set(path,label):
    f = h5py.File(path, 'r')
    data = f[label]
    return data

def batch_bgr2rgb(batch):

    for i in range(batch.shape[0]):
        batch[i,:,:,:] = cv2.cvtColor(batch[i,:,:,:], cv2.COLOR_BGR2RGB)

    return batch


def batch_bgr2gray(batch):

    batch_result = np.zeros([batch.shape[0],batch.shape[1],batch.shape[2]])
    for i in range(batch.shape[0]):
        batch_result[i,:,:] = cv2.cvtColor(batch[i,:,:,:], cv2.COLOR_BGR2GRAY)

    return batch_result

def modcrop(img, scale =2):
    """
    To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
    """
    # Check the image is gray

    if len(img.shape) ==3:
        h, w, _ = img.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        img = img[0:h, 0:w, :]
    else:
        h, w = img.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        img = img[0:h, 0:w]
    return img




