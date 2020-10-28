
import numpy as np
from scipy import misc
import cv2
import tensorflow as tf

class Benchmarks():

    def __init__(self,path_train, path_label, name):
        self.name = name
        # self.images_lr, self.names = self.load_images_by_model(model='LR')
        self.images_segment, self.names = self.load_images_by_model(path_train, model='test')
        # self.images_label, self.names = self.load_images_by_model(path_label, model='GT')

    def load_images_by_model(self, path, model, file_format='*'):
        """Loads all images that match '*_{model}.{file_format}' and returns sorted list of filenames and names"""
        # Get files that match the pattern
        filenames = glob.glob(os.path.join(path, '*_' + model + '.' + file_format))
        # Extract name/prefix eg: '/.../baby_LR.png' -> 'baby'
        names = [os.path.basename(x).split('_')[0] for x in filenames]
        return self.load_images(filenames), names


    def load_images(self, images):
        """Given a list of file names, return a list of images"""
        out = []
        for image in images:
            img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB).astype(np.uint8)
            out.append(img)

        return out


    def evaluate(self, sess, y_pred):

        pred = []

        for i, segment_img in enumerate(self.images_segment):


            pred_segment = sess.run([y_pred], feed_dict={'x:0':segment_img[np.newaxis]/255.})

            segment_result = np.squeeze(pred_segemet, axis=0)
            segment_result *= 255
            segment_result = segment_result.astype('uint8')

            pred.append(segment_result)
            
            pass














