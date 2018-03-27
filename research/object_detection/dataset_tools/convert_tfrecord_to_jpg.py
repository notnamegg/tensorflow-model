"""Convert TFRecord to jpegs for object_detection.

The output filename of jpeg will be:
<id>[<width>x<height>]Label_<lable>@[<xmin>,<ymin>]to[<xmax>,<ymax>].jpg
Example usage:
    python create_coco_tf_record.py --tfrecord=/path/to/your/tfrecord \
        --output_path=/where/you/want/to/save/jpgs

Reference: https://blog.csdn.net/miaomiaoyuan/article/details/56865361
"""

import os 
import tensorflow as tf 
from PIL import Image  
import matplotlib.pyplot as plt 
import numpy as np

flags = tf.app.flags
flags.DEFINE_string('tfrecord', '', 'filename for tfrecord.')
flags.DEFINE_string('output_filepath', '.', 'Path to output jpg')
FLAGS = flags.FLAGS

def number_of_jpeg(filename):
    number = 0
    for fn in filename:
      for record in tf.python_io.tf_record_iterator(fn):
         number += 1
    return number


#filename_queue = tf.train.string_input_producer(["train.record"]) 
def main(_):
    assert FLAGS.tfrecord, '`tfrecord` is missing.'
    filename=[FLAGS.tfrecord]
    number = number_of_jpeg(filename)
    filename_queue = tf.train.string_input_producer(filename) #read tfrecord
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image/height': tf.FixedLenFeature([], tf.int64),
                                           'image/width': tf.FixedLenFeature([], tf.int64),
                                           'image/object/bbox/xmin': tf.FixedLenFeature([], tf.float32),
                                           'image/object/bbox/xmax': tf.FixedLenFeature([], tf.float32),
                                           'image/object/bbox/ymin': tf.FixedLenFeature([], tf.float32),
                                           'image/object/bbox/ymax': tf.FixedLenFeature([], tf.float32),
                                           'image/object/class/label': tf.FixedLenFeature([], tf.int64),
                                           'image/encoded' : tf.FixedLenFeature([], tf.string),
                                       })  #get features
    image = tf.image.decode_jpeg(features['image/encoded'])
    #image = tf.reshape(image, [300, 300, 3])
    label = tf.cast(features['image/object/class/label'], tf.int32)
    if not os.path.exists(FLAGS.output_filepath):
        os.makedirs(FLAGS.output_filepath)
    with tf.Session() as sess: 
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        t_h = tf.cast(features['image/height'], tf.int32)
        t_w = tf.cast(features['image/width'], tf.int32)
        t_xm = tf.cast(features['image/object/bbox/xmin'], tf.float32)
        t_xM = tf.cast(features['image/object/bbox/xmax'], tf.float32)
        t_ym = tf.cast(features['image/object/bbox/ymin'], tf.float32)
        t_yM = tf.cast(features['image/object/bbox/ymax'], tf.float32)
        for i in range(number):
            #image = tf.reshape(image, [w, h, 3])
            example, l ,w,h,xm,xM,ym,yM= sess.run([image,label,t_w,t_h,t_xm,t_xM,t_ym,t_yM])
            img_data_jpg = tf.image.convert_image_dtype(example, dtype=tf.uint8) 
            encode_image_jpg = tf.image.encode_jpeg(img_data_jpg)
            folder = FLAGS.output_filepath
            #with tf.gfile.GFile(str(i)+'_'+str(w)+'x'+str(h)+'_'+'_'+'Label_'+str(l)+'_'+str(xm)+','+str(xM)+','+str(ym)+','+str(yM)+'_'+'.jpg', 'wb') as f:  
            with tf.gfile.GFile(folder+"/"+str(i)+'['+str(w)+'x'+str(h)+']'+'Label_'+str(l)+'@['+str(xm*w)+','+str(ym*h)+']to['+str(xM*w)+','+str(yM*h)+'].jpg', 'wb') as f:  
              f.write(encode_image_jpg.eval()) 
            print(str(l)+":"+str(w)+"x"+str(h))
        coord.request_stop()
        coord.join(threads)
    

if __name__ == "__main__":
    tf.app.run()
