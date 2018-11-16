r"""Convert raw WIDERFACE dataset to TFRecord for object_detection.

Example usage:
    python create_widerface_tf_record.py \
        --input_dir=/home/user/wider_face \
        --output_dir=/home/user/wider_face.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import os
import cv2
import logging

import tensorflow as tf

from object_detection.utils import dataset_util

logger = logging.getLogger()

flags = tf.app.flags

# tf.flags.DEFINE_boolean('merge_datasets', False,
#                         'Whether to merge train, validation and test '
#                         'datasets into single one. default: False')
tf.flags.DEFINE_boolean('resize', False,
                        'Whether to resize input images default: False')
tf.flags.DEFINE_integer('resize_height', 300,
                        'Height of resized images. default: 300')
tf.flags.DEFINE_integer('resize_width', 300,
                        'Width of resized images. default: 300')
tf.flags.DEFINE_string('input_dir', '',
                       'Directory with downloaded Wider Face datasets.')
tf.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')

tf.flags.DEFINE_string('log_file', '.\\conversion_log.txt',
                       'Path to log file. default: .\\conversion_log.txt')

FLAGS = flags.FLAGS


def create_tf_test_example(filename, images_path):
  height = None  # Image height
  width = None  # Image width
  encoded_image_data = None  # Encoded image bytes
  image_format = b'jpeg'  # b'jpeg' or b'png'

  if not filename:
    raise IOError()

  filepath = os.path.join(images_path, filename)

  image_raw = cv2.imread(filepath)

  height, width, channel = image_raw.shape

  if FLAGS.resize:
    image_resize = cv2.resize(image_raw, (FLAGS.resize_width, FLAGS.resize_height))
    path_pre = os.path.split(filename)[0]

    path = os.path.join(resize_image_dir, path_pre)
    if not os.path.exists(path):
      os.mkdir(path)

    output_img = os.path.join(resize_image_dir, filename)
    cv2.imwrite(output_img, image_resize)
    with open(output_img, 'rb') as ff:
      encoded_image_data = ff.read()
  else:
    with open(filepath, 'rb') as ff:
      encoded_image_data = ff.read()

  encoded_image_data = open(filepath, "rb").read()
  key = hashlib.sha256(encoded_image_data).hexdigest()

  tf_example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(int(height)),
    'image/width': dataset_util.int64_feature(int(width)),
    'image/filename': dataset_util.bytes_feature(filename.encode('utf-8')),
    'image/source_id': dataset_util.bytes_feature(filename.encode('utf-8')),
    'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
    'image/encoded': dataset_util.bytes_feature(encoded_image_data),
    'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
  }))

  return tf_example


def create_tf_example(filename, images_dir, f):
  """
  """
  xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = []  # List of normalized right x coordinates in bounding box (1 per box)
  ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = []  # List of normalized bottom y coordinates in bounding box (1 per box)
  classes_text = []  # List of string class name of bounding box (1 per box)
  classes = []  # List of integer class id of bounding box (1 per box)
  poses = []
  truncated = []
  difficulties = []

  # filename = f.readline().rstrip()
  #tf.logging.info(filename)
  filepath = os.path.join(images_dir, filename)
  #logging.info(filepath)
  image_raw = cv2.imread(filepath)
  height, width, channel = image_raw.shape
  #logging.info("height is %d, width is %d, channel is %d" % (height, width, channel))

  if FLAGS.resize:
    image_resize = cv2.resize(image_raw, (FLAGS.resize_width, FLAGS.resize_height))
    path_pre = os.path.split(filename)[0]

    path = os.path.join(resize_image_dir, path_pre)
    if not os.path.exists(path):
      os.mkdir(path)

    output_img = os.path.join(resize_image_dir, filename)
    cv2.imwrite(output_img, image_resize)
    with open(output_img, 'rb') as ff:
      encoded_image_data = ff.read()
  else:
    with open(filepath, 'rb') as ff:
      encoded_image_data = ff.read()

  key = hashlib.sha256(encoded_image_data).hexdigest()
  face_num = int(f.readline().rstrip())
  valid_face_num = 0

  for i in range(face_num):
    annot = f.readline().rstrip().split()
    # WIDER FACE DATASET CONTAINS SOME ANNOTATIONS WHAT EXCEEDS THE IMAGE BOUNDARY
    if float(annot[2]) > 25.0:
      if float(annot[3]) > 30.0:
        xmins.append(max(0.005, (float(annot[0]) / width)))
        ymins.append(max(0.005, (float(annot[1]) / height)))
        xmaxs.append(min(0.995, ((float(annot[0]) + float(annot[2])) / width)))
        ymaxs.append(min(0.995, ((float(annot[1]) + float(annot[3])) / height)))
        classes_text.append('face'.encode('utf8'))
        classes.append(1)
        poses.append("front".encode('utf8'))
        truncated.append(int(0))
        #logging.info(xmins[-1], ymins[-1], xmaxs[-1], ymaxs[-1], classes_text[-1], classes[-1])
        valid_face_num += 1


  #tf.logging.info("Face Number is %d" % face_num)
  #tf.logging.info("Valid face number is %d" % valid_face_num)

  tf_example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(int(height)),
    'image/width': dataset_util.int64_feature(int(width)),
    'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
    'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
    'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
    'image/encoded': dataset_util.bytes_feature(encoded_image_data),
    'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
    'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
    'image/object/class/label': dataset_util.int64_list_feature(classes),
    'image/object/difficult': dataset_util.int64_list_feature(int(0)),
    'image/object/truncated': dataset_util.int64_list_feature(truncated),
    'image/object/view': dataset_util.bytes_list_feature(poses),
  }))

  return valid_face_num, tf_example


def create_widerface_tf_record(annotations_file,
                               images_path,
                               output_path,
                               nobbox=False):

  logger.info('Processing %s dataset', images_path)
  logger.info('Annotation file is %s', annotations_file)

  writer = tf.python_io.TFRecordWriter(output_path)
  with open(annotations_file) as f:
    valid_image_num = 0
    invalid_image_num = 0
    # each picture start with filename, use for loop to get filename, other arg use readline fun to read
    processed_num = 0
    while True:
      next_line = f.readline()
      if not next_line:
        break

      filename = next_line.strip()

      if nobbox:
        tf_example = create_tf_test_example(filename, images_path)
      else:
        valid_face_number, tf_example = create_tf_example(filename, images_path, f)
        if valid_face_number != 0:
          valid_image_num += 1
        else:
          invalid_image_num += 1
          logger.info("%s image has no bboxes!"%filename)

      writer.write(tf_example.SerializeToString())

      processed_num += 1
      if processed_num % 1000 == 0:
        logger.info('%d images processed'%processed_num)

  writer.close()

  logger.info('Output file %s created. Total images processed %d'
               %(output_path, processed_num))
  logger.info("Valid image number is %d"%valid_image_num)
  logger.info("Invalid image number is %d"%invalid_image_num)

def init_logger():
  logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

  fileHandler = logging.FileHandler(FLAGS.log_file)
  fileHandler.setFormatter(logFormatter)
  logger.addHandler(fileHandler)

  consoleHandler = logging.StreamHandler()
  consoleHandler.setFormatter(logFormatter)
  logger.addHandler(consoleHandler)

  logger.setLevel(logging.INFO)


def main(_):

  assert FLAGS.input_dir, '`input_dir` missing.'
  assert FLAGS.output_dir, '`output_dir` missing.'

  print("input dir = {}".format(os.path.abspath(FLAGS.input_dir)))
  print("output dir = {}".format(os.path.abspath(FLAGS.output_dir)))
  print("log file = {}".format(os.path.abspath(FLAGS.log_file)))

  init_logger()

  if FLAGS.resize:
    global resize_image_dir
    resize_image_dir = os.path.join(FLAGS.output_dir, 'resized_images')

  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  test_image_dir = os.path.join(FLAGS.input_dir, 'WIDER_test/Images')
  val_image_dir = os.path.join(FLAGS.input_dir, 'WIDER_val/Images')
  train_image_dir = os.path.join(FLAGS.input_dir, 'WIDER_train/Images')

  train_annotations_file = os.path.join(FLAGS.input_dir, 'wider_face_split/wider_face_train_bbx_gt.txt')
  val_annotations_file = os.path.join(FLAGS.input_dir, 'wider_face_split/wider_face_val_bbx_gt.txt')
  test_annotations_file = os.path.join(FLAGS.input_dir, 'wider_face_split/wider_face_test_filelist.txt')

  train_output_path = os.path.join(FLAGS.output_dir, 'wider_train.tfrecord')
  val_output_path = os.path.join(FLAGS.output_dir, 'wider_val.tfrecord')
  test_output_path = os.path.join(FLAGS.output_dir, 'wider_test.tfrecord')

  create_widerface_tf_record(
    train_annotations_file,
    train_image_dir,
    train_output_path)
  create_widerface_tf_record(
    val_annotations_file,
    val_image_dir,
    val_output_path)
  create_widerface_tf_record(
    test_annotations_file,
    test_image_dir,
    test_output_path,
    nobbox=True)

  if FLAGS.resize:
    os.rmdir(resize_image_dir)

  logger.info('Conversion is finished!')

if __name__ == '__main__':
  tf.app.run()
