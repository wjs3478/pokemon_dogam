from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 라이브러리 불러오기
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

#줄이기
FLAGS = tf.app.flags.FLAGS

#구글이 만든 Inception-v3 모델을 사용한다. 밑에는 파일들
#classify_image_graph_def : Graph의 이진 표현
#imagenet_synset_to_human_label_map.txt : 이미지넷을 인간이 읽을수 있는 문자로 맵핑
#imagenet_2012_challenge_label_map_proto.pbtxt: image_graph.pb와 imagenet_map.txt를 매핑


#Inception-v3 모델은 이미지 분류를 위한 모델이다
tf.app.flags.DEFINE_string('model_dir', './tfr_dir','모델 불러올 경로')
#읽을 이미지 파일의 경로를 설정
tf.app.flags.DEFINE_string('image_file', '','읽을 이미지 경로')
#이미지의 추론결과를 몇개까지 표시할 것인지 설정
tf.app.flags.DEFINE_integer('num_top_predictions', 3, '추론결과 갯수')


#정수 형태의 node ID를 인간이 이해할 수 있는 레이블로 변환
class NodeLookup(object):

  def __init__(self, label_lookup_path=None, uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """각각의 softmax node에 대해 인간이 읽을 수 있는 영어 단어를 로드 함.
    Args:
      label_lookup_path: 정수 node ID에 대한 문자 UID.
      uid_lookup_path: 인간이 읽을 수 있는 문자에 대한 문자 UID.
    Returns:
      정수 node ID로부터 인간이 읽을 수 있는 문자에 대한 dict.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    #  문자 UID로부터 인간이 읽을 수 있는 문자로의 맵핑을 로드함.
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # 문자 UID로부터 정수 node ID에 대한 맵핑을 로드함.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # 마지막으로 정수 node ID로부터 인간이 읽을 수 있는 문자로의 맵핑을 로드함.
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph():
  """저장된 GraphDef 파일로부터 그래프를 생성하고 저장된 값을 리턴함."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image(image):
  """이미지에 대한 추론을 실행
  Args:
    image: 이미지 파일 이름.
  Returns:
    없음(Nothing)
  """
  answer=[]
  if not tf.gfile.Exists(image):
    tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # 저장된 GraphDef로부터 그래프 생성
  create_graph()

  with tf.Session() as sess:
    # 사용 가능한 텐서:
    # 'softmax:0': 1000개의 레이블에 대한 정규화된 예측결과값(normalized prediction)을 포함하고 있는 텐서   
    # 'pool_3:0': 2048개의 이미지에 대한 float 묘사를 포함하고 있는 next-to-last layer를 포함하고 있는 텐서
    # 'DecodeJpeg/contents:0': 제공된 이미지의 JPEG 인코딩 문자를 포함하고 있는 텐서

    # image_data를 인풋으로 graph에 집어넣고 softmax tesnor를 실행한다.
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    #softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
    predictions = sess.run(softmax_tensor,
                           {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    # node ID --> 영어 단어 lookup을 생성한다.
    node_lookup = NodeLookup()

    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      answer.append(human_string+" (score = "+ str(round(score,5))+")")
      print('%s (score = %.5f)' % (human_string, score))
    
  return answer


def get_predict(img):
  image=img
  aws=run_inference_on_image(image)
  return aws
  #print(aws)

def main(argv=None):
  
  # 인풋으로 입력할 이미지를 설정한다.
  # image = (FLAGS.image_file if FLAGS.image_file else
  #          os.path.join(FLAGS.model_dir, 'cropped_panda.jpg'))
  image='cropped_panda.jpg'

  # 인풋으로 입력되는 이미지에 대한 추론을 실행한다.
  aws=run_inference_on_image(image)
  print(aws)


if __name__ == '__main__':
  tf.app.run()