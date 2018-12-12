import tensorflow as tf
import sys

#훈련한 포켓몬 모델 파일
model_full_path = './tfr_dir/poketmon_image_graph.pb'
labels_full_path = './tfr_dir/poketmon_imagenet_map.txt'

#포켓몬 인식
def get_answer(image_path):
    #데이타 읽기
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # 패스에 있는 파일 라인별 재정립
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile(labels_full_path)]
    
    #빈라인 추가
    print()
    answer=[]
    # Unpersists graph from file
    with tf.gfile.FastGFile(model_full_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-3:][::-1]

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))
            answer.append(human_string+" (score = "+ str(round(score,5))+")")

    # get most likely classification
    #answer = label_lines[top_k[0]]
    return answer


if __name__ == '__main__':
    #이미지 가져오기, 메인으로 실행시
    test_image_path = sys.argv[1]

    ans = get_answer(test_image_path)
    print(ans)

