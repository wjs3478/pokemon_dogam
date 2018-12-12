#웹서버 라이브러리 선언
from flask import Flask, render_template, request, jsonify
#사물인식용
import imagenet as inet
#포켓몬용
import predict as pct
#플라스크 설정
app = Flask(__name__)

#루트에서 보여줄것
@app.route("/")
def hello():
    return render_template('index.html')

#사물인식 페이지 렌더
@app.route("/object_rec")
def object_render():
    #inet.get_predict("1.jpg")
    return render_template('object_rec.html')

#js렌더
@app.route("/upload.js")
def upload_js():
    #inet.get_predict("1.jpg")
    return render_template('upload.js')

#js렌더
@app.route("/upload2.js")
def upload2_js():
    #inet.get_predict("1.jpg")
    return render_template('upload2.js')

#사물인식 예측 실행
@app.route("/predict", methods=['GET', 'POST'])
def object_rec():
    data=inet.get_predict("porket.jpg")
    #print(data)
    return jsonify(data)

#포켓몬도감 페이지 렌더
@app.route("/poketmon", methods=['GET', 'POST'])
def poket_rec():
    return render_template('poket_dokam.html')

#포켓몬인식 예측 실행
@app.route("/dokam")
def poketmon_rec():
    data=pct.get_answer("porket.jpg")
    return jsonify(data)

#사진 파일 업로드
@app.route("/fileUpload", methods=['GET', 'POST'])
def upload_file():
   if request.method == 'POST':
      f = request.files['file']
      #저장할 경로 + 파일명
      #f.save(+secure_filename(f.filename))
      f.save('./porket.jpg')
      return 'uploads 디렉토리 -> 파일 업로드 성공!'

#웹서버 실행
if __name__ == '__main__':
   app.run(debug=True)