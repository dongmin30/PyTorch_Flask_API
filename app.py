import os

from flask import Flask, render_template, request, redirect, jsonify

from common.inference import get_prediction
from common.commons import format_class_name

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # file validation check
        if 'file' not in request.files:
            return redirect(request.url)
        _file = request.files.get('file')
        if not _file:
            return
        # file 읽어오기
        img_bytes = _file.read()
        # 추론 결과 값 id, name 가져오기
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        # 추론 결과 값을 사람이 보기 편하도록 포매팅
        class_name = format_class_name(class_name)
        # 화면 전환
        return render_template('result.html', class_id=class_id, class_name=class_name)
    return render_template('index.html')

# JSON 값으로 반환 적용 시 사용하는 API
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        _file = request.files['file']
        img_bytes = _file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
