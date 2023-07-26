# FLASK 간단한 웹 서버 구현
import io
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

from flask import Flask, jsonify, request
import json
app = Flask(__name__)

@app.route('/')
def hello():
  return 'Hello World!'

# predict post API
@app.route("/predict", methods=['POST'])
def predict():
  if request.method == 'POST':
    # we will get the file the request
    _file = request.files['file']
    # convert that to bytes
    imh_bytes = _file.read()
    class_id, class_name = get_prediction(image_bytes=image_bytes)
    return jsonify({'class_id': class_id, 'class_name': class_name})
    
    
# 추론(Inference)
# 이미지 준비하기 - 244 x 244의 3채널 RGB이미지 - DenseNet 모델 사용 - 이밎지 텐서를 평균 및 표준편차 값으로 정규화
def transform_image(image_bytes):
  my_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(244),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                        [0.485, 0.456, 0.406],
                                        [0.229, 0.224, 0.225])])
  image = Image.open(io.BytesIO(image_bytes))
  return my_transforms(image).unsqueeze(0)

# 해당 메소드를 통해 이미지를 byte 단위로 읽은 후, 일련의 변환을 적용하여 Tensor를 반환
with open("_static\img\sample_file.jpeg", 'rb') as f:
  image_bytes = f.read()
  tensor = transform_image(image_bytes=image_bytes)
  print(tensor)
  
# 이미 학습된 가중치를 사용하기 위해 - torchvision 라이브러리의 모델을 사용하여 모델을 읽어온다
model = models.densenet121(weights='IMAGENET1K_V1')
# 모델을 추론에만 사용할 것이기 때문에 'eval' 모드로 변경
model.eval()

# 읽어온 모델을 통해 추론을 진행
def get_prediction(image_bytes):
  tensor = transform_image(image_bytes=image_bytes)
  outputs = model.forward(tensor)
  _, y_hat = outputs.max(1)
  return y_hat

# 사람이 읽을 수 있는 분류명이 있어야하기 때문에, 이를 위해 이름과 분류 ID를 매핑 하기 위해 해당 파일 불러오기
imagenet_class_index = json.load(open("_static\imagenet_class_index.json"))

# 위 json 파일 데이터를 인식 후 반환하기 위해 메소드 재정의
def get_prediction(image_bytes):
  tensor = transform_image(image_bytes=image_bytes)
  outputs = model.forward(tensor)
  _, y_hat = outputs.max(1)
  predict_idx = str(y_hat.item())
  return imagenet_class_index[predict_idx]

# 이미지를 보고 추론을 진행
with open("_static\img\sample_file.jpg", 'rb') as f:
  image_bytes = f.read()
  print(get_prediction(image_bytes=image_bytes))