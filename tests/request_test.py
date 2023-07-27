# API 요청 테스트
import requests
resp = requests.post("http://localhost:5000/predict", files={"file": open('static\img\sample_crab.jpg','rb')})
print(resp.json())