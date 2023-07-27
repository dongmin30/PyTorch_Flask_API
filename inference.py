import json

from commons import get_model, transform_image

model = get_model()
imagenet_class_index = json.load(open('imagenet_class_index.json'))

def get_prediction(image_bytes):
    try:
        # 받아온 이미지 파일을 텐서에 맞게 변환
        tensor = transform_image(image_bytes=image_bytes)
        # 변환된 텐서 값을 모델 순전파 인수로 사용
        outputs = model.forward(tensor)
    except Exception:
        return 0, 'error'
    # 추론 값 중 가장 확률이 높은 값을 사용
    _, y_hat = outputs.max(1)
    # 추론 데이터 인덱스 추출
    predicted_idx = str(y_hat.item())
    # 데이터인덱스에 해당되는 json id 값 반환
    return imagenet_class_index[predicted_idx]
