import io


from PIL import Image
from torchvision import models
import torchvision.transforms as transforms


def get_model():
    # torchvision에서 제공하는 densentnet121 모델 가져오기
    model = models.densenet121(pretrained=True)
    # 추론에만 사용하기 때문에 eval 모드로 변경 - 평가모드
    model.eval()
    return model


def transform_image(image_bytes):
    # 이미지 크기 변환 처리 및 정규화
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        # 평균 및 표준편차 값으로 정규화
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    # 바이트 단위로 변환 후 텐서로 반환 
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


# ImageNet classes are often of the form `can_opener` or `Egyptian_cat`
# will use this method to properly format it so that we get
# `Can Opener` or `Egyptian Cat`
def format_class_name(class_name):
    class_name = class_name.replace('_', ' ')
    class_name = class_name.title()
    return class_name
