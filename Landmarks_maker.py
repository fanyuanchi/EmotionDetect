from face_recognition import face_landmarks, load_image_file
from PIL import Image, ImageDraw


# 从图片中检测出人脸范围并返回68个关键点坐标列表
def face_landmark(image):
    face_landmarks_list = face_landmarks(image)
    if len(face_landmarks_list) == 0:
        print("No face in the present frame.")
        return []
    return face_landmarks_list[0]
    pass


# 在图片中检测人脸并绘制关键点坐标连线
def draw_landmark(image):
    # 查找图像中所有面部的所有面部特征
    face_landmarks = face_landmark(image)
    # 打印此图像中每个面部特征的位置
    facial_features = [
        'chin',
        'left_eyebrow',
        'right_eyebrow',
        'nose_bridge',
        'nose_tip',
        'left_eye',
        'right_eye',
        'top_lip',
        'bottom_lip'
    ]
    # 关键点分为“脸颊”、“左右眼眉”、“鼻梁”、“鼻尖”、“左右眼”、“上下唇”
    for facial_feature in facial_features:
        print("The {} in this face has the following points: {}".format(
            facial_feature, face_landmarks[facial_feature]))

    # 在图像中描绘出每个人脸特征
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)

    for facial_feature in facial_features:
        d.line(face_landmarks[facial_feature], width=5)

    pil_image.show()
    pass
