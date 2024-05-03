import socket
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision.transforms as transforms
from POSTER_V2.models.PosterV2_7cls import *
from POSTER_V2.data_preprocessing.sam import SAM
from POSTER_V2.main import RecorderMeter, RecorderMeter1
import torch
import os

# Load the emotion recognition model
def load_emotion_recognition_model():
    model = pyramid_trans_expr2(img_size=224, num_classes=7)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load('/home/erthax/Programming/video-emotion-recognition/POSTER_V2/models/pretrain/affectnet-7-model_best.pth')
    model.load_state_dict(checkpoint['state_dict'])
    return model

# Socket setup
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host = '0.0.0.0'
port = 8080
server_socket.bind((host, port))
server_socket.listen(1)
print('Server listening on port', port)

# Emotion recognition and socket communication
emo_model = load_emotion_recognition_model()
my_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
mtcnn = MTCNN(select_largest=True)
vid = cv2.VideoCapture(0)

client_socket, addr = server_socket.accept()
print('Got a connection from', addr)

while True:
    ret, frame = vid.read()
    if not ret:
        print("Failed to grab frame")
        break
    frame_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    found_face = mtcnn(frame_rgb)
    if found_face is not None:
        transformed_face = my_transforms(found_face)
        tensor_unsqueezed = transformed_face.unsqueeze(0).cuda()
        result_emotion = emo_model(tensor_unsqueezed)
        _, max_emotion_index = result_emotion.max(dim=1)
        emotion = max_emotion_index.item()
        data_to_send = '0' if emotion == 1 else '1'
        client_socket.send(data_to_send.encode())

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

client_socket.close()
vid.release()
cv2.destroyAllWindows()
server_socket.close()
