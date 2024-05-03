import socket
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN
import torchvision.transforms as transforms
from POSTER_V2.models.PosterV2_7cls import pyramid_trans_expr2
import torch
import os

# Network setup
HOST = 'localhost'  # Server address
PORT = 65432        # Port to listen on (non-privileged ports are > 1023)
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen()

# Emotion recognition setup
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # Assuming single GPU, change if different setup
model = pyramid_trans_expr2(img_size=224, num_classes=7)
model = torch.nn.DataParallel(model).cuda()
model.load_state_dict(torch.load('path_to_model.pth')['state_dict'])
model.eval()

mtcnn = MTCNN(select_largest=True)
my_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
emotions = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]

# Video capture setup
vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Waiting for a client to connect...")
conn, addr = server_socket.accept()
print(f"Connected by {addr}")

try:
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
            result_emotion = model(tensor_unsqueezed)
            _, max_emotion_index = result_emotion.max(dim=1)

            # Sending data to client
            message = '0' if emotions[max_emotion_index.item()] == "Happy" else '1'
            conn.sendall(message.encode())

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    vid.release()
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()
