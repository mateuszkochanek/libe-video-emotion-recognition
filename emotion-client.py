# Ran on PC

import socket
import os
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN, InceptionResnetV1
import torchvision.transforms as transforms

from POSTER_V2.models.PosterV2_7cls import *
from POSTER_V2.data_preprocessing.sam import SAM
from POSTER_V2.main import RecorderMeter, RecorderMeter1
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN
import torchvision.transforms as transforms
from POSTER_V2.models.PosterV2_7cls import pyramid_trans_expr2
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument('--resume', default='/home/erthax/Programming/video-emotion-recognition/POSTER_V2/models/pretrain/affectnet-7-model_best.pth', type=str, metavar='PATH', help='path to checkpoint')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')

parser.add_argument('--lr', '--learning-rate', default=0.000035, type=float, metavar='LR', dest='lr')
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()

def load_emotion_recognition_model():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model = pyramid_trans_expr2(img_size=224, num_classes=7)
    model = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.CrossEntropyLoss()

    if args.optimizer == 'adamw':
        base_optimizer = torch.optim.AdamW
    elif args.optimizer == 'adam':
        base_optimizer = torch.optim.Adam
    elif args.optimizer == 'sgd':
        base_optimizer = torch.optim.SGD
    else:
        raise ValueError("Optimizer not supported.")

    optimizer = SAM(model.parameters(), base_optimizer, lr=args.lr, rho=0.05, adaptive=False, )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    recorder = RecorderMeter(1)
    recorder1 = RecorderMeter1(1)

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            recorder = checkpoint['recorder']
            recorder1 = checkpoint['recorder1']
            best_acc = best_acc.to()
            model.load_state_dict(checkpoint['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    return model


# IP address of your Raspberry Pi
HOST = 'my-portable-spell.local'  # Replace with the actual IP address of your Raspberry Pi
PORT = 65432  # Port to connect to the server

def send_command(command):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        s.sendall(command.encode('utf-8'))

model = load_emotion_recognition_model()
mtcnn = MTCNN(select_largest=True)
my_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
emotions = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger"]

# Video capture setup
vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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
            tensor_repeated = tensor_unsqueezed.repeat(4, 1, 1, 1)
            result_emotion = model(tensor_repeated)
            emotion_sums = result_emotion.sum(dim=0)
            _, max_emotion_index = emotion_sums.max(dim=0)

            # Sending data to client
            message = '0' if emotions[max_emotion_index.item()] == "Happy" else '1'
            send_command(message)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    vid.release()
    cv2.destroyAllWindows()
