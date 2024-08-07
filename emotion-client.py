# Ran on PC

import requests
import os
import argparse
import cv2
import time
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

vid = cv2.VideoCapture(0)

desired_width = 1280
desired_height = 720
vid.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

fps = vid.get(cv2.CAP_PROP_FPS)
print("Camera Frame Rate:", fps)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
landmarks_and_emotion_video_writer = cv2.VideoWriter('landmarks&emotion_view.mp4', fourcc, 5.0, (desired_width, desired_height))
faces_video_writer = cv2.VideoWriter('faces_only.mp4', fourcc, 5.0, (224, 224))  # Assuming face crops are resized to 224x224


# IP address of your Raspberry Pis
HOST = 'my-portable-spell.local'  # Replace with the actual IP address of your Raspberry Pi
PORT = 65432  # Port to connect to the server

def send_led_command(state):
    url = 'http://'+HOST+':5000/led'
    data = {'state': state}
    response = requests.post(url, data=data)
    if response.status_code == 200:
        print(f'Success: {response.text}')
    else:
        print(f'Error: {response.text}')

model = load_emotion_recognition_model()
mtcnn = MTCNN(select_largest=True)
my_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])
emotions = ["Neutral", "Happy", "Sad", "Happy", "Fear", "Disgust", "Anger"]

try:
    while True:
        ret, frame = vid.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        found_face = mtcnn(frame_rgb)
        boxes, probs, landmarks = mtcnn.detect(frame_rgb, landmarks=True)


        if found_face is not None:
            transformed_face = my_transforms(found_face)
            tensor_unsqueezed = transformed_face.unsqueeze(0).cuda()
            tensor_repeated = tensor_unsqueezed.repeat(4, 1, 1, 1)
            result_emotion = model(tensor_repeated)
            emotion_sums = result_emotion.sum(dim=0)
            _, max_emotion_index = emotion_sums.max(dim=0)

            # Sending data to client
            message = '0' if emotions[max_emotion_index.item()] == "Happy" else '1'
            send_led_command(message)

            frame_rgb2 = frame_rgb.copy()
            draw = ImageDraw.Draw(frame_rgb)  # Create a drawing context
            draw2 = ImageDraw.Draw(frame_rgb2)

            if boxes is not None and landmarks is not None:
                for box, landmark in zip(boxes, landmarks):
                    draw2.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
                    draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)
                    if landmark is not None:
                        for point in landmark:
                            x, y = point
                            draw2.ellipse((x-5, y-5, x+5, y+5), outline=(0, 255, 0), width=3)
                            draw.ellipse((x-5, y-5, x+5, y+5), outline=(0, 255, 0), width=3)
                    x, y, w, h = box
                    text = emotions[max_emotion_index.item()]
                    # Optional: define a font
                    font = ImageFont.truetype("Roboto.ttf", 40)
                    draw2.text((x, y-50), text, fill=(0, 0, 255), font=font)

            frame_show = cv2.cvtColor(np.array(frame_rgb), cv2.COLOR_RGB2BGR)
            frame_show2 = cv2.cvtColor(np.array(frame_rgb2), cv2.COLOR_RGB2BGR)

            landmarks_and_emotion_video_writer.write(frame_show2)


            faces_frame = found_face.permute(1, 2, 0).numpy()
            faces_frame = ((faces_frame + 1) / 2.0) * 255.0
            if faces_frame.max() <= 1.0:
                faces_frame = (faces_frame * 255).astype(np.uint8)
            else:
                faces_frame = faces_frame.astype(np.uint8)
            faces_frame = cv2.resize(faces_frame, (224, 224))
            faces_frame = cv2.cvtColor(faces_frame, cv2.COLOR_RGB2BGR)
            faces_video_writer.write(faces_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    vid.release()
    cv2.destroyAllWindows()
