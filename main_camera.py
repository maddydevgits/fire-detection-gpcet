import streamlit as st
from PIL import Image
import torch
import cv2

model = torch.hub.load('ultralytics/yolov5', 'custom', 'best.pt')  # force_reload=True to update

def yolo(im, size=640):
    g = (size / max(im.size))  # gain
    im = im.resize((int(x * g) for x in im.size), Image.ANTIALIAS)  # resize
    results = model(im)  # inference
    print(results)
    results.render()  # updates results.ims with boxes and labels
    return Image.fromarray(results.ims[0])

def read_image(img):
    return Image.open(img)

st.title('Fire Detection System')
run=st.checkbox('Run Camera')
FRAME_WINDOW=st.image([])

while run:
    cam=cv2.VideoCapture(0)
    status,frame=cam.read()
    
    if status:
        cv2.imwrite('test.jpg',frame)
        FRAME_WINDOW.image(yolo(read_image(open('test.jpg','rb'))))