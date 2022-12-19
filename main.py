import streamlit as st
from PIL import Image
import torch

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

img_file=st.file_uploader('Upload Image',type=['png','jpg','jpeg'])

if img_file is not None:
    file_details={}
    file_details['type']=img_file.type
    file_details['size']=img_file.size
    file_details['name']=img_file.name
    st.write(file_details)
    # st.image(read_image(img_file))
    st.image(yolo(read_image(img_file)))
