#!/usr/bin/env python
from flask import Flask, render_template, Response, redirect, url_for, request
#from picamera
import cv2
import socket
import io
import os
import shutil
import stream_model
import glob
from mtcnn.mtcnn import MTCNN


#기본설정
BASE_DIR = "static/facerec1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
image_path = glob.glob(os.path.join("static/facerec1/*.jpg"))

#탐지 모델
detector = MTCNN()
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#변수
PT = []
faceTrackers = dict()
ok_name = dict()
oks = dict()
crop_num = 0


app = Flask(__name__)
vc = cv2.VideoCapture(0)


@app.route('/')
def index():
    return render_template('index.html',image_names = image_load())

def gen():
    """Video streaming generator function."""
    while True:
        rval, frame = vc.read()
        cv2.flip(frame, 1)
        cv2.imwrite('cam.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('cam.jpg', 'rb').read() + b'\r\n')

def gen2():
    """Video streaming generator function."""
    model = stream_model.load_model()
    stream_model.predict()
    stream_model.livestream()
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + open('result-image.jpg', 'rb').read() + b'\r\n')


def image_load():
    image_names=[]
    image_ = os.listdir('static/train/')
    for i in image_ : 
        if '_' not in i : 
            image_names.append(i)
    return image_names


@app.route('/cam_streaming')
def cam_streaming():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/streaming')
def livestream():
    return Response(gen2(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/filename', methods = ['POST'])
def delete_target_image():
    filename = request.form['file_name']
    ilist = image_load()

    if os.path.isfile('static/train/'+ filename):
        filename = filename.split('_')
        ilist = image_load()
        for i in ilist:
            if filename[0] in i:
                os.remove('static/train/'+ i)


@app.route('/stop')
def stop():
    shutil.rmtree('static/facerec1/')
    shutil.rmtree('static/train/')
    os.makedirs('static/facerec1/')
    os.makedirs('static/train/')



@app.route('/camera')
def camera():
    stream_model.agree()
    stream_model.face_rec_crop()
    stream_model.preTreat(stream_model.PT)



if __name__ == '__main__':
        app.run(host='0.0.0.0', debug=True, threaded=True)