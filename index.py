#!/usr/bin/env python
from flask import Flask, render_template, Response, redirect, url_for, request
#import picamera
import cv2
import socket
import io
import os
import shutil

app = Flask(__name__)
vc = cv2.VideoCapture(0)


@app.route('/')
def index():
    return render_template('index.html',image_names = image_load())

def gen():
    """Video streaming generator function."""
    while True:
        rval, frame = vc.read()
        cv2.imwrite('stream.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('stream.jpg', 'rb').read() + b'\r\n')

def image_load():
    image_names = os.listdir('static/face-images/')
    return image_names


@app.route('/streaming')
def streaming():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



@app.route('/filename', methods = ['POST'])
def delete_target_image():
    filename = request.form['file_name']
    ilist = image_load()

    if os.path.isfile('static/face-images/'+ filename):
        filename = filename.split('-')
        ilist = image_load()
        for i in ilist:
            if filename[0] in i:
                os.remove('static/face-images/'+ i)

@app.route('/stop')
def stop():
    shutil.rmtree('static/face-images/')
    os.makedirs('static/face-images/')

"""
@app.route('/start')
def start():
    return render_template('index.html',s = 'start', image_names = image_load())

@app.route('/pause')
def pause():
    return render_template('index.html',s = 'pause', image_names = image_load())

@app.route('/stop')
def stop():
    return render_template('index.html',s = 'stop', image_names = image_load())
"""


if __name__ == '__main__':
        app.run(host='0.0.0.0', debug=True, threaded=True)