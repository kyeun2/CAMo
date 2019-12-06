#!/usr/bin/env python
from flask import Flask, render_template, Response
#import picamera
import cv2
import socket
import io
import os

app = Flask(__name__)
vc = cv2.VideoCapture(0)

@app.route('/')
@app.route('/#start')
@app.route('/#stop')
def index():
    """image_load"""
    image_names = os.listdir('static/face-images/')
    return render_template('index.html', image_names=image_names)

def gen():
    """Video streaming generator function."""
    while True:
        rval, frame = vc.read()
        cv2.imwrite('stream.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('stream.jpg', 'rb').read() + b'\r\n')


@app.route('/streaming')
def streaming():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/#<filename>')
def delete_target_image(filename):
    if os.path.isfile('static/face-images/'+ filename):
        os.remove('static/face-images/'+ filename)
    return Response(index())


if __name__ == '__main__':
        app.run(host='0.0.0.0', debug=True, threaded=True)