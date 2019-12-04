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
def index():
    image_names = os.listdir('static/face-images/')
    print(image_names)
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

"""
@app.route("/upload", methods=["POST"])
def upload():
    target = os.path.join(APP_ROOT, 'face-images/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)

    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("complete.html", image_name=filename)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

"""
if __name__ == '__main__':
        app.run(host='0.0.0.0', debug=True, threaded=True)