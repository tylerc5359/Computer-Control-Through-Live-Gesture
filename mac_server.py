#pip3 install opencv-python flask
#export CAMERA_HOST=<mac-tailscale-ip>  CAMERA_TOKEN=<some-long-random-string>
#python3 mac_server.py
import hmac
import os
import sys

import cv2
from flask import Flask, Response, abort, request

HOST = os.environ.get("CAMERA_HOST", "127.0.0.1")
PORT = int(os.environ.get("CAMERA_PORT", "5001"))
TOKEN = os.environ.get("CAMERA_TOKEN")

if not TOKEN:
    sys.exit("Set CAMERA_TOKEN before starting the server.")

app = Flask(__name__)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video')
def video():
    if not hmac.compare_digest(request.args.get("token", ""), TOKEN):
        abort(401)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host=HOST, port=PORT, threaded=True)
