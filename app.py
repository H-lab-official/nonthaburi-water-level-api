# app.py
from flask import Flask, jsonify, send_from_directory
from tanamnon import tamannon_capture_frame, image_save_path
from pakkred import pakkred_capture_frame, image_save_path

app = Flask(__name__)

@app.route('/', methods=['GET'])
def get_data():
    tamannon_data= tamannon_capture_frame()
    pakkred_data= pakkred_capture_frame()
    tamannon_json = tamannon_data.get_json()
    pakkred_json = pakkred_data.get_json()
    return jsonify({
          "status": "success",
          "data": [
              tamannon_json,
              pakkred_json
          ],
      })

@app.route('/images/<filename>', methods=['GET'])
def serve_image(filename):
    """Serve the saved images from the folder."""
    return send_from_directory(image_save_path, filename)

if __name__ == '__main__':
    app.run(debug=True)
