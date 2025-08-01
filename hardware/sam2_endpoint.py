from PIL import Image
from flask import Flask, request, send_file
import numpy as np
import cv2
import io
import json
from sam2_predictor import SAM2Predictor

app = Flask(__name__)

predictor = SAM2Predictor()

@app.route('/sam2', methods=['POST'])
def sam2_endpoint():
    file = request.files['image']
    clicks = json.loads(request.form['clicks'])

    print(f"Received image shape: {cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR).shape}")
    print(f"Received clicks: {clicks}")

    # img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    # img = io.BytesIO(file.read())
    img = Image.open(file.stream).convert('RGB')  # Convert to RGB format
    img = np.array(img)  # Convert PIL image to numpy array
    
    result = predictor.run_sam2(img, clicks)  # 运行 SAM2 推理，返回掩码 np.ndarray

    # result['masks'] is np.ndarray of dtype float32 and shape (3, H, W), convert to cv2 format
    masks = (result['masks'] * 255).astype(np.uint8)
    masks = np.transpose(masks, (1, 2, 0))
    
    # 返回掩码（用 PNG 编码返回更安全）
    is_success, buffer = cv2.imencode(".png", masks)
    return send_file(io.BytesIO(buffer.tobytes()), mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)