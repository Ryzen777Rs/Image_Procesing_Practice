# training/src/test_screen.py
import os
import cv2
import numpy as np
import onnxruntime as ort
import mss
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(current_dir, "..", "model.onnx"))

if not os.path.exists(model_path):
    print(f"⚠️ EROARE: Nu am găsit model.onnx!")
    exit()

# Setăm 4 fire de execuție. Elimina ambuteiajul pe procesor (Thread Contention)
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4  
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session = ort.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])

sct = mss.MSS()
monitor = sct.monitors[1] 

WINDOW_NAME = "Diagnosticare Model"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)

prev_time = time.time()

while True:
    sct_img = sct.grab(monitor)
    frame = np.array(sct_img, dtype=np.uint8)[:, :, :3]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    
    orig_h, orig_w, _ = frame.shape
    input_w, input_h = 800, 800
    img_resized = cv2.resize(frame, (input_w, input_h))

    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_data = img_rgb.astype(np.float32) / 255.0
    img_data = np.transpose(img_data, (2, 0, 1))

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_data})
    
    boxes, labels, scores = outputs[0], outputs[1], outputs[2]

    if len(boxes.shape) == 3:
        boxes = boxes[0]
        labels = labels[0]
        scores = scores[0]

    for box, label, score in zip(boxes, labels, scores):
        if score > 0.4:
            # 🔴 PRINTĂM DATELE BRUTE DIN MODEL!
            print(f"📦 RAW BOX: {box} | Scor: {score:.2f} | Clasa: {label}")
            
            # Ne întoarcem la formula standard PyTorch (simplă, fără deducții)
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
            
            xmin = int(x1 * (orig_w / input_w))
            ymin = int(y1 * (orig_h / input_h))
            xmax = int(x2 * (orig_w / input_w))
            ymax = int(y2 * (orig_h / input_h))
            
            # Desenăm direct ce am calculat
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"C:{label}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()