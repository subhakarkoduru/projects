# from fastapi import FastAPI, WebSocket
# from fastapi.responses import FileResponse
# import cv2
# import numpy as np
# import os
# from starlette.websockets import WebSocketDisconnect
# import base64
# import json
# import ultralytics
# from ultralytics import YOLO

# app = FastAPI()

# certfile = 'localhost.pem'
# keyfile = 'localhost-key.pem'

# # Store connected clients
# connected_clients = set()

# model = YOLO(f'best.pt')


# @app.websocket("/video")
# async def video_stream(websocket: WebSocket):
#     await websocket.accept()
#     connected_clients.add(websocket)
#     print('WebSocket connection established:', websocket.client)

#     try:
#         while True:
#             message = await websocket.receive_bytes()  # Receive the WebSocket message as bytes
#             if message.startswith(b'\xff\xd8'):  # JPEG start marker
#                 # This appears to be a valid JPEG image
#                 frame = cv2.imdecode(np.frombuffer(message, np.uint8), cv2.IMREAD_COLOR)

#                 results = model([frame])
#                 for result in results:
#                     boxes = result.boxes  # Boxes object for bounding box outputs
                
#                 #cv2.imshow('Processed Image', frame)  # Show the processed image in a window
#                 #cv2.waitKey(1)
#                 # Process the frame here if needed
#                 # For example, you can display it, save it, or perform any other operation
#                 response_data = { 
#                     "rectangle": {"x1": 100, "y1": 100, "x2": 122, "y2": 122}
#                        }
#                 await websocket.send_text(json.dumps(boxes[0]))
#                 # Send the processed frame back to the client
#                 _, buffer = cv2.imencode('.jpg', frame)
#                 #await websocket.send_bytes(buffer.tobytes())
#             else:
#                 print('Received invalid image data: Not a JPEG image')
#                 continue
#     except WebSocketDisconnect:
#         pass
#     finally:
#         connected_clients.remove(websocket)


# @app.get("/")
# async def get_root():
#     file_path = os.path.join(os.path.dirname(__file__), "index.html")
#     return FileResponse(file_path)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, ssl_keyfile=keyfile, ssl_certfile=certfile)

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
import cv2
import numpy as np
import os
import asyncio
from starlette.websockets import WebSocketDisconnect
import json
from ultralytics import YOLO
import tensorflow as tf
import time

app = FastAPI()

certfile = 'localhost.pem'
keyfile = 'localhost-key.pem'

# Store connected clients
connected_clients = set()

#model = YOLO('best.pt')
interpreter = tf.lite.Interpreter(model_path="best_int8.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def preprocess_image(image_bytes):
    # Preprocess the image to fit your model's input requirements
    image = tf.image.decode_image(image_bytes, channels=3)
    image = tf.image.resize(image, [input_details[0]['shape'][1], input_details[0]['shape'][2]])
    image = image / 255.0  # normalize to [0,1] if necessary
    return np.expand_dims(image, axis=0)

def extract_detections(output_tensor, confidence_threshold=0.5):
    # Assuming output_tensor shape is [1, 8, 8400]
    # Change indices as per your model's output format
    detections = output_tensor[0].transpose()  # Transpose to make it [8400, 8]
    results = []

    for detection in detections:
        ymin, xmin, ymax, xmax, confidence, class_id = detection[:6]
        
        if confidence > confidence_threshold:
            result = {
                'bounding_box': [float(ymin), float(xmin), float(ymax), float(xmax)],
                'class_id': int(class_id),
                'confidence': float(confidence)
            }
            results.append(result)
    
    return results



@app.websocket("/video")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        last_sent_time = asyncio.get_event_loop().time() 
        while True:
          try:
            message = await websocket.receive_bytes()
            if message.startswith(b'\xff\xd8'):  # JPEG start marker
                start = time.time()
                #frame = cv2.imdecode(np.frombuffer(message, np.uint8), cv2.IMREAD_COLOR)
                processed_image = preprocess_image(message)
                interpreter.set_tensor(input_details[0]['index'], processed_image)
                interpreter.invoke()
                #results = interpreter.get_tensor(output_details[0]['index'])
                output_tensor = interpreter.get_tensor(output_details[0]['index'])
                detections = extract_detections(output_tensor, confidence_threshold=0.5)

                end = time.time()

                print(start-end)
                #results = model([frame])
                #found_box = False
                #for result in results:
                #    boxes = result.boxes
                if len(detections) > 0:
                    #print("Sending boxes:", boxes[0])  
                    boxxyn = detections[0] # Assuming this is the format you choose
                    print(boxxyn)
                    #box_data = [{
                    #        "x1": int(box[0].item()),
                    #        "y1": int(box[1].item()),
                    #        "x2": int(box[2].item()),
                    #        "y2": int(box[3].item())
                    #    } for box in boxxyn]
                    await websocket.send_json(boxxyn)
                    last_sent_time = asyncio.get_event_loop().time()
                else:
                    await websocket.send_text("None")
            else:
                continue
          except Exception as e:
            print("Client Disconnected",e)
            break
    except Exception as e:
        print("Client Disconnected",e)
    #finally:
    #    connected_clients.remove(websocket)
    #    print("WebSocket closed and cleaned up")


@app.get("/")
async def get_root():
    file_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, ssl_keyfile=keyfile, ssl_certfile=certfile,ws_ping_interval=25)

