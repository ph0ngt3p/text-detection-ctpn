import numpy as np
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub

import grpc
import cv2

# Your config
SERVICE_HOST = 'localhost:8500'  # sample
MODEL_NAME = 'cropper'


def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])

#######


# Connect to server
channel = grpc.insecure_channel(SERVICE_HOST)
stub = PredictionServiceStub(channel)

# Prepare request object
request = predict_pb2.PredictRequest()
request.model_spec.name = MODEL_NAME
request.model_spec.signature_name = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

# Copy image into request's content
img = cv2.imread('data/demo/1.jpg')
img, (rh, rw) = resize_image(img)
h, w, c = img.shape
im_info = np.array([h, w, c], dtype=np.float32).reshape([1, 3])
input_img = np.expand_dims(img, axis=0)  # we do batch inferencing, so input is a 4-D tensor
request.inputs['image'].CopyFrom(
    tf.contrib.util.make_tensor_proto(input_img.astype('float32')))
request.inputs['im_info'].CopyFrom(
    tf.contrib.util.make_tensor_proto(im_info))

# Do inference
result = stub.Predict.future(request, 5)  # 5s timeout

# Get output depends on our input/output signature, and their types
# for example, our output signature key is 'output' and has string value
box_pred = np.array(result.result().outputs['boxes'].float_val, dtype=np.float32).reshape(-1, 4)  # we have batch result, so just take first index
print(box_pred)
