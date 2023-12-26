import cv2
import boto3
import botocore
import json
import numpy as np
import os

def determineResult(predict_label, filename):
    if predict_label in filename:
        return 'correct'
    return 'false'

face_detector1 = cv2.CascadeClassifier('../haarcascade_frontalface_default.xml')
predict_labels = ['angry', 'disgust', 'fear',
                  'happy', 'neutral', 'sad', 'surprise']

# Tensor serving endpoint config
endpoint = 'sagemaker-tensorflow-serving-2023-09-24-06-34-31-904'
config = botocore.config.Config(read_timeout=80)
runtime = boto3.client('runtime.sagemaker', config=config)

# Loop through and segment the faceprint out of the samples
for filename in os.listdir('test_set'):
    frame = cv2.imread(filename=os.path.join('test_set', filename))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Infer emotion
    face = cv2.resize(gray, (48, 48))
    face = face[..., np.newaxis]
    face = np.expand_dims(face, axis=0)
    data = np.array(face)
    payload = json.dumps({"instances": data.tolist()})

    response = runtime.invoke_endpoint(EndpointName=endpoint,
                                        ContentType='application/json',
                                        Body=payload)
    result = json.loads(response['Body'].read().decode())

    predict_label = predict_labels[(
        np.array(result['predictions'][0]).argmax())]
    predict_certainty = float(
        np.array(result['predictions'][0]).max()) * 100
    result = determineResult(predict_label, filename)

    # Put prediction info overlay
    text = predict_label + '_' + str(predict_certainty) + ' - '
    
    cv2.imwrite(os.path.join('result', result, text + filename), frame)

print("DONE")

cv2.destroyAllWindows()
