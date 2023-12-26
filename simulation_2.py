import cv2
import boto3
import botocore
import json
import numpy as np

face_detector1 = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_dectector1 = cv2.CascadeClassifier('haarcascade_eye.xml')

predict_labels = ['Angry', 'Disgust', 'Fear',
                  'Happy', 'Neutral', 'Sad', 'Surprise']

# Segment the faceprint out of video stream
cap = cv2.VideoCapture(
    'stock/The 7 basic emotions - Do you recognise all facial expressions_.mp4')

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Tensor serving endpoint config
endpoint = 'sagemaker-tensorflow-serving-2023-11-01-08-26-46-398'
config = botocore.config.Config(read_timeout=80)
runtime = boto3.client('runtime.sagemaker', config=config)

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('output_emotion_detect.avi', cv2.VideoWriter_fourcc(
    'M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

while cap.isOpened():
    _, frame = cap.read()
    if (_):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector1.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=6)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = eye_dectector1.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=6)
            # Only accept faceprint has at least 2 eyes
            if (len(eyes) > 1):
                # Found faceprint!
                cv2.rectangle(frame, pt1=(x, y), pt2=(x+w, y+h),
                              color=(255, 0, 0), thickness=3)

                # Infer emotion
                face = cv2.resize(roi_gray, (48, 48))
                # cv2.imshow("Resized image", face)
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

                text = 'Not sure'
                if (predict_certainty > 60):
                    text = predict_label + ' - ' + str(predict_certainty) + '%'
                
                cv2.putText(frame, text, (x, y + h + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            # Write the frame into the file 'output.avi'
            out.write(frame)
            # cv2.imshow("window", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    else:
        break

print("DONE")
frame.release()
out.release()
cap.release()
cv2.destroyAllWindows()
