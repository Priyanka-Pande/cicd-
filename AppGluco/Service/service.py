import torch
import cv2
import numpy as np
import pandas as pd
from Users.Data.data import is_profile_exsits_data
from AppGluco.Data.data import *
from threading import Thread
from tensorflow.keras.models import load_model


def get_professional_all_patients(user):
    profile_id = is_profile_exsits_data(user.id,user.type)
    patients = get_all_patients_of_professional_data(profile_id)
    return patients

def get_report_for_personal(user):
    patient_id = is_profile_exsits_data(user.id,user.type)
    patients = get_result_report_personal_data(patient_id)
    return patients


def getFrame(sec, count, video_uid, x_test,  yolo_model, final_predictions,cModel):
    global roi
    # Update the path where video is stored
    vidcap = cv2.VideoCapture(
        "https://storage.googleapis.com/glucoqr-p1-preprod/" + video_uid)
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = vidcap.read()

    if hasFrames:
        img_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ##########----- YOLO ROI Extraction -----##########
        results = yolo_model(img_array)
        # Get the predicted bounding boxes, labels, and scores
        pred_boxes = results.xyxy[0]
        selected_box = pred_boxes[pred_boxes[:, 5] == 2]
        if selected_box.shape[0] > 0:
            selected_box = selected_box[0]
            x, y, w, h = selected_box[0:4].int().tolist()

            # Crop the region of interest (ROI) from the image
            roi = image[y:h, x:w]
            roiV2 = roi.copy()


            dim = (16, 9)
            width, height = roi.shape[1], roi.shape[0]

            # process crop width and height for max available dimension
            crop_width = dim[0] if dim[0] < roi.shape[1] else roi.shape[1]
            crop_height = dim[1] if dim[1] < roi.shape[0] else roi.shape[0]
            mid_x, mid_y = int(width / 2), int(height / 2)
            cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
            roi = roi[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
            roi_normal = roi.copy()
            roi = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
            ##########----- YOLO Resenet Regression Array -----##########
            img_array_r = cv2.cvtColor(roi_normal, cv2.COLOR_BGR2RGB)
            img_array_r = cv2.resize(img_array_r, (64, 36))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_array_r = ((img_array_r / 255.0) - mean) / std
            img_array_r = img_array_r.reshape((1,) + img_array_r.shape)
            x_test.append(img_array_r)

            ##########----- YOLO Classification Resenet Array -----##########
            img_array_c = cv2.cvtColor(roi_normal, cv2.COLOR_BGR2RGB)
            img_array_c = cv2.resize(img_array_c, (64, 36))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_array_c = ((img_array_c / 255.0) - mean) / std
            img_array_c = img_array_c.reshape((1,) + img_array_c.shape)
            predictions = cModel.predict(img_array_c)
            predictions = pd.DataFrame(predictions)
            pred_probabilities = np.array(predictions)
            predicted_label_index = np.argmax(pred_probabilities)
            print("The value of normalised:",predicted_label_index)
            final_predictions.append(predicted_label_index)


        return x_test, final_predictions


def upload_video_for_patient(patient_id,video_file,user):
    video = upload_video_file(patient_id,video_file)
    video_name = get_video_name_of_patient(patient_id,video.id)
    video_file_name = str(video_name.video_file)
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/gqrp1_preprod/gqrp1_preprod/models/yolo_model.pt', force_reload=True)
    sec = 0
    frameRate = 0.116  # //it will capture image in each 0.5 second
    count = 1
    labels = ['nan', 'Class_60_80', 'Class_80_100', 'Class_100_120', 'Class_120_140', 'Class_140_180',
          'Class_180_220', 'Class_220_280', 'Class_280_650']

    final_predictions = []
    x_test = []
    cModel = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/Classification_modelV3normalized.h5')
    # get video id as .mp4
    video_uid=  video_file_name
    while sec < 7:
        count = count + 1

        feature = getFrame(sec, count, video_uid, x_test, yolo_model,final_predictions, cModel)
        sec = sec + frameRate
        sec = round(sec, 2)

    # ------------Resnet Model Prediction-------------#
    x_test = np.array(x_test)
    x_test = x_test.reshape(len(x_test), 36, 64, 3)
    # ------------Resnet YOlO Model V2 Prediction-------------#
 #  v2final_predictions = np.array(v2final_predictions)
 #  v2final_predictions = v2final_predictions.reshape(len(v2final_predictions))
 #  v2final_predictions = np.bincount(v2final_predictions).argmax()
 #  final_prediction_labelsV2 = labels[v2final_predictions]
 #   print("The mean value is V2:", v2final_predictions)

    # ------------Resnet YOlO Model Prediction-------------#
    final_predictions = np.array(final_predictions)
    final_predictions = final_predictions.reshape(len(final_predictions))
    final_prediction = np.bincount(final_predictions).argmax()
    final_prediction_labels = labels[final_prediction]
    if (final_prediction == 1):
        rYoloModel = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class1.h5', compile=False)
        rYoloModel.compile(optimizer='adam', loss='mean_squared_error')
        rPredictions = rYoloModel.predict(x_test)
        rRrediction_mean = round(rPredictions.mean() * 80, 2)
        print("Regression Value 1 : ", str(rRrediction_mean))
        # Insert query for inserting gluco value mapped in rRrediction_mean
        insert_reading = insert_gluco_value_for_patient(user,patient_id,video,rRrediction_mean)
    elif (final_prediction == 2):
        rYoloModel = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class2.h5', compile=False)
        rYoloModel.compile(optimizer='adam', loss='mean_squared_error')
        rPredictions = rYoloModel.predict(x_test)
        rRrediction_mean = round(rPredictions.mean() * 100, 2)
        print("Regression Value 2 : ", str(rRrediction_mean))
        # Insert query for inserting gluco value mapped in rRrediction_mean
        insert_reading = insert_gluco_value_for_patient(user,patient_id,video,rRrediction_mean)
    elif (final_prediction == 3):
        rYoloModel = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class3.h5', compile=False)
        rYoloModel.compile(optimizer='adam', loss='mean_squared_error')
        rPredictions = rYoloModel.predict(x_test)
        rRrediction_mean = round(rPredictions.mean() * 120, 2)
        print("Regression Value 3 : ", str(rRrediction_mean))
        # Insert query for inserting gluco value mapped in rRrediction_mean
        insert_reading = insert_gluco_value_for_patient(user,patient_id,video,rRrediction_mean)
    elif (final_prediction == 4):
        rYoloModel = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class4.h5', compile=False)
        rYoloModel.compile(optimizer='adam', loss='mean_squared_error')
        rPredictions = rYoloModel.predict(x_test)
        rRrediction_mean = round(rPredictions.mean() * 140, 2)
        print("Regression Value 4 : ", str(rRrediction_mean))
        # Insert query for inserting gluco value mapped in rRrediction_mean
        insert_reading = insert_gluco_value_for_patient(user,patient_id,video,rRrediction_mean)
    elif (final_prediction == 5):
        rYoloModel = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class5.h5', compile=False)
        rYoloModel.compile(optimizer='adam', loss='mean_squared_error')
        rPredictions = rYoloModel.predict(x_test)
        rRrediction_mean = round(rPredictions.mean() * 180, 2)
        print("Regression Value 5 : ", str(rRrediction_mean))
        # Insert query for inserting gluco value mapped in rRrediction_mean
        insert_reading = insert_gluco_value_for_patient(user,patient_id,video,rRrediction_mean)
    elif (final_prediction == 6):
        rYoloModel = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class6.h5', compile=False)
        rYoloModel.compile(optimizer='adam', loss='mean_squared_error')
        rPredictions = rYoloModel.predict(x_test)
        rRrediction_mean = round(rPredictions.mean() * 220, 2)
        print("Regression Value 6 : ", str(rRrediction_mean))
        # Insert query for inserting gluco value mapped in rRrediction_mean
        insert_reading = insert_gluco_value_for_patient(user,patient_id,video,rRrediction_mean)
    elif (final_prediction == 7):
        rYoloModel = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class7.h5', compile=False)
        rYoloModel.compile(optimizer='adam', loss='mean_squared_error')
        rPredictions = rYoloModel.predict(x_test)
        rRrediction_mean = round(rPredictions.mean() * 280, 2)
        print("Regression Value 7 : ", str(rRrediction_mean))
        # Insert query for inserting gluco value mapped in rRrediction_mean
        insert_reading = "INSERT INTO GlucoQR_Reading (ScanID, AppUserID, BalancedData, UnbalancedData,BalancedClasses) VALUES (%s,%s,%s,%s,%s)"
        insert_reading = insert_gluco_value_for_patient(user,patient_id,video,rRrediction_mean)
    elif (final_prediction == 8):
        rYoloModel = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class8.h5', compile=False)
        rYoloModel.compile(optimizer='adam', loss='mean_squared_error')
        rPredictions = rYoloModel.predict(x_test)
        rRrediction_mean = round(rPredictions.mean() * 650, 2)
        print("Regression Value 8 : ", str(rRrediction_mean))
        # Insert query for inserting gluco value mapped in rRrediction_mean
        insert_reading = insert_gluco_value_for_patient(user,patient_id,video,rRrediction_mean)

    result = { 
        "patient_id" : insert_reading.patient_id.id,
        "full_name" : insert_reading.patient_id.full_name,
        "age" : insert_reading.patient_id.age,
        "gluco_value" : insert_reading.gluco_value,
    }
    return result
