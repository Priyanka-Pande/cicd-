import torch
import cv2
import pickle
import numpy as np
import pandas as pd
from threading import Thread
from joblib import dump, load
from AppGluco.Data.data import *
from rest_framework import status
from rest_framework.response import Response
from tensorflow.keras.models import load_model
from Users.Data.data import is_profile_exsits_data


def get_frequently_asked_questions():
    faq = get_frequently_asked_questions_data()
    return faq

def get_professional_all_patients(user):
    profile_id = is_profile_exsits_data(user.id,user.type)
    patients = get_all_patients_of_professional_data(profile_id)
    return patients

def get_report_for_personal(user):
    patient_id = is_profile_exsits_data(user.id,user.type)
    patients = get_result_report_personal_data(patient_id)
    return patients

def adjust_levels(image, level_in_low, level_in_high, level_out_low=0, level_out_high=255):
    # Compute scale and offset
    scale = (level_out_high - level_out_low) / (level_in_high - level_in_low)
    offset = level_out_low - level_in_low * scale
    # Apply adjustment
    image = cv2.convertScaleAbs(image, alpha=scale, beta=offset)
    
    return image

def apply_gaussian_blur(frame, kernel_size=(5, 5), sigma=0):
    return cv2.GaussianBlur(frame, kernel_size, sigma)


def apply_auto_white_balance(frame):
    # Convert the frame to LAB color space
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)

    # Extract the L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(lab_frame)

    # Find the average value of the A and B channels
    avg_a = a_channel.mean()
    avg_b = b_channel.mean()

    # Adjust the A and B channels based on the gray world assumption
    a_channel = a_channel - ((128 - avg_a) * (l_channel / 255.0) * 1.1)
    b_channel = b_channel - ((128 - avg_b) * (l_channel / 255.0) * 1.1)

    # Clip the values to ensure they stay within the valid range [0, 255]
    a_channel = np.clip(a_channel, 0, 255).astype(np.uint8)
    b_channel = np.clip(b_channel, 0, 255).astype(np.uint8)

    # Merge the adjusted channels
    balanced_lab_frame = cv2.merge([l_channel, a_channel, b_channel])

    # Convert the balanced frame back to RGB color space
    balanced_frame = cv2.cvtColor(balanced_lab_frame, cv2.COLOR_LAB2RGB)

    return balanced_frame

# def getFrame(sec, count, video_uid, x_test,  yolo_model, final_predictions,cModel):
#     global roi
#     # Update the path where video is stored
#     vidcap = cv2.VideoCapture(
#         "https://storage.googleapis.com/glucoqr-p1-preprod/" + video_uid)
#     vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
#     hasFrames, image = vidcap.read()

#     if hasFrames:
#         img_array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         blurred_frame = apply_gaussian_blur(img_array, kernel_size=(5, 5), sigma=0)
#         # Apply auto white balance to the blurred frame
#         img_array = apply_auto_white_balance(blurred_frame)
#         ##########----- YOLO ROI Extraction -----##########
#         results = yolo_model(img_array)
#         # Get the predicted bounding boxes, labels, and scores
#         pred_boxes = results.xyxy[0]
#         selected_box = pred_boxes[pred_boxes[:, 5] == 2]
#         if selected_box.shape[0] > 0:
#             selected_box = selected_box[0]
#             x, y, w, h = selected_box[0:4].int().tolist()

#             # Crop the region of interest (ROI) from the image
#             roi = img_array[y:h, x:w]
#             roiV2 = roi.copy()


#             dim = (16, 9)
#             width, height = roi.shape[1], roi.shape[0]

#             # process crop width and height for max available dimension
#             crop_width = dim[0] if dim[0] < roi.shape[1] else roi.shape[1]
#             crop_height = dim[1] if dim[1] < roi.shape[0] else roi.shape[0]
#             mid_x, mid_y = int(width / 2), int(height / 2)
#             cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
#             roi = roi[mid_y - ch2:mid_y + ch2, mid_x - cw2:mid_x + cw2]
#             roi_normal = roi.copy()
#             roi = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
#             img_array = cv2.resize(roiV2, (100, 50))
#             img_array = img_array[12:-6, 18:-18, :]
#             ##########----- YOLO Resenet Regression Array -----##########
#             img_array_r = cv2.cvtColor(roi_normal, cv2.COLOR_BGR2RGB)
#             # level_in_low = np.min(img_array_r)
#             # level_in_high = np.max(img_array_r)
#             # # Adjust the levels
#             # img_array_r = adjust_levels(img_array_r, level_in_low, level_in_high, level_out_low=0, level_out_high=255)
#             img_array_r = cv2.resize(img_array_r, (64, 36))
#             mean = np.array([0.485, 0.456, 0.406])
#             std = np.array([0.229, 0.224, 0.225])
#             img_array_r = ((img_array_r / 255.0) - mean) / std
#             img_array_r = img_array_r.reshape((1,) + img_array_r.shape)
#             x_test.append(img_array_r)

#             ##########----- YOLO Classification Resenet Array -----##########
#             img_array_c = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
#             level_in_low = np.min(img_array_c)
#             level_in_high = np.max(img_array_c)
#             # Adjust the levels
#             img_array_c = adjust_levels(img_array_c, level_in_low, level_in_high, level_out_low=0, level_out_high=255)
#             # img_array_c = cv2.resize(img_array_c, (64, 36))
#             mean = np.array([0.485, 0.456, 0.406])
#             std = np.array([0.229, 0.224, 0.225])
#             img_array_c = ((img_array_c / 255.0) - mean) / std
#             img_array_c = img_array_c.reshape((1,) + img_array_c.shape)
#             predictions = cModel.predict(img_array_c)
#             predictions = pd.DataFrame(predictions)
#             pred_probabilities = np.array(predictions)
#             predicted_label_index = np.argmax(pred_probabilities)
#             print("The value of normalised:",predicted_label_index)
#             final_predictions.append(predicted_label_index)


#         return x_test, final_predictions


# def upload_video_for_patient(patient_id,video_file,user):
#     video = upload_video_file(patient_id,video_file)
#     video_name = get_video_name_of_patient(patient_id,video.id)
#     video_file_name = str(video_name.video_file)
#     yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/gqrp1_preprod/gqrp1_preprod/models/yolo_model.pt', force_reload=True)
#     sec = 0
#     frameRate = 0.116  # //it will capture image in each 0.5 second
#     count = 1
#     labels = ['nan', 'Class_60_80', 'Class_80_100', 'Class_100_120', 'Class_120_140', 'Class_140_180',
#           'Class_180_220', 'Class_220_280', 'Class_280_650']

#     final_predictions = []
#     x_test = []
#     x_test_reg = []
#     cModel = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/Classification_modelV6normalized_full_preprocess.h5')
#     # get video id as .mp4
#     video_uid=  video_file_name
#     while sec < 7:
#         count = count + 1

#         feature = getFrame(sec, count, video_uid, x_test, yolo_model,final_predictions, cModel)
#         sec = sec + frameRate
#         sec = round(sec, 2)

#     # ------------Resnet Model Prediction-------------#
#     x_test = np.array(x_test)
#     x_test = x_test.reshape(len(x_test), 36, 64, 3)
#     # ------------Resnet YOlO Model V2 Prediction-------------#
#  #  v2final_predictions = np.array(v2final_predictions)
#  #  v2final_predictions = v2final_predictions.reshape(len(v2final_predictions))
#  #  v2final_predictions = np.bincount(v2final_predictions).argmax()
#  #  final_prediction_labelsV2 = labels[v2final_predictions]
#  #   print("The mean value is V2:", v2final_predictions)

#     # ------------Resnet YOlO Model Prediction-------------#
#     final_predictions = np.array(final_predictions)
#     final_predictions = final_predictions.reshape(len(final_predictions))
#     final_prediction = np.bincount(final_predictions).argmax()
#     final_prediction_labels = labels[final_prediction]
#     if (final_prediction == 1):
#         rYoloModel = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class1.h5', compile=False)
#         rYoloModel.compile(optimizer='adam', loss='mean_squared_error')
#         rPredictions = rYoloModel.predict(x_test)
#         rRrediction_mean = round(rPredictions.mean() * 80, 2)
#         print("Regression Value 1 : ", str(rRrediction_mean))
#         # Insert query for inserting gluco value mapped in rRrediction_mean
#         insert_reading = insert_gluco_value_for_patient(user,patient_id,video,rRrediction_mean)
#     elif (final_prediction == 2):
#         rYoloModel = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class2.h5', compile=False)
#         rYoloModel.compile(optimizer='adam', loss='mean_squared_error')
#         rPredictions = rYoloModel.predict(x_test)
#         rRrediction_mean = round(rPredictions.mean() * 100, 2)
#         print("Regression Value 2 : ", str(rRrediction_mean))
#         # Insert query for inserting gluco value mapped in rRrediction_mean
#         insert_reading = insert_gluco_value_for_patient(user,patient_id,video,rRrediction_mean)
#     elif (final_prediction == 3):
#         rYoloModel = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class3.h5', compile=False)
#         rYoloModel.compile(optimizer='adam', loss='mean_squared_error')
#         rPredictions = rYoloModel.predict(x_test)
#         rRrediction_mean = round(rPredictions.mean() * 120, 2)
#         print("Regression Value 3 : ", str(rRrediction_mean))
#         # Insert query for inserting gluco value mapped in rRrediction_mean
#         insert_reading = insert_gluco_value_for_patient(user,patient_id,video,rRrediction_mean)
#     elif (final_prediction == 4):
#         rYoloModel = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class4.h5', compile=False)
#         rYoloModel.compile(optimizer='adam', loss='mean_squared_error')
#         rPredictions = rYoloModel.predict(x_test)
#         rRrediction_mean = round(rPredictions.mean() * 140, 2)
#         print("Regression Value 4 : ", str(rRrediction_mean))
#         # Insert query for inserting gluco value mapped in rRrediction_mean
#         insert_reading = insert_gluco_value_for_patient(user,patient_id,video,rRrediction_mean)
#     elif (final_prediction == 5):
#         rYoloModel = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class5.h5', compile=False)
#         rYoloModel.compile(optimizer='adam', loss='mean_squared_error')
#         rPredictions = rYoloModel.predict(x_test)
#         rRrediction_mean = round(rPredictions.mean() * 180, 2)
#         print("Regression Value 5 : ", str(rRrediction_mean))
#         # Insert query for inserting gluco value mapped in rRrediction_mean
#         insert_reading = insert_gluco_value_for_patient(user,patient_id,video,rRrediction_mean)
#     elif (final_prediction == 6):
#         rYoloModel = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class6.h5', compile=False)
#         rYoloModel.compile(optimizer='adam', loss='mean_squared_error')
#         rPredictions = rYoloModel.predict(x_test)
#         rRrediction_mean = round(rPredictions.mean() * 220, 2)
#         print("Regression Value 6 : ", str(rRrediction_mean))
#         # Insert query for inserting gluco value mapped in rRrediction_mean
#         insert_reading = insert_gluco_value_for_patient(user,patient_id,video,rRrediction_mean)
#     elif (final_prediction == 7):
#         rYoloModel = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class7.h5', compile=False)
#         rYoloModel.compile(optimizer='adam', loss='mean_squared_error')
#         rPredictions = rYoloModel.predict(x_test)
#         rRrediction_mean = round(rPredictions.mean() * 280, 2)
#         print("Regression Value 7 : ", str(rRrediction_mean))
#         # Insert query for inserting gluco value mapped in rRrediction_mean
#         insert_reading = "INSERT INTO GlucoQR_Reading (ScanID, AppUserID, BalancedData, UnbalancedData,BalancedClasses) VALUES (%s,%s,%s,%s,%s)"
#         insert_reading = insert_gluco_value_for_patient(user,patient_id,video,rRrediction_mean)
#     elif (final_prediction == 8):
#         rYoloModel = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class8.h5', compile=False)
#         rYoloModel.compile(optimizer='adam', loss='mean_squared_error')
#         rPredictions = rYoloModel.predict(x_test)
#         rRrediction_mean = round(rPredictions.mean() * 650, 2)
#         print("Regression Value 8 : ", str(rRrediction_mean))
#         # Insert query for inserting gluco value mapped in rRrediction_mean
#         insert_reading = insert_gluco_value_for_patient(user,patient_id,video,rRrediction_mean)

#     result = { 
#         "patient_id" : insert_reading.patient_id.id,
#         "full_name" : insert_reading.patient_id.full_name,
#         "age" : insert_reading.patient_id.age,
#         "gluco_value" : insert_reading.gluco_value,
#     }
#     return result

def getFrame(sec, count, video_uid, x_test,  yolo_model,x_test_reg):
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
            roi = img_array[y:h, x:w]
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
            img_array = cv2.resize(roiV2, (100, 50))
            img_array = img_array[12:-6, 18:-18, :]
            ##########----- YOLO Resenet Regression Array -----##########
            img_array_r = cv2.cvtColor(roi_normal, cv2.COLOR_BGR2RGB)
            # level_in_low = np.min(img_array_r)
            # level_in_high = np.max(img_array_r)
            # # Adjust the levels
            # img_array_r = adjust_levels(img_array_r, level_in_low, level_in_high, level_out_low=0, level_out_high=255)
            img_array_r = cv2.resize(img_array_r, (64, 36))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_array_r = ((img_array_r / 255.0) - mean) / std
            img_array_r = img_array_r.reshape((1,) + img_array_r.shape)
            x_test_reg.append(img_array_r)

            img_array = cv2.resize(roiV2, (100, 50))
            img_array = img_array[12:-6, 18:-18, :]

            # --------------------------------------------------------------------------
            #                Get Min and Max Level of image
            # --------------------------------------------------------------------------
            level_in_low = np.min(img_array)
            level_in_high = np.max(img_array)
            # Call to adjustment level function and return adjusted image to store in img_array variable
            img_array = adjust_levels(img_array, level_in_low, level_in_high, level_out_low=0, level_out_high=255)
            # --------------------------------------------------------------------------
            #               Level Adjustment Ends
            # --------------------------------------------------------------------------

            # img_array = cv2.resize(img_array, (64, 36))
            img_array = img_array.reshape((1,) + img_array.shape)
            img_normalized = img_array / 255.0  # ensures values are in range [0, 1]
            img_uint8 = (img_normalized * 255).astype(np.uint8)
            # Compute histogram
            hist = cv2.calcHist([img_uint8], [0], None, [256], [0, 256])
            x_test.append(hist)

        return x_test,x_test_reg


def upload_video_for_patient(patient_id,video_file,user):
    video = upload_video_file(patient_id,video_file)
    video_name = get_video_name_of_patient(patient_id,video.id)
    video_file_name = str(video_name.video_file)
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/gqrp1_preprod/gqrp1_preprod/models/yolo_model.pt', force_reload=True)
    sec = 0
    frameRate = 0.116  # //it will capture image in each 0.5 second
    count = 1
    labels = ['Class_60_80', 'Class_80_100', 'Class_100_120', 'Class_120_140', 'Class_140_180',
          'Class_180_220', 'Class_220_280', 'Class_280_650']

    final_predictions = []
    x_test = []
    x_test_reg = []
    knnModel = load('/home/gqrp1_preprod/gqrp1_preprod/models/knn_model_yolo_optimized_v1_5_distance_euclidean.joblib')
    # get video id as .mp4
    video_uid=  video_file_name
    while sec < 7:
        count = count + 1

        feature = getFrame(sec, count, video_uid, x_test, yolo_model,x_test_reg)
        sec = sec + frameRate
        sec = round(sec, 2)

#    print('-------------------------')
#    print(f'Count {count}')
#    print('-------------------------')
#    # ------------Resnet Model Prediction-------------#
#    x_test = np.array(x_test)
#    x_test = x_test.reshape(len(x_test), 36, 64, 3)
#    # ------------Resnet YOlO Model V2 Prediction-------------#
# #  v2final_predictions = np.array(v2final_predictions)
# #  v2final_predictions = v2final_predictions.reshape(len(v2final_predictions))
# #  v2final_predictions = np.bincount(v2final_predictions).argmax()
# #  final_prediction_labelsV2 = labels[v2final_predictions]
# #   print("The mean value is V2:", v2final_predictions)
#
#    # ------------Resnet YOlO Model Prediction-------------#
#    final_predictions = np.array(final_predictions)
#    final_predictions = final_predictions.reshape(len(final_predictions))
#    final_prediction = np.bincount(final_predictions).argmax()
#    final_prediction_labels = labels[final_prediction]
#    if (final_prediction == 1):
#        rYoloModel = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class1.h5', compile=False)
#        rYoloModel.compile(optimizer='adam', loss='mean_squared_error')
#        rPredictions = rYoloModel.predict(x_test)
#        rRrediction_mean = round(rPredictions.mean() * 80, 2)
#        print("Regression Value 1 : ", str(rRrediction_mean))
#        # Insert query for inserting gluco value mapped in rRrediction_mean
#        insert_reading = insert_gluco_value_for_patient(user,patient_id,video,rRrediction_mean)
#    elif (final_prediction == 2):
#        rYoloModel = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class2.h5', compile=False)
#        rYoloModel.compile(optimizer='adam', loss='mean_squared_error')
#        rPredictions = rYoloModel.predict(x_test)
#        rRrediction_mean = round(rPredictions.mean() * 100, 2)
#        print("Regression Value 2 : ", str(rRrediction_mean))
#        # Insert query for inserting gluco value mapped in rRrediction_mean
#        insert_reading = insert_gluco_value_for_patient(user,patient_id,video,rRrediction_mean)
#    elif (final_prediction == 3):
#        rYoloModel = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class3.h5', compile=False)
#        rYoloModel.compile(optimizer='adam', loss='mean_squared_error')
#        rPredictions = rYoloModel.predict(x_test)
#        rRrediction_mean = round(rPredictions.mean() * 120, 2)
#        print("Regression Value 3 : ", str(rRrediction_mean))
#        # Insert query for inserting gluco value mapped in rRrediction_mean
#        insert_reading = insert_gluco_value_for_patient(user,patient_id,video,rRrediction_mean)
#    elif (final_prediction == 4):
#        rYoloModel = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class4.h5', compile=False)
#        rYoloModel.compile(optimizer='adam', loss='mean_squared_error')
#        rPredictions = rYoloModel.predict(x_test)
#        rRrediction_mean = round(rPredictions.mean() * 140, 2)
#        print("Regression Value 4 : ", str(rRrediction_mean))
#        # Insert query for inserting gluco value mapped in rRrediction_mean
#        insert_reading = insert_gluco_value_for_patient(user,patient_id,video,rRrediction_mean)
#    elif (final_prediction == 5):
#        rYoloModel = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class5.h5', compile=False)
#        rYoloModel.compile(optimizer='adam', loss='mean_squared_error')
#        rPredictions = rYoloModel.predict(x_test)
#        rRrediction_mean = round(rPredictions.mean() * 180, 2)
#        print("Regression Value 5 : ", str(rRrediction_mean))
#        # Insert query for inserting gluco value mapped in rRrediction_mean
#        insert_reading = insert_gluco_value_for_patient(user,patient_id,video,rRrediction_mean)
#    elif (final_prediction == 6):
#        rYoloModel = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class6.h5', compile=False)
#        rYoloModel.compile(optimizer='adam', loss='mean_squared_error')
#        rPredictions = rYoloModel.predict(x_test)
#        rRrediction_mean = round(rPredictions.mean() * 220, 2)
#        print("Regression Value 6 : ", str(rRrediction_mean))
#        # Insert query for inserting gluco value mapped in rRrediction_mean
#        insert_reading = insert_gluco_value_for_patient(user,patient_id,video,rRrediction_mean)
#    elif (final_prediction == 7):
#        rYoloModel = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class7.h5', compile=False)
#        rYoloModel.compile(optimizer='adam', loss='mean_squared_error')
#        rPredictions = rYoloModel.predict(x_test)
#        rRrediction_mean = round(rPredictions.mean() * 280, 2)
#        print("Regression Value 7 : ", str(rRrediction_mean))
#        # Insert query for inserting gluco value mapped in rRrediction_mean
#        insert_reading = "INSERT INTO GlucoQR_Reading (ScanID, AppUserID, BalancedData, UnbalancedData,BalancedClasses) VALUES (%s,%s,%s,%s,%s)"
#        insert_reading = insert_gluco_value_for_patient(user,patient_id,video,rRrediction_mean)
#    elif (final_prediction == 8):
#        rYoloModel = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class8.h5', compile=False)
#        rYoloModel.compile(optimizer='adam', loss='mean_squared_error')
#        rPredictions = rYoloModel.predict(x_test)
#        rRrediction_mean = round(rPredictions.mean() * 650, 2)
#        print("Regression Value 8 : ", str(rRrediction_mean))
#        # Insert query for inserting gluco value mapped in rRrediction_mean
#        insert_reading = insert_gluco_value_for_patient(user,patient_id,video,rRrediction_mean)
#
#    result = { 
#       "patient_id" : insert_reading.patient_id.id,
#        "full_name" : insert_reading.patient_id.full_name,
#        "age" : insert_reading.patient_id.age,
#        "gluco_value" : insert_reading.gluco_value,
    x_test_reg = np.array(x_test_reg)
    x_test_reg = x_test_reg.reshape(len(x_test_reg), 36, 64, 3)
    rPredictions = ""
    predication_list = []
    # Flatten test histograms and predict labels
    X_test_flattened = [hist.flatten() for hist in x_test]
    predictions = knnModel.predict(X_test_flattened)
    final_predictions = pd.DataFrame(predictions)
    identified_class = final_predictions[0].value_counts().idxmax()

    # Find the index of the first occurrence of the matching element
    identified_class_final_idx = labels.index(identified_class)

    if ('Class_60_80' in final_predictions[0].values):
        matching_indices_1 = labels.index('Class_60_80')
        predication_list.append(matching_indices_1)
    if ('Class_80_100' in final_predictions[0].values):
        matching_indices_2 = labels.index('Class_80_100')
        predication_list.append(matching_indices_2)
    if ('Class_100_120' in final_predictions[0].values):
        matching_indices_3 = labels.index('Class_100_120')
        predication_list.append(matching_indices_3)
    if ('Class_120_140' in final_predictions[0].values):
        matching_indices_4 = labels.index('Class_120_140')
        predication_list.append(matching_indices_4)
    if ('Class_140_180' in final_predictions[0].values):
        matching_indices_5 = labels.index('Class_140_180')
        predication_list.append(matching_indices_5)
    if ('Class_180_220' in final_predictions[0].values):
        matching_indices_6 = labels.index('Class_180_220')
        predication_list.append(matching_indices_6)
    if ('Class_220_280' in final_predictions[0].values):
        matching_indices_7 = labels.index('Class_220_280')
        predication_list.append(matching_indices_7)
    if ('Class_280_650' in final_predictions[0].values):
        matching_indices_8 = labels.index('Class_280_650')
        predication_list.append(matching_indices_8)

    predication_list_values = []
    for i in range(0, 8):
        if i in predication_list:
            print(final_predictions[0].value_counts()[labels[i]])
            predication_list_values.append(final_predictions[0].value_counts()[labels[i]])
        else:
            predication_list_values.append(0)

    class1_2 = predication_list_values[0] + predication_list_values[1]
    class2_3 = predication_list_values[1] + predication_list_values[2]
    class3_4 = predication_list_values[2] + predication_list_values[3]
    class4_5 = predication_list_values[3] + predication_list_values[4]
    class5_6 = predication_list_values[4] + predication_list_values[5]
    class6_7 = predication_list_values[5] + predication_list_values[6]
    class7_8 = predication_list_values[6] + predication_list_values[7]


    final_count = predication_list_values[0]+predication_list_values[1]+predication_list_values[2]+predication_list_values[3]+predication_list_values[4]+predication_list_values[5]+predication_list_values[6]+predication_list_values[7]
    condition_count = round((final_count *40)/100)
    # Create a dictionary to store class names and counts
    class_counts = {
        '1': class1_2,
        '2': class2_3,
        '3': class3_4,
        '4': class4_5,
        '5': class5_6,
        '6': class6_7,
        '7': class7_8,
    }

    # Find the variable name with the highest count
    highest_count_added_classes = max(class_counts, key=class_counts.get)

    if (highest_count_added_classes == '7'):
        if (class_counts[highest_count_added_classes] >= condition_count):
            # rYoloModel7 = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class7.h5', compile=False)
            # rYoloModel7.compile(optimizer='adam', loss='mean_squared_error')
            # rPredictions7 = rYoloModel7.predict(x_test_reg)
            # rRrediction_mean7 = round(rPredictions7.mean() * 280, 2)
            # print("Regression Value 7 : ", str(rRrediction_mean7), flush=True)

            # rYoloModel8 = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class8.h5', compile=False)
            # rYoloModel8.compile(optimizer='adam', loss='mean_squared_error')
            # rPredictions8 = rYoloModel8.predict(x_test_reg)
            # rRrediction_mean8 = round(rPredictions8.mean() * 650, 2)
            # print("Regression Value 8 : ", str(rRrediction_mean8), flush=True)

            # # Calculate the average of the two variables
            # final_gluco_value = round((rRrediction_mean7 + rRrediction_mean8) / 2, 2)

            model_filename7 = '/home/gqrp1_preprod/gqrp1_preprod/models/knn_class_7_v2_final_k1_full.pkl'
            # Load the model from the file using pickle
            with open(model_filename7, "rb") as f:
                knnModel7 = pickle.load(f)
            X_test_flattened = [hist.flatten() for hist in x_test]
            rRrediction_mean7 = knnModel7.predict(X_test_flattened)
            rRrediction_mean7 = np.mean(rRrediction_mean7)
            print("Regression Value 7 KNN: ", str(rRrediction_mean7), flush=True)

            model_filename8 = '/home/gqrp1_preprod/gqrp1_preprod/models/knn_class_8_v2_final_k1_full.pkl'
            # Load the model from the file using pickle
            with open(model_filename8, "rb") as f:
                knnModel8 = pickle.load(f)
            X_test_flattened = [hist.flatten() for hist in x_test]
            rRrediction_mean8 = knnModel8.predict(X_test_flattened)
            rRrediction_mean8 = np.mean(rRrediction_mean8)
            print("Regression Value 8 KNN: ", str(rRrediction_mean8), flush=True)

            #-----------------------------------------------------------
            #                Weighted Average Calculation
            #-----------------------------------------------------------
            rRrediction_mean8 = predication_list_values[7] * rRrediction_mean8
            rRrediction_mean7 = predication_list_values[6] * rRrediction_mean7
            final_gluco_value = round((rRrediction_mean8 + rRrediction_mean7)/(predication_list_values[7] + predication_list_values[6]),2)

            insert_reading = insert_gluco_value_for_patient(user,patient_id,video,final_gluco_value)
            response =  { 
                "patient_id" : insert_reading.patient_id.id,
                "full_name" : insert_reading.patient_id.full_name,
                "age" : insert_reading.patient_id.age,
                "gluco_value" : insert_reading.gluco_value,
            }

            return Response(response, status=status.HTTP_200_OK)
        else:
            return Response({"message": "Test Failed Retake The Video"}, status=status.HTTP_400_BAD_REQUEST)
    
    if (highest_count_added_classes == '6'):
        if (class_counts[highest_count_added_classes] >= condition_count):

            # rYoloModel6 = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class6.h5', compile=False)
            # rYoloModel6.compile(optimizer='adam', loss='mean_squared_error')
            # rPredictions6 = rYoloModel6.predict(x_test_reg)
            # rRrediction_mean6 = round(rPredictions6.mean() * 220, 2)
            # print("Regression Value 6 : ", str(rRrediction_mean6))

            # rYoloModel7 = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class7.h5', compile=False)
            # rYoloModel7.compile(optimizer='adam', loss='mean_squared_error')
            # rPredictions7 = rYoloModel7.predict(x_test_reg)
            # rRrediction_mean7 = round(rPredictions7.mean() * 280, 2)
            # print("Regression Value 7 : ", str(rRrediction_mean7), flush=True)

            # # Calculate the average of the two variables
            # final_gluco_value = round((rRrediction_mean6 + rRrediction_mean7) / 2, 2)

            model_filename6 = '/home/gqrp1_preprod/gqrp1_preprod/models/knn_class_6_v2_final_k1_full.pkl'
            # Load the model from the file using pickle
            with open(model_filename6, "rb") as f:
                knnModel6 = pickle.load(f)
            X_test_flattened = [hist.flatten() for hist in x_test]
            rRrediction_mean6 = knnModel6.predict(X_test_flattened)
            rRrediction_mean6 = np.mean(rRrediction_mean6)
            print("Regression Value 6 KNN: ", str(rRrediction_mean6), flush=True)

            model_filename7 = '/home/gqrp1_preprod/gqrp1_preprod/models/knn_class_7_v2_final_k1_full.pkl'
            # Load the model from the file using pickle
            with open(model_filename7, "rb") as f:
                knnModel7 = pickle.load(f)
            X_test_flattened = [hist.flatten() for hist in x_test]
            rRrediction_mean7 = knnModel7.predict(X_test_flattened)
            rRrediction_mean7 = np.mean(rRrediction_mean7)
            print("Regression Value 7 KNN: ", str(rRrediction_mean7), flush=True)

            # -----------------------------------------------------------
            #                Weighted Average Calculation
            # -----------------------------------------------------------
            rRrediction_mean6 = predication_list_values[5] * rRrediction_mean6
            rRrediction_mean7 = predication_list_values[6] * rRrediction_mean7
            final_gluco_value = round((rRrediction_mean6 + rRrediction_mean7) / (predication_list_values[5] + predication_list_values[6]), 2)

            insert_reading = insert_gluco_value_for_patient(user,patient_id,video,final_gluco_value)
            response =  { 
                "patient_id" : insert_reading.patient_id.id,
                "full_name" : insert_reading.patient_id.full_name,
                "age" : insert_reading.patient_id.age,
                "gluco_value" : insert_reading.gluco_value,
            }

            return Response(response, status=status.HTTP_200_OK)
        else:
            return Response({"message": "Test Failed Retake The Video"}, status=status.HTTP_400_BAD_REQUEST)

    if (highest_count_added_classes == '5'):
        if (class_counts[highest_count_added_classes] >= condition_count):

            # rYoloModel5 = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class5.h5', compile=False)
            # rYoloModel5.compile(optimizer='adam', loss='mean_squared_error')
            # rPredictions5 = rYoloModel5.predict(x_test_reg)
            # rRrediction_mean5 = round(rPredictions5.mean() * 180, 2)
            # print("Regression Value 5 : ", str(rRrediction_mean5))

            # rYoloModel6 = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class6.h5', compile=False)
            # rYoloModel6.compile(optimizer='adam', loss='mean_squared_error')
            # rPredictions6 = rYoloModel6.predict(x_test_reg)
            # rRrediction_mean6 = round(rPredictions6.mean() * 220, 2)
            # print("Regression Value 6 : ", str(rRrediction_mean6))

            # # Calculate the average of the two variables
            # final_gluco_value = round((rRrediction_mean5 + rRrediction_mean6) / 2, 2)

            model_filename5 = '/home/gqrp1_preprod/gqrp1_preprod/models/knn_class_5_v2_final_k1_full.pkl'
            # Load the model from the file using pickle
            with open(model_filename5, "rb") as f:
                knnModel5 = pickle.load(f)
            X_test_flattened = [hist.flatten() for hist in x_test]
            rRrediction_mean5 = knnModel5.predict(X_test_flattened)
            rRrediction_mean5 = np.mean(rRrediction_mean5)
            print("Regression Value 5 KNN: ", str(rRrediction_mean5), flush=True)

            model_filename6 = '/home/gqrp1_preprod/gqrp1_preprod/models/knn_class_6_v2_final_k1_full.pkl'
            # Load the model from the file using pickle
            with open(model_filename6, "rb") as f:
                knnModel6 = pickle.load(f)
            X_test_flattened = [hist.flatten() for hist in x_test]
            rRrediction_mean6 = knnModel6.predict(X_test_flattened)
            rRrediction_mean6 = np.mean(rRrediction_mean6)
            print("Regression Value 6 KNN: ", str(rRrediction_mean6), flush=True)

            # -----------------------------------------------------------
            #                Weighted Average Calculation
            # -----------------------------------------------------------
            rRrediction_mean5 = predication_list_values[4] * rRrediction_mean5
            rRrediction_mean6 = predication_list_values[5] * rRrediction_mean6
            final_gluco_value = round((rRrediction_mean5 + rRrediction_mean6) / (predication_list_values[4] + predication_list_values[5]), 2)

            insert_reading = insert_gluco_value_for_patient(user,patient_id,video,final_gluco_value)
            response =  { 
                "patient_id" : insert_reading.patient_id.id,
                "full_name" : insert_reading.patient_id.full_name,
                "age" : insert_reading.patient_id.age,
                "gluco_value" : insert_reading.gluco_value,
            }

            return Response(response, status=status.HTTP_200_OK)
        else:
            return Response({"message": "Test Failed Retake The Video"}, status=status.HTTP_400_BAD_REQUEST)

    if (highest_count_added_classes == '4'):
        if (class_counts[highest_count_added_classes] >= condition_count):

            # rYoloModel4 = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class4.h5', compile=False)
            # rYoloModel4.compile(optimizer='adam', loss='mean_squared_error')
            # rPredictions4 = rYoloModel4.predict(x_test_reg)
            # rRrediction_mean4 = round(rPredictions4.mean() * 140, 2)
            # print("Regression Value 4 : ", str(rRrediction_mean4))

            # rYoloModel5 = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class5.h5', compile=False)
            # rYoloModel5.compile(optimizer='adam', loss='mean_squared_error')
            # rPredictions5 = rYoloModel5.predict(x_test_reg)
            # rRrediction_mean5 = round(rPredictions5.mean() * 180, 2)
            # print("Regression Value 5 : ", str(rRrediction_mean5))

            # # Calculate the average of the two variables
            # final_gluco_value = round((rRrediction_mean4 + rRrediction_mean5) / 2, 2)

            model_filename4 = '/home/gqrp1_preprod/gqrp1_preprod/models/knn_class_4_v2_final_k1_full.pkl'
            # Load the model from the file using pickle
            with open(model_filename4, "rb") as f:
                knnModel4 = pickle.load(f)
            X_test_flattened = [hist.flatten() for hist in x_test]
            rRrediction_mean4 = knnModel4.predict(X_test_flattened)
            rRrediction_mean4 = np.mean(rRrediction_mean4)
            print("Regression Value 4 KNN: ", str(rRrediction_mean4), flush=True)

            model_filename5 = '/home/gqrp1_preprod/gqrp1_preprod/models/knn_class_5_v2_final_k1_full.pkl'
            # Load the model from the file using pickle
            with open(model_filename5, "rb") as f:
                knnModel5 = pickle.load(f)
            X_test_flattened = [hist.flatten() for hist in x_test]
            rRrediction_mean5 = knnModel5.predict(X_test_flattened)
            rRrediction_mean5 = np.mean(rRrediction_mean5)
            print("Regression Value 5 KNN: ", str(rRrediction_mean5), flush=True)

            # -----------------------------------------------------------
            #                Weighted Average Calculation
            # -----------------------------------------------------------
            rRrediction_mean4 = predication_list_values[3] * rRrediction_mean4
            rRrediction_mean5 = predication_list_values[4] * rRrediction_mean5
            final_gluco_value = round((rRrediction_mean4 + rRrediction_mean5) / (predication_list_values[3] + predication_list_values[4]), 2)

            insert_reading = insert_gluco_value_for_patient(user,patient_id,video,final_gluco_value)
            response =  { 
                "patient_id" : insert_reading.patient_id.id,
                "full_name" : insert_reading.patient_id.full_name,
                "age" : insert_reading.patient_id.age,
                "gluco_value" : insert_reading.gluco_value,
            }

            return Response(response, status=status.HTTP_200_OK)
        else:
            return Response({"message": "Test Failed Retake The Video"}, status=status.HTTP_400_BAD_REQUEST)

    if (highest_count_added_classes == '3'):
        if (class_counts[highest_count_added_classes] >= condition_count):

            # rYoloModel3 = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class3.h5', compile=False)
            # rYoloModel3.compile(optimizer='adam', loss='mean_squared_error')
            # rPredictions3 = rYoloModel3.predict(x_test_reg)
            # rRrediction_mean3 = round(rPredictions3.mean() * 120, 2)
            # print("Regression Value 3 : ", str(rRrediction_mean3))

            # rYoloModel4 = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class4.h5', compile=False)
            # rYoloModel4.compile(optimizer='adam', loss='mean_squared_error')
            # rPredictions4 = rYoloModel4.predict(x_test_reg)
            # rRrediction_mean4 = round(rPredictions4.mean() * 140, 2)
            # print("Regression Value 4 : ", str(rRrediction_mean4))

            # # Calculate the average of the two variables
            # final_gluco_value = round((rRrediction_mean3 + rRrediction_mean4) / 2, 2)

            model_filename3 = '/home/gqrp1_preprod/gqrp1_preprod/models/knn_class_3_v2_final_k1_full.pkl'
            # Load the model from the file using pickle
            with open(model_filename3, "rb") as f:
                knnModel3 = pickle.load(f)
            X_test_flattened = [hist.flatten() for hist in x_test]
            rRrediction_mean3 = knnModel3.predict(X_test_flattened)
            rRrediction_mean3 = np.mean(rRrediction_mean3)
            print("Regression Value 3 KNN: ", str(rRrediction_mean3), flush=True)

            model_filename4 = '/home/gqrp1_preprod/gqrp1_preprod/models/knn_class_4_v2_final_k1_full.pkl'
            # Load the model from the file using pickle
            with open(model_filename4, "rb") as f:
                knnModel4 = pickle.load(f)
            X_test_flattened = [hist.flatten() for hist in x_test]
            rRrediction_mean4 = knnModel4.predict(X_test_flattened)
            rRrediction_mean4 = np.mean(rRrediction_mean4)
            print("Regression Value 4 KNN: ", str(rRrediction_mean4), flush=True)

            # -----------------------------------------------------------
            #                Weighted Average Calculation
            # -----------------------------------------------------------
            rRrediction_mean3 = predication_list_values[2] * rRrediction_mean3
            rRrediction_mean4 = predication_list_values[3] * rRrediction_mean4
            final_gluco_value = round((rRrediction_mean3 + rRrediction_mean4) / (predication_list_values[2] + predication_list_values[3]), 2)

            insert_reading = insert_gluco_value_for_patient(user,patient_id,video,final_gluco_value)
            response =  { 
                "patient_id" : insert_reading.patient_id.id,
                "full_name" : insert_reading.patient_id.full_name,
                "age" : insert_reading.patient_id.age,
                "gluco_value" : insert_reading.gluco_value,
            }

            return Response(response, status=status.HTTP_200_OK)
        else:
            return Response({"message": "Test Failed Retake The Video"}, status=status.HTTP_400_BAD_REQUEST)

    if (highest_count_added_classes == '2'):
        if (class_counts[highest_count_added_classes] >= condition_count):

            # rYoloModel2 = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class2.h5', compile=False)
            # rYoloModel2.compile(optimizer='adam', loss='mean_squared_error')
            # rPredictions2 = rYoloModel2.predict(x_test_reg)
            # rRrediction_mean2 = round(rPredictions2.mean() * 100, 2)
            # print("Regression Value 2 : ", str(rRrediction_mean2))

            # rYoloModel3 = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class3.h5', compile=False)
            # rYoloModel3.compile(optimizer='adam', loss='mean_squared_error')
            # rPredictions3 = rYoloModel3.predict(x_test_reg)
            # rRrediction_mean3 = round(rPredictions3.mean() * 120, 2)
            # print("Regression Value 3 : ", str(rRrediction_mean3))

            # # Calculate the average of the two variables
            # final_gluco_value = round((rRrediction_mean2 + rRrediction_mean3) / 2, 2)

            model_filename2 = '/home/gqrp1_preprod/gqrp1_preprod/models/knn_class_2_v2_final_k1_full.pkl'
            # Load the model from the file using pickle
            with open(model_filename2, "rb") as f:
                knnModel2 = pickle.load(f)
            X_test_flattened = [hist.flatten() for hist in x_test]
            rRrediction_mean2 = knnModel2.predict(X_test_flattened)
            rRrediction_mean2 = np.mean(rRrediction_mean2)
            print("Regression Value 2 KNN: ", str(rRrediction_mean2), flush=True)

            model_filename3 = '/home/gqrp1_preprod/gqrp1_preprod/models/knn_class_3_v2_final_k1_full.pkl'
            # Load the model from the file using pickle
            with open(model_filename3, "rb") as f:
                knnModel3 = pickle.load(f)
            X_test_flattened = [hist.flatten() for hist in x_test]
            rRrediction_mean3 = knnModel3.predict(X_test_flattened)
            rRrediction_mean3 = np.mean(rRrediction_mean3)
            print("Regression Value 3 KNN: ", str(rRrediction_mean3), flush=True)

            # -----------------------------------------------------------
            #                Weighted Average Calculation
            # -----------------------------------------------------------
            rRrediction_mean2 = predication_list_values[1] * rRrediction_mean2
            rRrediction_mean3 = predication_list_values[2] * rRrediction_mean3
            final_gluco_value = round((rRrediction_mean2 + rRrediction_mean3) / (predication_list_values[1] + predication_list_values[2]), 2)

            insert_reading = insert_gluco_value_for_patient(user,patient_id,video,final_gluco_value)
            response =  { 
                "patient_id" : insert_reading.patient_id.id,
                "full_name" : insert_reading.patient_id.full_name,
                "age" : insert_reading.patient_id.age,
                "gluco_value" : insert_reading.gluco_value,
            }

            return Response(response, status=status.HTTP_200_OK)
        else:
            return Response({"message": "Test Failed Retake The Video"}, status=status.HTTP_404_NOT_FOUND)

    if (highest_count_added_classes == '1'):
        if (class_counts[highest_count_added_classes] >= condition_count):

            # rYoloModel1 = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class1.h5', compile=False)
            # rYoloModel1.compile(optimizer='adam', loss='mean_squared_error')
            # rPredictions1 = rYoloModel1.predict(x_test_reg)
            # rRrediction_mean1 = round(rPredictions1.mean() * 80, 2)
            # print("Regression Value 1 : ", str(rRrediction_mean1))

            # rYoloModel2 = load_model('/home/gqrp1_preprod/gqrp1_preprod/models/resnetV3normalizedrevised_class2.h5', compile=False)
            # rYoloModel2.compile(optimizer='adam', loss='mean_squared_error')
            # rPredictions2 = rYoloModel2.predict(x_test_reg)
            # rRrediction_mean2 = round(rPredictions2.mean() * 100, 2)
            # print("Regression Value 2 : ", str(rRrediction_mean2))

            # # Calculate the average of the two variables
            # final_gluco_value = round((rRrediction_mean1 + rRrediction_mean2) / 2, 2)

            model_filename2 = '/home/gqrp1_preprod/gqrp1_preprod/models/knn_class_2_v2_final_k1_full.pkl'
            # Load the model from the file using pickle
            with open(model_filename2, "rb") as f:
                knnModel2 = pickle.load(f)
            X_test_flattened = [hist.flatten() for hist in x_test]
            rRrediction_mean2 = knnModel2.predict(X_test_flattened)
            rRrediction_mean2 = np.mean(rRrediction_mean2)
            print("Regression Value 2 KNN: ", str(rRrediction_mean2), flush=True)

            model_filename3 = '/home/gqrp1_preprod/gqrp1_preprod/models/knn_class_3_v2_final_k1_full.pkl'
            # Load the model from the file using pickle
            with open(model_filename3, "rb") as f:
                knnModel3 = pickle.load(f)
            X_test_flattened = [hist.flatten() for hist in x_test]
            rRrediction_mean3 = knnModel3.predict(X_test_flattened)
            rRrediction_mean3 = np.mean(rRrediction_mean3)
            print("Regression Value 3 KNN: ", str(rRrediction_mean3), flush=True)

            # -----------------------------------------------------------
            #                Weighted Average Calculation
            # -----------------------------------------------------------
            rRrediction_mean2 = predication_list_values[1] * rRrediction_mean2
            rRrediction_mean3 = predication_list_values[2] * rRrediction_mean3
            final_gluco_value = round((rRrediction_mean2 + rRrediction_mean3) / (predication_list_values[1] + predication_list_values[2]), 2)

            insert_reading = insert_gluco_value_for_patient(user,patient_id,video,final_gluco_value)
            response =  { 
                "patient_id" : insert_reading.patient_id.id,
                "full_name" : insert_reading.patient_id.full_name,
                "age" : insert_reading.patient_id.age,
                "gluco_value" : insert_reading.gluco_value,
            }

            return Response(response, status=status.HTTP_200_OK)
        else:
            return Response({"message": "Test Failed Retake The Video"}, status=status.HTTP_404_NOT_FOUND)
