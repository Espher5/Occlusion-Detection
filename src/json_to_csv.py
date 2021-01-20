import pandas as pd
import os
import json
import csv
import math


def json_to_csv():
    base_json_directory = 'Dataset\\JSON\\'
    base_csv_directory = 'Dataset\\Csv\\'
    test_directories = ['1_003', '2_003', '3_003', '4_003', '5_003', '6_003', '8A.015', '9A.015', '10_008', '11_008']
    val_directories = ['1_002', '2_002', '3_002', '4_002', '5_002', '6_002', '8A.007', '9A.007', '10_007', '11_007']

    outdoor_directories = ['10_007', '10_008', '10_009', '10_012', '10_014', '10_015', '10_024', '10_027', '10_039',
                           '10_049', '11_007', '11_008', '11_009', '11_012', '11_014', '11_015', '11_024', '11_027',
                           '11_039', '11_049']
    good_frames = 0
    bad_frames = 0

    train_csv = open(os.path.join(base_csv_directory, 'train.csv'), mode='w', newline='')
    train_writer = csv.writer(train_csv)
    test_csv = open(os.path.join(base_csv_directory, 'test.csv'), mode='w', newline='')
    test_writer = csv.writer(test_csv)
    val_csv = open(os.path.join(base_csv_directory, 'val.csv'), mode='w', newline='')
    val_writer = csv.writer(val_csv)

    entries = os.listdir(base_json_directory)
    for directory in entries:
        keypoint_threshold = 6
        accuracy_threshold = 0.6
        if directory in outdoor_directories:
            keypoint_threshold = 14
            accuracy_threshold = 0.7

        json_files = os.listdir(os.path.join(base_json_directory, directory))

        for file in json_files:
            path = os.path.join(base_json_directory, directory, file)

            with(open(path, 'rb')) as data_file:
                data = json.load(data_file)
                json_struct = pd.read_json(path)

                # OpenPose has detected more than a person in the frame
                if json_struct.shape[0] > 1:
                    max_distance = -1
                    num_of_elements = json_struct.shape[0]
                    best = 1
                    for i in range(num_of_elements):
                        if data['people'][i - 1]['pose_keypoints_2d'][3] != 0 and \
                                data['people'][i - 1]['pose_keypoints_2d'][24] != 0:
                            p1 = [data['people'][i - 1]['pose_keypoints_2d'][3],
                                  data['people'][i - 1]['pose_keypoints_2d'][4]]
                            p2 = [data['people'][i - 1]['pose_keypoints_2d'][24],
                                  data['people'][i - 1]['pose_keypoints_2d'][25]]
                            distance = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
                            if max_distance < distance:
                                max_distance = distance
                                best = i
                    frame = data['people'][best - 1]['pose_keypoints_2d']
                elif json_struct.shape[0] == 1:
                    frame = data['people'][0]['pose_keypoints_2d']

            # Determines whether the frame is good or bad
            if frame is not None:
                i = 0
                detected_keypoints = 0
                while i < len(frame):
                    x = frame[i]
                    y = frame[i + 1]
                    acc = frame[i + 2]
                    i += 3
                    if x != 0 and y != 0 and acc > accuracy_threshold:
                        detected_keypoints += 1

                if detected_keypoints >= keypoint_threshold:
                    # Good frame
                    label = 1
                    good_frames += 1
                else:
                    # Bad frame
                    label = 0
                    bad_frames += 1
                image_name = directory + '/' + file.replace(base_json_directory, '').replace('_keypoints.json', '')

                if directory in test_directories:
                    test_writer.writerow([image_name] + [label])
                elif directory in val_directories:
                    val_writer.writerow([image_name] + [label])
                else:
                    train_writer.writerow([image_name] + [label])
    print('Good frames: {}. Bad frames: {}'.format(good_frames, bad_frames))
    train_csv.close()
    test_csv.close()
    val_csv.close()
