import os
import cv2


def segment_videos():
    base_in_dir = 'Dataset\\Videos'
    base_out_dir = 'Dataset\\Images'
    dir_entries = os.listdir(base_in_dir)

    for i, directory in enumerate(dir_entries):
        video_directories = os.listdir(base_in_dir + '\\' + directory)

        for d in video_directories:
            image_directory = d[:d.rfind('.')]
            file_path = os.path.join(base_in_dir, directory, d)
            out_dir = os.path.join(base_out_dir, directory, image_directory) + '\\'
            video = cv2.VideoCapture(file_path)

            i = 0
            # OpenPose frame naming convention
            name = '000000000000'
            while video.isOpened():
                ret, frame = video.read()
                if not ret:
                    break

                n = len(str(i))
                file_name = d[:d.rfind('.')] + '_' + name[0: len(name) - n] + str(i)
                cv2.imwrite(os.path.join(out_dir + file_name + '.jpg'), frame)
                i += 1

            video.release()
            cv2.destroyAllWindows()
