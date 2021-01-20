import os
import cv2

base_in_dir = 'Dataset\\Images'


def rename_images():
    dir_entries = os.listdir(base_in_dir)

    for directory in dir_entries:
        image_directories = os.listdir(base_in_dir + '\\' + directory)
        for d in image_directories:
            files_path = os.listdir(os.path.join(base_in_dir, directory, d))
            for f in files_path:
                old_path = os.path.join(base_in_dir, directory, d, f)
                new_path = os.path.join(base_in_dir, directory, d,
                                        f.replace('.mp4', '')
                                        .replace('.MOV', '')
                                        .replace('.mov', ''))
                os.rename(old_path, new_path)


def rotate_images():
    dir_entries = os.listdir(base_in_dir)

    for directory in dir_entries:
        image_directories = os.listdir(base_in_dir + '\\' + directory)
        for image_file in image_directories:
            path = base_in_dir + '\\' + directory + '\\' + image_file
            source = cv2.imread(path)
            img = cv2.rotate(source, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(path, img)
