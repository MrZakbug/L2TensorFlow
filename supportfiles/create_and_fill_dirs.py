import os
import random
from pathlib import Path
from shutil import copyfile


def create_dirs(main, *subnames):
    main_path = os.path.join(os.getcwd(), 'supportfiles', main)
    os.mkdir(main_path)

    train_path = os.path.join(main_path, 'training')
    validation_path = os.path.join(main_path, 'testing')
    os.mkdir(train_path)
    os.mkdir(validation_path)

    for name in subnames:
        os.mkdir(os.path.join(train_path, name))
        os.mkdir(os.path.join(validation_path, name))

    return train_path, validation_path


def fill_dirs(source, dir_name, split_size):
    # creates a directory structure and gets training and validation dirs path
    training, validation = create_dirs(dir_name, *os.listdir(source))

    for directory in os.listdir(source):
        source_files = os.listdir(os.path.join(source, directory))
        print(''.join(['Source directory ', directory, ' contains ',
                       str(len(source_files)), ' files']))

        train_size = split_size * len(source_files)  # how many items will go to training dir
        target_path = os.path.join(training, directory)
        shuffled_list = random.sample(source_files, len(source_files))  # shuffles list randomly
        print(target_path)
        while len(os.listdir(target_path)) < train_size:  # while training directory has less then defined requested amount of files
            for file in shuffled_list:
                file_path = os.path.join(source, directory, file)
                target_file_path = os.path.join(target_path, file)
                if os.path.getsize(file_path) > 0:  # check if file is not empty
                    copyfile(file_path, target_file_path)
                shuffled_list.remove(file)  # removes already copied or empty files from the list

        for file in shuffled_list:  # works on the items left in the list
            file_path = os.path.join(source, directory, file)
            target_file_path = os.path.join(validation, directory, file)
            if os.path.getsize(file_path) > 0:
                copyfile(file_path, target_file_path)

    return None


if __name__ == "__main__":
    fill_dirs(os.path.join(Path.home(), 'Downloads/kagglecatsanddogs_3367a/PetImages'), 'cats_vs_dogss', 0.9)
