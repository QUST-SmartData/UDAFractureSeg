import os


def save_file_list(directory, output_filename):
    image_files = [file for file in os.listdir(directory) if os.path.splitext(file)[-1] == '.png']
    with open(os.path.join(output_directory, output_filename), 'w') as f:
        for image_file in image_files:
            f.write(f"{image_file}\n")


if __name__ == '__main__':
    directories = [
        ('road/data/images', 'road_train_list.txt'),
        ('core/data/images', 'core_train_list.txt'),
        ('core/test/images', 'core_test_list.txt'),
        ('cfd/train/images', 'cfd_train_list.txt'),
    ]

    output_directory = './'

    for image_directory, output_filename in directories:
        save_file_list(image_directory, output_filename)
