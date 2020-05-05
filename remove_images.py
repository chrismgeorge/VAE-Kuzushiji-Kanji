### Removes all but one example of each Kuzushiji-Kanji

import os
import pdb

def main():
    image_dir = './kmnist/kkanji2/'
    folders = os.listdir(image_dir)
    for folder in folders:
        folder_path = image_dir + folder + '/'
        images = os.listdir(folder_path)
        image_save = None
        for image in images:
            if image_save == None and '.png' in image:
                image_save = image
                continue
            elif '.png' in image:
                image_path = folder_path + image
                os.remove(image_path)

if __name__ == '__main__':
    main()
