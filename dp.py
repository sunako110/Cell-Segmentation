from keras.preprocessing.image import ImageDataGenerator
import glob as gb
import os
import numpy as np
import skimage.io as io
from PIL import Image
import cv2
import re
import skimage.transform as trans

def cut(from_path, to_path, file_name,vx,vy):

    name1 = os.path.join(from_path,file_name)
    name2 = to_path + "/crop_pic/" + os.path.splitext(file_name)[0]+"_c"
    im =Image.open(name1)
    width,height = im.size

    dx = 256
    dy = 256
    n = 1

    x1 = 0
    y1 = 0
    x2 = vx
    y2 = vy

    while y2 <= height:

        while x2 <= width:
            name3 = name2 + str(n) + os.path.splitext(file_name)[1]
            # print n,x1,y1,x2,y2
            im2 = im.crop((x1, y1, x2, y2))
            im2.save(name3)
            x1 = x1 + dx
            x2 = x1 + vx
            n = n + 1
        y1 = y1 + dy
        y2 = y2 + vy
        x1 = 0
        x2 = vx

    return n-1

def sorter(path):
    file = os.path.split(path)
    digits = re.split("\D",file[1])
    # score = int(digits[1])*100 + int(digits[2])*10 + int(digits[4])
    score = 0
    for entry in digits:
        if entry:
            if score == 0:
                score = score + int(entry) * 10000
            elif score % 10000 == 0:
                score = score + int(entry) * 100
            elif score % 100 == 0:
                score = score + int(entry)
    return score

def adjustData(img, mask):
    img = img / 255.0
    # retval, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
    # mask = mask / 255.0
    mask[mask < 127.5] = 1.0
    mask[mask >= 127.5] = 0.0
    # np.set_printoptions(threshold = np.inf)
    # print(mask)
    # count1 = 0
    # count0 = 0
    # for i in range(mask.shape[0]):
    #     for j in range(256):
    #         for k in range(256):
    #             if mask[i][j][k] == [1]:
    #                 count1 +=1
    #             if mask[i][j][k] == [0]:
    #                 count0 += 1
    # print(count1,count0)
    return (img, mask)

def get_input(input_path,image_folder):
    input_img_path = os.path.join(input_path,image_folder)
    input_dataset = np.array()
    for img in os.listdir(input_img_path):
    	input_data = np.array(Image.open(os.path.join(input_img_path,img)))
    	input_data = input_data /255.0
    	input_dataset = np.append(input_dataset,input_data,axis=0)
    return input_dataset

def get_output(mask_path,image_folder):
    mask_img_path = os.path.join(mask_path,image_folder)
    mask_dataset = np.array()
    for img in os.listdir(mask_img_path):
    	mask_data = np.array(Image.open(os.path.join(mask_img_path,img)))
    	mask_data[mask_data < 127.5] = 1
    	mask_data[mask_data >= 127.5] = 0
    	mask_dataset = np.append(mask_dataset,mask_data,axis=0)
    return mask_dataset

def trainGenerator(aug_dict, img_gen_arg_dict, mask_gen_arg_dict, path, img_folder, mask_folder):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"

        Usage: myGene = trainGenerator(data_gen_args, img_gen_arg_dict, mask_gen_arg_dict)
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    img_path = os.path.join(path,img_folder)
    mask_path = os.path.join(path,mask_folder)

    images_x = []
    images_y = []
    
    for file in os.listdir(img_path):
        if file.endswith('.tif'):
            open = cv2.imread(os.path.join(img_path, file))
            images_x.append(cv2.cvtColor(open,cv2.COLOR_BGR2GRAY))
    for file in os.listdir(mask_path):
        if file.endswith('.tif'):
            open = cv2.imread(os.path.join(mask_path, file))
            images_y.append(cv2.cvtColor(open, cv2.COLOR_BGR2GRAY))
    images_x = np.asarray(images_x)
    images_y = np.asarray(images_y)

    images_x = images_x.reshape(images_x.shape[0], 256, 256, 1)
    images_y = images_y.reshape(images_y.shape[0], 256, 256, 1)

    image_datagen.fit(images_x, augment=True, seed=1)
    mask_datagen.fit(images_y, augment=True, seed=1)

    image_generator = image_datagen.flow_from_directory(**img_gen_arg_dict)
    mask_generator = mask_datagen.flow_from_directory(**mask_gen_arg_dict)

    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img, mask = adjustData(img, mask)
        yield (img, mask)


def testGenerator(test_path, test_image_type, target_size=(256, 256)):

    # for file in os.listdir(test_path):
           # if os.path.splitext(file)[1] == ('.'+test_image_type):
               # cut(test_path,result_path, file, 256, 256)

    # path = os.path.join('/Users/who_iam/Documents/Master of IT/Stem Cells Project/2019dataset/predict/test/*.tif')
    path = test_path+"/*."+test_image_type
    print(path)
    test_images = sorted(gb.glob(path), key=sorter)
    img_array = []
    for test_image in test_images:
        if test_image.endswith('.tif'):
            open = cv2.imread(test_image)
            img = cv2.cvtColor(open, cv2.COLOR_BGR2GRAY)
            img = img / 255.0
            # img = cv2.resize(img, (256,256))
            img = np.reshape(img, img.shape + (1,))
            img = np.reshape(img, (1,) + img.shape)
            yield img
            # img_array.append(img)
    # img_data = np.asarray(img_array)
    # img_data = img_data.reshape(img_data.shape[0],256,256,1)
    # return img_data


def saveResult(save_path, npyfile, target_size=(512, 512)):
    for i, img_array in enumerate(npyfile):
        # img_array = img_array[0] if len(img_array.shape) == 4 else img_array
        # img_array = img_array[:, :, 0] if len(img_array.shape) == 3 else img_array
        img_array = img_array * 255
        # img_array = img_array.astype(np.uint8())
        if i % 12 == 0:
            img1 = img_array
        elif i % 12 == 4:
            img2 = img_array
        elif i % 12 == 8:
            img3 = img_array
        elif i % 12 < 4:
            img1 = np.concatenate((img1, img_array), axis=1)
        elif i % 12 < 8:
            img2 = np.concatenate((img2, img_array), axis=1)
        elif i % 12 < 12:
            img3 = np.concatenate((img3, img_array), axis=1)
            if i % 12 == 11:
                img = np.vstack((img1, img2))
                img = np.vstack((img, img3))
                # retval, im_at_fixed = cv2.threshold(img, 65, 255, cv2.THRESH_BINARY)
                cv2.imwrite(os.path.join(save_path, "mask/%d_predict.tif" % int(i / 11)), img)
        cv2.imwrite(os.path.join(save_path, "%d_predict.tif" % (i)), img_array)


