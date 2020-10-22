from keras.callbacks import ModelCheckpoint
from keras.models import Model

from PIL import Image
import numpy
import h5py
import cv2
from dp import trainGenerator, testGenerator, saveResult
from model import unet
from history import LossHistory

# config constant
train_path = '/data/cephfs/punim0619/2019dataset/trainingset'
val_path = '/data/cephfs/punim0619/2019dataset/testset'
test_path = '/data/cephfs/punim0619/2019dataset/testset/test_crop'
test_image_type = 'tif'
image_folder = 'train_crop'
val_image_folder = 'test_crop'
mask_folder = 'mask_crop'
image_color_mode = 'grayscale'
mask_color_mode = 'grayscale'
target_size = (256, 256)
batch_size = 8
save_to_dir = None
image_save_prefix  = 'after_train'
mask_save_prefix  = 'after_mask'
seed = 1
result_image_path = '/home/wsha/testing_result_2019/loss.jpg'
model_path = '/home/wsha/testing_result_2019/unet_best.hdf5'
result_save_path = '/home/wsha/testing_result_2019/result'

def main():
    data_gen_args = dict(# featurewise_center=True,
                    # featurewise_std_normalization=True,
                    rotation_range=90,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    # shear_range=0.05,
                    zoom_range=0.05,
                    # horizontal_flip=True,
                    # vertical_flip=True,
                    fill_mode='nearest')

    img_gen_arg_dict = dict(directory = train_path,
                        classes = [image_folder],
                        class_mode = None,
                        shuffle = False,
                        color_mode = image_color_mode,
                        target_size = target_size,
                        batch_size = batch_size,
                        save_to_dir = save_to_dir,
                        save_prefix  = image_save_prefix,
                        seed = seed)

    mask_gen_arg_dict = dict(directory = train_path,
                        classes = [mask_folder],
                        class_mode = None,
                        shuffle	= False,
                        color_mode = mask_color_mode,
                        target_size = target_size,
                        batch_size = batch_size,
                        save_to_dir = save_to_dir,
                        save_prefix  = mask_save_prefix,
                        seed = seed)
    val_img_gen_arg_dict = dict(directory = val_path,
                        classes = [val_image_folder],
                        class_mode = None,
                        shuffle	= False,
                        color_mode = image_color_mode,
                        target_size = target_size,
                        batch_size = batch_size,
                        save_to_dir = save_to_dir,
                        save_prefix  = image_save_prefix,
                        seed = seed)
    val_mask_gen_arg_dict = dict(directory = val_path,
                        classes = [mask_folder],
                        class_mode = None,
                        shuffle	= False,
                        color_mode = mask_color_mode,
                        target_size = target_size,
                        batch_size = batch_size,
                        save_to_dir = save_to_dir,
                        save_prefix  = mask_save_prefix,
                        seed = seed)
                        
    # augmentation = ImageDataGenerator(data_gen_args)
    # x_train = get_input(train_path,image_folder)
    # y_train = get_output(train_path,mask_folder)
    # x_val = get_input(val_path,val_image_folder)
    # y_val = get_output(val_path,mask_folder)
    myGene = trainGenerator(data_gen_args, img_gen_arg_dict, mask_gen_arg_dict, train_path, image_folder, mask_folder)
    valGene = trainGenerator(data_gen_args, val_img_gen_arg_dict, val_mask_gen_arg_dict, val_path, val_image_folder, mask_folder)

    history = LossHistory()
    model = unet()
    model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    model.fit_generator(myGene, steps_per_epoch=30, epochs=10, validation_data=valGene, validation_steps=30, callbacks=[model_checkpoint, history])

    history.loss_plot('epoch', result_image_path)
    print('result saved')

    testGene = testGenerator(test_path, test_image_type)
    results = model.predict_generator(testGene, steps = 96, verbose=0)
    saveResult(result_save_path, results)

if __name__ == '__main__':
    main()
