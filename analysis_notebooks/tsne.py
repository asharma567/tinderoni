'''
- make this into an api
'''

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from tsne_grid_search_tool import find_best_tsne_plot
import numpy as np
from PIL import Image
import os, glob
import pandas as pd


GRID_SEARCHED_PARAMS_FOR_GUYS_FIRST_PIC  = {'perplexity': 45, 'n_iter': 5000, 'init': 'pca', 'learning_rate': 1000, 'early_exaggeration': 6.0}
GRID_SEARCHED_PARAMS_FOR_GUYS  = {'perplexity': 25, 'n_iter': 5000, 'init': 'pca', 'learning_rate': 1000, 'early_exaggeration': 6.0}

GRID_SEARCHED_PARAMS_FOR_GIRLS_FIRST_PIC = {'perplexity': 35, 'n_iter': 5000, 'init': 'pca', 'learning_rate': 1000, 'early_exaggeration': 6.0}



def multithread_map(fn, work_list, num_workers=50):
    from concurrent.futures import ThreadPoolExecutor
    '''
    spawns a threadpool and assigns num_workers to some 
    list, array, or any other container. Motivation behind 
    this was for functions that involve scraping.
    '''
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(fn, work_list))

def image_scatter_plot(images, tsne_scatter_plot, res=500, cval=1.):
    '''
    I:
        feature_M: numpy array
        Features to visualize
        images: list or numpy array
            Corresponding images to feature_M. Expects float images from (0,1).
        img_res: float or int
            Resolution to embed images at
        res: float or int
            Size of embedding image in pixels
        cval: float or numpy array
            Background color value
    O: 
        canvas: numpy array
    '''

    #find the max ratios
    max_width = images[0].shape[0]
    max_height = images[0].shape[1]
    

    xx = tsne_scatter_plot[:, 0]
    yy = tsne_scatter_plot[:, 1]
    
    x_min, x_max = xx.min(), xx.max()
    y_min, y_max = yy.min(), yy.max()
    
    # Fix the ratios
    sx = (x_max - x_min)
    sy = (y_max - y_min)
    if sx > sy:
        res_x = sx / float(sy) * res
        res_y = res
    else:
        res_x = res
        res_y = sy / float(sx) * res

    res_x = int(res_x)
    res_y = int(res_y)
    #create a blank canvas

    canvas = np.ones((res_x + max_width, res_y + max_height, 3)) * cval
    
    x_coords = np.linspace(x_min, x_max, res_x)
    y_coords = np.linspace(y_min, y_max, res_y)
    
    #fill coordinates with the images
    for x, y, image in zip(xx, yy, images):
        width, height = image.shape[:2]
        x_idx = np.argmin((x - x_coords) ** 2)
        y_idx = np.argmin((y - y_coords) ** 2)
        canvas[x_idx:x_idx + width, y_idx:y_idx + height] = image
    return canvas



def rescaler(im_file, size=(128, 128)):
    try:
        im = Image.open(im_file)
        width, height= (im.size)

        
        #checks if the image is already smaller than the thumbnail
        if width < size[0] or height < size[1]: 
            raise IOError
        
        im = im.resize(size, Image.ANTIALIAS)
        return im

    except IOError:
        print ("cannot create thumbnail for '%s'" % im_file)
        return None


def dedupe_numpy_array(original_array):
    
    print (original_array.shape)
    base = np.ascontiguousarray(original_array).view(np.dtype((np.void, original_array.dtype.itemsize * original_array.shape[1])))
    deduped_array = np.unique(base).view(original_array.dtype).reshape(-1, original_array.shape[1])
    print (deduped_array.shape)
    return deduped_array


def dedupe_list_of_obj(list_of_obj):
#refactor thsi
    
    new_list = []
    for img in list_of_obj:
        if img in new_list:
            continue
        new_list.append(img)
    print(len(new_list), len(list_of_obj))
    return new_list

def main(path, save_fname, save_plots_fname):
    

    #rescales
    if type(path) == list:
        print ('=' * 100)
        print ('Looking in the following path--')
        print (path[:1])
        print ('=' * 100)
        
        arrays_rescaled = multithread_map(rescaler, path)

    else:
        print ('=' * 100)
        print ('Looking in the following path--')
        print (path)
        print ('=' * 100)

        arrays_rescaled = multithread_map(rescaler, glob.glob(path))
    
    #this should be a raise error
    if len(arrays_rescaled) == 0: 
        print('no files loaded, check path')
        return None
    
    #clean-up
    arrays_rescaled = [img for img in arrays_rescaled if img != None]
    
    #add flag for this
    arrays_rescaled = dedupe_list_of_obj (arrays_rescaled)
    
    X = np.array(list(map(lambda x: np.array(np.array(x, dtype=np.float32)), arrays_rescaled)))
    arr_pics_feature_matrix = rgb_arrays_to_flattened_feature_matrix(X)
    
    
    #grid search needs to be dropped in here
    # params_dic = find_best_tsne_plot(arr_pics_feature_matrix)
    # save_plots_with_params(params_dic, save_plots_fname)
    
    tsne_embeddings = TSNE().fit_transform(arr_pics_feature_matrix)
    
    plot_data = image_scatter_plot(
                tsne_scatter_plot=tsne_embeddings, 
                images=X,
                res=5000
            )

    print(plot_data.shape)
    im = Image.fromarray(np.uint8(plot_data))
    im.show()
    
    if save_fname:
        im.save(save_fname, 'JPEG')

        

def rgb_arrays_to_flattened_feature_matrix(array_of_images):
    '''
    I: a list of numpy arrays of images eg [np.array(image_file1), np.array(image_file2)]
    O: numpy array of these images flattened into row form
    '''
    
    pics = np.array(array_of_images[0].ravel())
    for index in range(1, len(array_of_images)):
        pics = np.c_[pics, array_of_images[index].ravel()]
    return pics.T


if __name__ == '__main__':

#change this to arg_parse    
    # main(
    #     path='/Users/ajay/Desktop/project_prototypes/tinderoni/tinder_pics/first_pic/*.jpeg', 
    #     save_fname='first_pic_girls.jpeg',
    #     save_plots_fname='first_pic_girls'
    # )

    main(
        path='CF_ALL_faces/*', 
        save_fname='CF_ALL_faces.jpeg',
        save_plots_fname='CF_ALL_faces'
    )