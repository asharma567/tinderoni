from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from tsne_grid_search_tool import find_best_tsne_plot, save_plots_with_params
import numpy as np
from PIL import Image
import os, glob


GRID_SEARCHED_PARAMS_FOR_GUYS_FIRST_PIC  = {'perplexity': 45, 'n_iter': 5000, 'init': 'pca', 'learning_rate': 1000, 'early_exaggeration': 6.0}


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

    #create a blank canvas
    canvas = np.ones((res_x + max_width, res_y + max_height, 3)) * cval
    
    x_coords = np.linspace(x_min, x_max, res_x)
    y_coords = np.linspace(y_min, y_max, res_y)
    
    #fill coordinates with the images
    for x, y, image in zip(xx, yy, images):
        w, h = image.shape[:2]
        x_idx = np.argmin((x - x_coords) ** 2)
        y_idx = np.argmin((y - y_coords) ** 2)
        canvas[x_idx:x_idx + w, y_idx:y_idx + h] = image
    return canvas

def rescaler(im_file):
    try:
        im = Image.open(im_file)
        x,y=(im.SIZE)
        if x < SIZE[0] or y < SIZE[0]: 
            raise IOError
        im = im.resize(SIZE, Image.ANTIALIAS)
        return im

    except IOError:
        print "cannot create thumbnail for '%s'" % im_file
        return None

#parameters
INPUT_RES = 3000
SIZE = 32, 32

def main(path, save_fname, save_plots_fname):
    
    
    print '=' * 100
    print 'Looking in the following path--'
    print path
    print '=' * 100


    #rescales
    arrays_rescaled = multithread_map(rescaler, glob.glob(path))
    
    #clean-up
    arrays_rescaled = [img for img in arrays_rescaled if img != None]

    X = map(lambda x: np.array(np.array(x, dtype=np.float32)), arrays_rescaled)
    pics_feature_matrix = rgb_arrays_to_flattened_feature_matrix(X)
    
    #grid search needs to be dropped in here
    params_dic = find_best_tsne_plot(pics_feature_matrix)
    save_plots_with_params(params_dic,save_plots_fname)
    
    # tsne_embeddings = TSNE(**GRID_SEARCHED_PARAMS_FOR_GUYS_FIRST_PIC).fit_transform(pics_feature_matrix)
    
    # plot_data = image_scatter_plot(
    #         tsne_scatter_plot=tsne_embeddings, 
    #         images=X, 
    #         res=INPUT_RES
    #         )

    # print(plot_data.shape)
    # im = Image.fromarray(np.uint8(plot_data))
    # im.show()
    
    # if save_fname:
    #     im.save(save_fname, 'JPEG')

        

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
    main(
        path='/Users/ajay/Desktop/project_prototypes/tinderoni/guys/*.jpeg', 
        save_fname='testing.jpeg',
        save_plots_fname='guys_all'
    )
