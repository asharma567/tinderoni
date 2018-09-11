def montagify(list_of_PIL_images, resize=(96, 96), tiles=(10, 10)):
    from imutils import build_montages
    import numpy as np
    from tsne import rescaler
    from PIL import Image
    montage = build_montages(
        list_of_PIL_images, 
        resize, 
        tiles
    )[0]

    return Image.fromarray(np.uint8(montage))

def plot_distribution(data_points_list):
    '''
    kurtosis:
    Is about the fatness of the tails which is also indicative out of outliers.
    skew: 
    when a distribution "leans" to the right or left, it's called skewed the right/left. 
    Think of a skewer. This it's a indication of outliers that live on that side of the distribution.
    *these are both aggregate stats and very subjectable to the size of the target sample
    '''
    import pandas as pd
    import matplotlib.pyplot as plt
   
    print(pd.Series(data_points_list).describe())
    
    skew, kurtosis = _skew_and_kurtosis(data_points_list)
    print ('skew -- ', skew)
    print ('kurtosis --', kurtosis)
    
    plot_transformation(data_points_list, 'no_transformation');
    plt.violinplot(
       data_points_list,
       showmeans=False,
       showmedians=True
    );

def _skew_and_kurtosis(data_points_list): 
    from scipy.stats import skew, kurtosis
    return (skew(data_points_list), kurtosis(data_points_list))

def plot_transformation(data, name_of_transformation):
    import matplotlib.pyplot as plt

    #setting up canvas
    figure = plt.figure(figsize=(10,5))
    
    plt.suptitle(name_of_transformation)
    
    figure.add_subplot(121)
    
    plt.hist(data, alpha=0.75, bins=100) 
    
    figure.add_subplot(122)
    plt.boxplot(data)
    
    plt.show()