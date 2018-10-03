def find_the_closest_pts(encodings, arrays_rescaled, data_pt, top_x=5):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    df = pd.DataFrame(encodings)

    mediod = df.median(axis=0)
    centroid = df.mean(axis=0)
    
    df['dist_from_datapt'] = df[:].sub(np.array(data_pt)).pow(2).sum(1).pow(0.5)
    plot_distribution(df['dist_from_datapt'])
    plt.show()

    args = np.argsort(df['dist_from_datapt'])[:top_x]
    display(montagify([np.array(arrays_rescaled[arg]) for arg in args],
            (128,128), 
            tiles=_square_sizer(10)))
    
    
def display_cluster_montages(cluster_labels, arrays_rescaled, rating_dic=None):
    import pandas as pd
    import numpy as np
    
    for cluster_idx, val in zip(pd.Series(cluster_labels).value_counts().index, pd.Series(cluster_labels).value_counts()):
        if cluster_idx  == -1: continue
        
        args = np.where(cluster_labels == cluster_idx)[0]
        
        # display
        if rating_dic:
            print (cluster_idx, val, [rating_dic[paths[arg]] for arg in args])
        else:
            print (cluster_idx, val)
        x, y = _square_sizer(val)
        display(
            montagify([np.array(arrays_rescaled[arg]) for arg in args],
            (128,128), 
            tiles=(x,y))
        )


def _square_sizer(val):
    for x, y in [(4,4), (3, 3), (2,2), (2,1)]: 
        if val == (x * y):
            return x, y
        if val > x * y :
            break
    px, py = x, y
    return px, py


def montagify_indices(data, colors_dic, ratings_set={1,2}):
    import cv2
    indices = [i for i, c_ in enumerate(colors_dic) if c_ in ratings_set]
    # loop over the sampled indexes
    faces = []
    for i in indices:
        # load the input image and extract the face ROI
        image = cv2.imread(data[i]["imagePath"])

        try:
            (top, right, bottom, left) = data[i].get("loc")
        except: 
            continue

        face = image[top:bottom, left:right]

        # force resize the face ROI to 96x96 and then add it to the
        # faces montage list
        face = cv2.resize(face, (96, 96))
        faces.append(face)

    return montagify(faces, (96, 96), (10, 10))

def load_encodings(file_name):
    import pickle
    import numpy as np
    from tsne import rescaler
    data = np.array(pickle.loads(open(file_name, "rb").read()))
    paths = [d['imagePath']  for d in data]
    encodings = [d["encoding"] for d in data]
    arrays_rescaled = list(map(rescaler, paths))
    X = np.array(list(map(lambda x: np.array(np.array(x, dtype=np.float32)), arrays_rescaled)))

    return data, paths, encodings, arrays_rescaled, X


def map_colors_to_ratings(colors_dic, encodings, colors_pal):
    from sklearn.manifold import TSNE

    params = {
        'init':'pca',
        'n_iter': 5000,
        'learning_rate': 1000,
        'early_exaggeration': 6.0,
        # 'perplexity': 10

    }

    tsne_embeddings = TSNE(**params).fit_transform(encodings)

    xx = tsne_embeddings[:, 0]
    yy = tsne_embeddings[:, 1]

    lis=[]
    for depth in colors_dic:
        if depth: 
            lis.append(colors_pal[depth - 1])
        else:
            lis.append(colors_pal[0])

    return tsne_embeddings, xx, yy, lis



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