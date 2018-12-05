def made_readable_df(probas, labels=None, index=None):
    import pandas as pd
    
    df = pd.DataFrame(probas).applymap(lambda x: '%.3f' % x)

    if labels:
        df.columns = labels
    if index:
        df.index = map(lambda x: x.split('/')[-1].replace('_face.jpg',''), index)
    return df

def get_multi_label_roc_score(input_labels, X, instatiated_clf):
    '''
    DO NOT labeled 1,0 if so just change it to string instead of int
    
    I: labels (pd.Series), Feature matrix (DataFrame), classifier 
    O: weighted average roc auc score of all labels

    Each class is first binarized and scored stepe-wise. 
    Then it calculates the weighted average roc auc for each class.
    '''
    from sklearn.cross_validation import StratifiedKFold, cross_val_score    
    import numpy as np

    output_list = []
    all_class_labels = dict(input_labels.value_counts(1))
    for label_tag in all_class_labels.keys():

        # binarize labels
        temp_label = input_labels.copy()
        
        # this sequence matters
        temp_label[temp_label != label_tag] = 0
        temp_label[temp_label == label_tag] = 1
        Y_train = np.asarray(temp_label, dtype=int)

        # score
        skf = StratifiedKFold(Y_train, n_folds=10, shuffle=False)
        scores = cross_val_score(instatiated_clf, X, Y_train, cv=skf, scoring='roc_auc')
        output_list.append((all_class_labels[label_tag], scores.mean()))
    
    return sum([weight*score for weight, score in output_list])

def score_classifier(clf, feature_M, labels, class_imbalance=True):
    from sklearn.cross_validation import StratifiedKFold, cross_val_score    
    import numpy as np

    
    #scoring mechanism
    if class_imbalance:
        skf = StratifiedKFold(labels, n_folds=10, shuffle=True)
    
    #put the else here for non-strat
    scores_for_each_fold = cross_val_score(
        clf, 
        feature_M, 
        labels, 
        cv=skf, 
        scoring='roc_auc'
    )
    
    median = np.median(scores_for_each_fold)
    mean = np.mean(scores_for_each_fold)
    std = np.std(scores_for_each_fold)
    
    return mean, median, std, scores_for_each_fold


def append_new_dataset(orig_paths, orig_encodings, orig_arrays_rescaled, orig_X, newpath):
    import numpy as np

    data_, paths_, encodings_, arrays_rescaled_, X_ = load_encodings(newpath)

    new_arrays_rescaled = orig_arrays_rescaled + arrays_rescaled_ 
    new_encodings = orig_encodings + encodings_ 
    new_paths = orig_paths + paths_
    new_X = np.concatenate((orig_X, X_))

    print (len(new_encodings), len(new_paths), len(new_arrays_rescaled), new_X.shape[0])

    return new_paths, new_encodings, new_arrays_rescaled, new_X

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
    for x, y in [(10,10), (6,6), (5,5),(4,4), (3, 3), (2,2), (2,1)]: 
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
    '''
    I: pickled images
    O: raw_data, paths, encodings, arrays_rescaled, X
    '''
    import pickle
    import numpy as np
    from tsne import rescaler
    raw_data = np.array(pickle.loads(open(file_name, "rb").read()))
    paths = ['../data/' + d['imagePath']  for d in raw_data]    
    # paths = [d['imagePath']  for d in raw_data]    
    encodings = [d["encoding"] for d in raw_data]

    arrays_rescaled = list(map(rescaler, paths))
    X = np.array(list(map(lambda x: np.array(np.array(x, dtype=np.float32)), arrays_rescaled)))

    return raw_data, paths, encodings, arrays_rescaled, X


def map_colors_to_ratings_tsne(colors_dic, encodings, colors_pal):
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

def map_colors_to_ratings_UMAP(colors_dic, encodings, colors_pal, params=None):
    from umap import UMAP
    if not params:
        params = {
            #5 to 50, with a choice of 10 to 15 being a sensible default.
            "n_neighbors":5,
            #0.001 to 0.5, with 0.1 being a reasonable default.
     
            "min_dist":0.1,

            #metric: This determines the choice of metric used to measure distance in the input space. 
            "metric":'euclidean'
        }

    umap_embeddings = UMAP(**params).fit_transform(encodings)

    xx = umap_embeddings[:, 0]
    yy = umap_embeddings[:, 1]

    lis=[]
    for depth in colors_dic:
        if depth: 
            lis.append(colors_pal[depth - 1])
        else:
            lis.append(colors_pal[0])

    return umap_embeddings, xx, yy, lis



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