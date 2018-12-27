from sklearn.manifold import TSNE
from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

'''
todo
----
- use class structure and replace the global variable feature_M
'''

grid_search_params = {
    
    # inits with PCA gives a better global structure
    'init': ['pca', 'random'],
    
    # 'precomputed' is also an option or 
    # also passing in a custom dist function
    'metric': ['euclidean'],
    
    #increases/decreases accuracy for the Barnes hut algo 
    'angle': [0.2, 0.5, 0.8],
    
    #defaults at 1000 but 5000 is known to work the best
    'n_iter': [3000, 5000],
    'learning_rate': [100, 500, 1000],
    'early_exaggeration': [2.0, 4.0, 6.0],
    
    #this actually affects the inputs to the cost-function
    'perplexity': [10, 35, 50]
}

def multithread_map(fn, work_list, num_workers=50):
    
    '''
    spawns a threadpool and assigns num_workers to some 
    list, array, or any other container. Motivation behind 
    this was for functions that involve scraping.
    '''
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        return list(executor.map(fn, work_list))


def greedily_search_for_lowest_KL(perplexity_param):
    best_kl_error = None
    best_params = None
    current_params = {}
    current_params['perplexity'] = perplexity_param
    current_params['init'] = 'pca'
    
    #find the lowest KL 
    for n_iter in grid_search_params['n_iter']:
        current_params['n_iter'] = n_iter
        
        #walk through all params related to tuning the cost-function            
        for learning_rate in grid_search_params['learning_rate']:
            current_params['learning_rate'] = learning_rate

            for early_exaggeration in grid_search_params['early_exaggeration']:
                current_params['early_exaggeration'] = early_exaggeration

                for angle in grid_search_params['angle']:
                    current_params['angle'] = angle
                

                    #fit tsne and return the embeddings & kl error
                    fitted_tsne = TSNE(**current_params).fit(feature_M)
                    embeddings, curr_error = fitted_tsne.embedding_, fitted_tsne.kl_divergence_
                    #greedy search tool
                    if not best_kl_error or best_kl_error > curr_error:
                        best_kl_error = curr_error
                        best_params = current_params
                        best_embeddings = embeddings
                    
    return best_params, best_embeddings, best_kl_error

def find_best_tsne_plot(feature_M_input, verbose=None):
    
    '''
    For every perplexity level we should optimize for the hyperparameters 
    for the lowest KL-Divergence
    '''
    
    #this global was added because the way the multithreading function was designed
    global feature_M
    feature_M = feature_M_input
    # print multithread_map(greedily_search_for_lowest_KL, grid_search_params['perplexity'])
    stor_of_output={}
    outputs_from_threads =  multithread_map(greedily_search_for_lowest_KL, grid_search_params['perplexity'])
    for best_params, embeddings, error in outputs_from_threads: 
        stor_of_output['perp_' + str(best_params['perplexity'])] = best_params, embeddings, error
    
    return  stor_of_output

if __name__ == '__main__':
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    
    # df_ratings_sample = pd.read_csv('ratings_sample.tsv', sep='\t')
    
    # features = [
    #     'player_age',
    #     'player_height',
    #     'player_age_max',
    #     'player_age_min',
    #     'player_distance_max',
    #     'player_height_min',
    #     'player_height_max',
    #     'subject_age',
    #     'subject_height',
    #     'subject_age_max',
    #     'subject_age_min',
    #     'subject_distance_max',
    #     'subject_height_min',
    #     'subject_height_max',
    #     'distance',
    #     'player_saved',
    #     'player_rated',
    #     'subject_saved',
    #     'subject_rated',
    #     'like'
    # ]


    # labels = df_ratings_sample.like == 1.0
    # X = df_ratings_sample[features]
    # X = X[X.like == 1.0]
    df_anomalies = pd.read_csv('df_anomalies')
    X = scaler.fit_transform(df_anomalies)

    dics = find_best_tsne_plot(X, verbose=True)
    
    for perp, vals in dics.items():
        
        #print plot for every configuration
        best_params, embeddings, error = vals
        print (best_params, perp, error)
        plt.scatter(embeddings[:,0], embeddings[:,1], alpha=.2, color='red')
        plt.show()