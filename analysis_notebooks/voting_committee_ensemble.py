'''
voter1.find_full_consensus_examples(0.5, 0.5)

# motivation:

difficulty of seeing a pattern with all the classifiers. Upon training and 
minor tuning of some classifiers, I've seen strong performance from all of them however, 
they all gave very different example-by example probability distributions.

As such, it was difficult to use a classifier to discover a pattern among photos: 

RF 95 roc auc
KNN 98 

had wildly different top deciles more i.e. it was just difficult to interpret the results. 
LR was much more sensible in tis regard.

To eradicate this confusion we can ensemble the models and get consensus vote. 
The most agreeable will be the patterns; the disagreeable will be noteworthy but 
for purposes of this exercise weighed in on less.

This method is somewhat analogous to crowdsourcing and averaging which took for 
attractiveness rating with that one chinese PhD project.

Psuedocode

clf score all pts
customize threshold across all 3 s.t. 0.4-0.6 are uncertain < 0.4 bottom decile; 
these are the portions which we'll inspect by hand.

once we have the thresholds defined, (plot probability 
distributions overlayed or side by side)

take_consensus_voted examples e.g. 1,1,1 and 0,0,0 will 
be agreeables the rest diseagreeable

in addition, 

compute_controversials
    using std, MAD1, MAD2
    
compute_ensembled_vote
    take the weighted average score for each example; 
    weights being the classifiers strength

'''

class model_voting_comittee(object):
    def __init__(self, list_of_fitted_models, X):
    
        for model in list_of_fitted_models:
            print (model)

        self.list_of_fitted_models = list_of_fitted_models 
        
        self.feature_M = X
        
        self.score_all_points()
    
    def score_all_points(self):
        self.df_probas = pd.DataFrame()
        
        for model in self.list_of_fitted_models:
            
            #this might get deprecated as we have multiple of the same model
            name = model.__str__().split('(')[0]
            
            
            self.df_probas[name] = pd.Series(model.predict_proba(encodings)[:,1])

    def find_full_consensus_examples(self, neg_threshold=0.5, pos_threshold=0.5):
        '''
        motivation for this technique
        '''
        self._compute_threshold_df(neg_threshold, pos_threshold)
        total = len(self.list_of_fitted_models)
        
        srs = self.df_threshold_votes.dropna().sum(axis=1)
        consensus_pos_indices = srs[srs == total].index
        consensus_neg_indices = srs[srs == 0.0].index
        
        return consensus_pos_indices, consensus_neg_indices

    def _compute_threshold_df(self, neg_threshold, pos_threshold):
        self.df_threshold_votes = pd.DataFrame()
        
        for col_name in self.df_probas.columns:
            print (col_name)
            self.df_threshold_votes[col_name] = pd.Series(self.df_probas[col_name] >= pos_threshold).astype(int)
            indices = self.df_probas[col_name].between(neg_threshold, pos_threshold, inclusive=False)
            self.df_threshold_votes[col_name].loc[indices] = np.nan

        
    
    
    def plot_all_distributions(self):
        for col_name in df_probas.columns:
            print (col_name)
            self.df_probas[col_name].hist(alpha=0.3,bins=10)
            plt.show()
            plt.violinplot(
                   self.df_probas[col_name],
                   showmeans=False,
                   showmedians=True
                )
            plt.show()
            


    def compute_ensemble_vote(self, roc_auc_scores):
        self._apply_weights(roc_auc_scores)
        return self.df_probas_wtd.mean(axis=1)
        
    def _apply_weights(self, weights):
        self.df_probas_wtd = self.df_probas[:]


        for idx, col in enumerat(self.df_probas_wtd.columns):
            self.df_probas_wtd[col] = self.df_probas[col].apply(lambda x:x * weights[idx])
        
        pass

        

        
from numpy import mean, absolute, std, median

def mad(data, axis=None):
    return mean(absolute(data - mean(data, axis)), axis)

def mad2(data, axis=None):
    return median(absolute(data - median(data, axis)), axis)

def outlier_scorer(vector_of_points):
    '''
    This returns an absolute score of a data point's outlierness regardless 
    of directionality meaning it's possible to have a largest outlier score 
    and it not be the best deal. 

    ALSO has a shortcoming of finding the true outlierness score and 
    this is designed to address that issue. It measures a point's distance 
    from the regression line not the distance from other points.

    USE-CASE: "deal-score" of a point. Say if we're looking at two independent datasets
    We could make the claim, that one deal is better than other based on this measure.
    '''

    gaps_to_closest_neighbor_of_each_point = [
            find_distance_of_closest_neighbor(i, vector_of_points) \
            for i, distance_of_point in enumerate(vector_of_points)
        ]
    
    return gaps_to_closest_neighbor_of_each_point
