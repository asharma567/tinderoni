import pandas as pd
import os
import os.path
import shutil


df = pd.read_csv('tinder_pics_likes_faces_deduped.csv', header=None)
df['new_col'] = df[0].apply(lambda x: x.split('_')[0][:len('1526002520') - 2])
df['newer_col'] = df['new_col'].apply(lambda x: x + '_') +  df[0].apply(lambda x: x.split('_',1)[1])

lis = list(df['newer_col'])

n_lis=[]
hashes=[]
for idx, tup in enumerate(lis):
    curr_t_stamp, curr_name, _, _ = tup.split('_')
    hash_ = curr_t_stamp + curr_name
    
    if hash_ not in set(hashes):
        n_lis.append(df[0].iloc[idx])
        hashes.append(hash_)
    
    

for file_name in n_lis:
    
    srcpath = os.path.join('tinder_pics_likes_faces_deduped', file_name)
    dstpath = os.path.join('tinder_pics_likes_faces_deduped_firsts', file_name)

    shutil.copyfile(srcpath, dstpath)