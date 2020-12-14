import os
import pandas as pd

data_dir = os.path.join(os.getcwd(),'Data')
cn_data_dir = os.path.join(data_dir,'ChineseDataset')
cn_embeddings = os.path.join(data_dir, 'sgns.zhihu.bigram')
en_embeddings = os.path.join(data_dir, 'word2vec-google-news-300.gz')
cn_training_path = os.path.join(cn_data_dir, 'bilingual_wordSim353.txt')
cn_training_path2 = os.path.join(cn_data_dir, 'cn_en_quiz.txt')


cn_df = pd.read_csv(cn_training_path, delimiter='\t', names=['cn1', 'cn2', 'en1', 'en2', 'score'])
cn_training = cn_df[['cn1','en1']]
cn_training2 = cn_df[['cn2','en2']]
cn_training2.dropna(inplace=True)
cn_training2.rename(columns={'cn2':'cn1', 'en2': 'en1'}, inplace=True)
cn_training = cn_training.append(cn_training2)
print(cn_training.size)
cn_training.dropna(inplace=True, axis=1)
cn_training.drop_duplicates(inplace=True)
print(cn_training.size)

def pickone(row):
    row['en1'] = row['en1'].split(',')[0]
    return row

cn_df = pd.read_csv(cn_training_path2, delimiter='\t', names=['cn1', 'en1'])
cn_training2 = cn_df.apply(pickone, axis=1)
cn_training = cn_training.append(cn_training2)
cn_training.dropna(inplace=True, axis=1)
cn_training.drop_duplicates(inplace=True)
print(cn_training.shape)

cn_training.to_csv(os.path.join(data_dir, 'transformation_training.csv'))

