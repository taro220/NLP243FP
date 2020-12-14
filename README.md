# NLP243FP
To download w2v weights, use 

import gensim.downloader as api \
path = api.load("word2vec-google-news-300", return_path=True)

Place the weights into the weights directory or adjust
the path in the code

You will need the model weights for the english trained biLSTM.
It can be downloaded here. 
Place it into the weights directory.


weights for english bilstm - weight directory

https://drive.google.com/file/d/1Hq8Wf7Zm2Gtd8OSI4Y7wzcWFAsmtpT-x/view?usp=sharing

weights for chinese word embeddings - weight directory

https://drive.google.com/file/d/1mmCxVJmNn1anONrPS1fEKNSdmIgNt5K_/view?usp=sharing

training data for translation matrix - data directory

https://drive.google.com/file/d/1kNQy5nK0fx4Q2XKPFqSW9IoJLGIjj0Nz/view?usp=sharing

https://drive.google.com/file/d/1xYiEFReVamLyepzdFDc6IiSOqn5aJ-xq/view?usp=sharing

https://drive.google.com/file/d/1NyBxGnVw3fPv6M84yMIzpoEg9bhP_K7Z/view?usp=sharing

Each model can be ran independently. 

bilstm.py is the initial english trained sentiment model.

tranformation_matrix_model.py is for training the translation matrix.

bilingual.py is the final model combining the english bilstm and the translation matrix.
