Train:
Input:Ch_vec(Preloaded Chinese word vectors),  En_vec(Preloaded English word vectors),  train_data(Mixed Chinese and English)
1.train_data -> tokens
2.Dict_Ch = [all chinese tokens]
3.Trans_Ch = [all translated chinese tokens]
4.W = transform_w(Trans_Ch,Ch_vec,En_vec)
5.Tokens->vector:
Chinese tokens->Ch_vec(Chinese tokens)*W = English vector
English tokens->En_vec(English tokens) = English vector
6.train LSTM model

Predict:
Input:sentence: ”I 喜欢 NLP”
1.sentence ->tokens[‘I’, ‘喜欢’ , ’ NLP’]
2.Tokens -> vector:
Chinese tokens->Ch_vec(Chinese tokens)*W = English vector
English tokens->En_vec(English tokens) = English vector
3.predict in model