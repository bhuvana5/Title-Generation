import  numpy as np
from keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding, GRU
from keras.models import Sequential

import matplotlib.pylab as plt

#weights to be given to the model
def CreateSampleWeights(y,maxlen):
    sw = np.zeros((len(y), maxlen))
    for i in range(len(y)):
        for t, char in enumerate(y[i]):
            if(y[i,t,char]==1):
                sw[i,t]=1
    return sw

#defining architecture (layers) of LSTM
def buildmodel(words,embsize, hiddensize ,maxlen,name='headlines_gen',loadweights=False):
    model = Sequential()
    model.add(Embedding(words, embsize, input_length=maxlen,mask_zero=True))
    model.add(LSTM(hiddensize, return_sequences=True)) 
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(words)))
    model.add(Activation('softmax'))
    if(loadweights):
        model.load_weights(name)
    model.compile( sample_weight_mode="temporal",loss='categorical_crossentropy', optimizer='rmsprop')
    return model
#fitting the model 
def train(model,x_train,y_train,epochs,maxlen, name, batch_size=128):
    model.fit(x_train, y_train, batch_size=batch_size,
         sample_weight=CreateSampleWeights(y_train,maxlen),
     nb_epoch=epochs,validation_split=0.33)
    model.save_weights(name,overwrite=True)
    
def sample(a, temperature=1.0):
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))
#predicting titles
def Generatetitles(sample_no,x_test,model,maxlen,indices_words):
    for j in range(100):
        input_pre = np.zeros((1,maxlen))
        input_pre[0]=x_test[j]
        pred= model.predict(input_pre,batch_size=1)
        headline = list(input_pre[0])
        title=''
        for word in range(len(headline)):
            if(headline[word]!=0.0):
                title= title +' '+indices_words[headline[word]]
        
        length=np.random.randint(1,10)
        for word in range(length):
            chart= list(pred[0,word])
            next_index = chart.index(max(chart)) 
            title= headlinetext +' '+indices_words[next_index]
        print(title+'.')
    print('\n')
