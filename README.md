
# FAKE NEWS DETECTOR 

Running the code : We can run the main.py file. You just need to input whether you need the prediction to be “binary” or “multiclass”.

About the model: I have used the statement as well as the justification together as text input to train the model. I merged the statement and justification as a single text. First we use pre-trained 100 dimension Glove embeddings. This constitutes the first layer of the sequential model. Then we have used a 1D Convnet with relu activation, and filters =32. We added another layer of 1D convent with the same activation and filters = 64. Then we used Bidirectional LTSM with two layers(128, 64) and followed by a softmax layer, which has 2 or 6 neurons based on our input as “binary” or “multiclass”. 

The accuracy can be improved by the following few measures: 
1. Using the metadata related to previous counts or true/fake statements of each speaker in both binary and multiclass. 
2. Using parallel BiLSTM as mentioned in paper might improve the results to a good extent. 
3. Using both training data and validation data to train our final model will improve accuracy to a minimal extent as well. 

## Accuracy : 
1. Multiclass : 0.832 
2. Binary : 0.523
