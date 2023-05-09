# 23springNeuralNetworkProject - Xiaoting Yu

# Report (First Solution)
Implement sentiment analysis based on IMDB dataset using textclassification module and transformers 

Given the issues with transformer to be solved the Neural Network Architecture in this solution consists of the following parts: an embedding layer, an average pooling layer, and a linear layer. The embedding layer maps each word in the vocabulary to a low-dimensional vector, and the average pooling layer computes the average of the embeddings for each review. Finally, the linear layer maps the pooled embeddings to the output labels. This architecture is a commonly used approach for text classification tasks and has shown to be effective.

The loss function used is cross-entropy loss, which is suitable for multi-class classification problems. The optimization algorithm used is stochastic gradient descent (SGD) with a learning rate of 5. The learning rate is reduced by a factor of 10 after every epoch if the validation accuracy does not improve, using the StepLR scheduler. These hyperparameters were chosen based on experimentation and previous work in the field.

Classification Accuracy:
The code reports the validation accuracy of the model after each epoch. The reported validation accuracy for each epoch is as follows:

Epoch 1: 73.4%
Epoch 2: 80.3%
Epoch 3: 72.5%
Epoch 4: 84.2%
Epoch 5: 84.1%
Epoch 6: 84.6%
Epoch 7: 84.6%
Epoch 8: 84.7%
Epoch 9: 84.7%
Epoch 10: 84.7%
The reported test accuracy is 84.5%.

The classification accuracy achieved by the model is quite good, considering the simplicity of the architecture and the relatively small size of the dataset. However, it is important to note that the accuracy on the training set may be significantly higher than the validation and test accuracies, indicating overfitting. This can be seen in the reported validation accuracies, which plateau at around 84-85% after the fourth epoch, while the training accuracy may continue to increase.

Ideas for Improvements:
To improve the generalization capabilities of the model, several strategies can be used. One possible strategy is to increase the size of the dataset by using data augmentation techniques such as adding noise, synonyms, or perturbations to the text data. Another strategy is to use a more complex neural network architecture, such as a LSTM or transformer (can be seen in v2), which can capture more complex patterns in the text data. Additionally, regularization techniques such as dropout or weight decay can be applied to prevent overfitting. Finally, the hyperparameters of the model can be tuned using a grid search or a random search to find the optimal combination of learning rate, batch size, and architecture parameters.

code is based on https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

# Report (Final Solution)

This report provides an overview of four different versions of code for sentiment analysis. The first three versions utilize the IMDB dataset, while the fourth version is based on the Tweet text dataset from Kaggle. 

## Version 1 - Simple Networks

This version uses a simple neural network to classify the reviews into positive and negative sentiments.

## Version 2 - Transformer from Scratch

The second version replaces the simple networks with a transformer built from scratch. However, this version had difficulty learning, and after several attempts, the network still could not learn effectively.

### Note: In Version 2, the transformer model was challenging to train, and it was difficult to achieve good results. However, the code still demonstrates the various attempts made to improve its performance. It's essential to learn from this experience and consider using a pre-trained model or alternative networks like LSTM for better results.

## Version 3 - LSTM Network

The third version employs an LSTM network for classification, resulting in improved accuracy. Additionally, visualizations such as loss and accuracy curves are incorporated to enhance the analysis. A new evaluation metric, ROC and AUC, is introduced, with the corresponding curve plotted and the area calculated at the end of the training process. These elements are handled separately in three distinct kernels. To avoid confusion, the simple version has been commented out.

## Version 4 - Tweet Text Dataset

To run this code successfully on Colab, we need to upload train.csv to path /content/train.csv.

This version uses the Tweet text dataset from Kaggle to classify racist or sexist tweets from other tweets. The dataset contains about 32k tweets and is unbalanced, with 29,720 non-hatred and 2,242 hatred tweets.

An advanced version of tweet text dataset added more illustrations and visualizations on the dataset, and fix the metric problem. Due to the fact that the dataset is unbalanced, the hatred speech only accounts for 5.6% of the total dataset. So the accuracy cannot prove the effectiveness of the network. So again, ROC curve is introduced to further illustrate the results after each epoch. After 20 epochs, we save the 20th weights and test the unseen data with this weights which turned out to be similar to the training data.

## Experimenting with Learning Rate, Optimizer, and Hyperparameters （Version 2）
### Learning Rate
LR 0.1

Epoch = 10
```
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
model = Net(
    vocab_size=len(vocab),
    d_model=200,
    nhead=2,  # the number of heads in the multiheadattention models
    dim_feedforward=200,  # the dimension of the feedforward network model in nn.TransformerEncoder
    num_layers=2,
    dropout=0.2,
    classifier_dropout=0.0,
).to(device)
```
Accuracy: 0.507

LR 0.01

Epoch = 10

```
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
model = Net(
    vocab_size=len(vocab),
    d_model=200,
    nhead=2,  # the number of heads in the multiheadattention models
    dim_feedforward=200,  # the dimension of the feedforward network model in nn.TransformerEncoder
    num_layers=2,
    dropout=0.2,
    classifier_dropout=0.0,
).to(device)
```
Accuracy: 0.506

### Optimizer
Change From SGD to Adam

```
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
```
Accuracy: 0.498

LR: 0.01

### Hyperparameters
```
model = Net(
    vocab_size=len(vocab),
    d_model=200,
    nhead=2,  # the number of heads in the multiheadattention models
    dim_feedforward=2048,  # the dimension of the feedforward network model in nn.TransformerEncoder
    num_layers=2,
    dropout=0.2,
    classifier_dropout=0.0,
).to(device)
```
LR = 0.1, EPOCH = 10, BATCH_SIZE = 50

```
predicted_label = model(text, offsets)
predicted_label = torch.tensor(predicted_label.argmax(1), dtype=torch.float32)
loss = criterion(predicted_label, label)
loss.requires_grad = True
loss.backward()
```
Accuracy: 0.482

Change LR to 5: Accuracy = 0.517~0.520
Change EPOCH to 30: Accuracy = 0.494
Change LR to 1: Accuracy = 0.483
```
dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
```
Accuracy = 0.518

Change EPOCH to 100: Accuracy = 0.489
```
import math

model = Net(
    vocab_size=len(vocab),
    d_model=200,
    nhead=2,  # the number of heads in the multiheadattention models
    dim_feedforward=2048,  # the dimension of the feedforward network model in nn.TransformerEncoder
    num_layers=2,
    dropout=0.1,
    classifier_dropout=0.,
).to(device)
```
Despite the various experiments, it is still challenging to train a transformer from scratch. Using a pre-trained model could be a more efficient choice, but for learning purposes, choosing another network like LSTM might be a better choice.

## About Dataset -- Twitter Sentiment Analysis

### Context

The objective of this task is to detect hate speech in tweets. For the sake of simplicity, we say a tweet contains hate speech if it has a racist or sexist sentiment associated with it. So, the task is to classify racist or sexist tweets from other tweets.

Formally, given a training sample of tweets and labels, where label '1' denotes the tweet is racist/sexist and label '0' denotes the tweet is not racist/sexist, your objective is to predict the labels on the test dataset.

### Content

Full tweet texts are provided with their labels for training data.
Mentioned users' username is replaced with [@user](https://www.kaggle.com/user).

For train.csv they have about 32K tweets and consists of three columns: id, label, and texts. Since it's a hatred detection dataset, it's unbalanced including 29,720 non-hatred and 2,242 hatred.

### Acknowledgements

Dataset is provided by [Analytics Vidhya](http://https//datahack.analyticsvidhya.com/contest/practice-problem-twitter-sentiment-analysis/)

link : https://www.kaggle.com/datasets/arkhoshghalb/twitter-sentiment-analysis-hatred-speech?select=train.csv
