# 23springNeuralNetworkProject - Xiaoting Yu

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

# Brief Introduction and Instructions on How to Run the Code

This report provides an overview of four different versions of code for sentiment analysis. The first three versions utilize the IMDB dataset, while the fourth version is based on the Tweet text dataset from Kaggle. 

## Version 1 - Simple Networks

This version uses a simple neural network to classify the reviews into positive and negative sentiments.

## Version 2 - Transformer from Scratch

The second version replaces the simple networks with a transformer built from scratch. However, this version had difficulty learning, and after several attempts, the network still could not learn effectively.

## Version 3 - LSTM Network

The third version implements an LSTM network for classification and achieves higher accuracy. This version also includes visualizations by drawing loss and accuracy curves, introduces a new measurement metric (ROC and AUC), and calculates the area at the end of the entire training.

## Version 4 - Tweet Text Dataset

This version uses the Tweet text dataset from Kaggle to classify racist or sexist tweets from other tweets. The dataset contains about 32k tweets and is unbalanced, with 29,720 non-hatred and 2,242 hatred tweets.

# How to Run the Code

1. Choose the version of the code you'd like to run (either Version 1, 2, 3, or 4).
2. Ensure you have the necessary libraries and dependencies installed, such as PyTorch.
3. Load the corresponding dataset (IMDB for Versions 1-3 or the Tweet text dataset for Version 4).
4. Configure the hyperparameters according to the version you've chosen.
5. Run the code to train the model and evaluate its performance.

Note: In Version 2, the transformer model was challenging to train, and it was difficult to achieve good results. However, the code still demonstrates the various attempts made to improve its performance. It's essential to learn from this experience and consider using a pre-trained model or alternative networks like LSTM for better results.

# Experimenting with Learning Rate, Optimizer, and Hyperparameters
## Learning Rate
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

## Optimizer
Change From SGD to Adam

```
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
```
Accuracy: 0.498

LR: 0.01

## Hyperparameters
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
Despite the various experiments, it is still challenging to train a transformer from scratch. Using a pre-trained model could be a more efficient choice, but for learning purposes, choosing another network like LSTM might be a better choice
