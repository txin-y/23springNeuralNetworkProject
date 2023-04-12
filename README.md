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
