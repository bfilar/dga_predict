import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sklearn
from sklearn import feature_extraction
from sklearn.model_selection import train_test_split

import dga_classifier.data as data
from IPython import embed

class BigramModel(nn.Module):
    def __init__(self, num_feats):
        super(BigramModel, self).__init__()
        self.linear = nn.Linear(num_feats, 1)
    
    def forward(self, bigram_vec):
        y_pred = torch.sigmoid(self.linear(bigram_vec))
        return y_pred.squeeze()

def run():
    print("fetching data...")
    indata = data.get_data()
    # Extract data and labels
    X, labels = zip(*indata)
    y = np.asarray([0 if x == 'benign' else 1 for x in labels])

    # Create feature vectors
    print("vectorizing data...")
    ngram_vectorizer = feature_extraction.text.CountVectorizer(analyzer='char', ngram_range=(2, 2))
    count_vec = ngram_vectorizer.fit_transform(X)
    max_features = count_vec.shape[1]

    X_data = torch.autograd.Variable(torch.from_numpy(count_vec.todense())).float()
    y_data = torch.tensor(y).float()
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.10)

    lr = 0.001
    epochs = 2

    model = BigramModel(max_features)
    criterion = nn.BCELoss(reduction='mean')
    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    trainset = TensorDataset(X_train, y_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)

    testset = TensorDataset(X_test, y_test)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, _data in enumerate(trainloader):
            inputs, labels = _data
            optimizer.zero_grad()

            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:
                print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0

    # test
    with torch.no_grad():
        predicted = model(X_test)
        auc = sklearn.metrics.roc_auc_score(y_test, predicted)
        print('Final AUC: {}'.format(auc))