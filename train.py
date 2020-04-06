import numpy as np
import torch
from model import GCNModelVAE
from data_process import data_prepare
from torch.optim import Adam
from loss import loss_function

from metric import mse

if __name__ == '__main__':
    dataset = "davis"
    fold = 0
    data, drugs, proteins, affinity = data_prepare(dataset, fold)
    train_data = data["train"]
    valid_data = data["valid"]
    test_data = data["test"]
    # train data
    # print(len(drugs))
    # print(len(proteins))
    # print(np.array(train_data[0]).shape)
    # print(np.array(train_data[2]).shape)

    adj_train = np.zeros((len(drugs) + len(proteins), len(drugs) + len(proteins)), dtype=np.float32)
    train_rows, train_cols = train_data[0], train_data[1]
    valid_rows, valid_cols = valid_data[0], valid_data[1]
    test_rows, test_cols = test_data[0], test_data[1]
    R = np.zeros(affinity.shape) # drug protein interaction matrix
    R[train_rows, train_cols] = 1

    adj_train[:len(drugs), len(drugs):] = R
    adj_train[len(drugs):, :len(drugs)] = R.T
    adj_train = torch.FloatTensor(adj_train)

    epoch = 10000
    lr = 0.001
    model = GCNModelVAE(128, 32, 64).cuda()
    optim = Adam(model.parameters(), lr=lr)

    for i in range(epoch):
        # train
        model.train()
        optim.zero_grad()
        prediction, mu, logvar = model(torch.Tensor(drugs).cuda(), torch.LongTensor(proteins).cuda(), adj_train.cuda())
        # print("affinity",affinity[train_rows, train_cols].shape)
        # print(torch.FloatTensor(affinity[train_rows, train_cols]).size())
        # print("prediction",prediction[train_rows, train_cols].size())
        # print(prediction[train_rows, train_cols])
        loss = loss_function(torch.FloatTensor(affinity[train_rows, train_cols]).cuda(),
                             prediction[train_rows, train_cols],
                             mu, logvar, len(drugs) + len(proteins))
        # print(float(loss))
        loss.backward()
        optim.step()
        if i % 10 == 0:
            print("epoch:", i, "===========")
            print("training loss:", float(loss))
            model.eval()
            with torch.no_grad():
                mse_val = mse(affinity[valid_rows, valid_cols],
                              prediction[valid_rows, valid_cols].cpu().numpy().flatten())
                print("validation mse:", mse_val)

                mse_test = mse(affinity[test_rows, test_cols],
                               prediction[test_rows, test_cols].cpu().numpy().flatten())
                print("test mse:", mse_test)

    # test
