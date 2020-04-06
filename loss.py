import torch
import torch.nn

mse_loss = torch.nn.MSELoss()


def loss_function(labels, preds, mu, logvar, n_nodes):
    cost = mse_loss(preds, labels)
    # print(preds)
    # print(torch.where(torch.isnan(preds)))
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
    # print("cost",cost)
    # print("mu",mu)
    # print("logvar",logvar)
    # print("n_nodes",n_nodes)
    # print("KLD",KLD)
    return cost + KLD
