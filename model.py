import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = nn.Dropout(dropout)
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = self.dropout(input)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout=0.):
        super(GCNModelVAE, self).__init__()
        self.embedding_dim = input_feat_dim
        self.gc1 = GraphConvolution(self.embedding_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)  # mu,mean
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)  # logvar,std
        self.dc = InnerProductDecoder(dropout, act=F.relu)

        # drug embedding
        self.drug_nn1 = nn.Linear(167, self.embedding_dim)
        self.drug_nn2 = nn.Linear(self.embedding_dim, self.embedding_dim * 2)
        self.drug_nn3 = nn.Linear(self.embedding_dim * 2, self.embedding_dim)

        # protein embedding
        self.embedding_xt = nn.Embedding(26, self.embedding_dim)  # for 26 kinds of residues
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=32, kernel_size=8)  # cnn of protein embedding
        self.fc1_xt = nn.Linear(3872, self.embedding_dim)

        self.dropout = nn.Dropout(dropout)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, drug_x, pro_x, net_adj):
        drug_embedding = []
        pro_embedding = []
        # all drug noedes
        for i in range(drug_x.size()[0]):
            drug_temp_embedding = F.relu(self.drug_nn1(drug_x[i]))
            drug_temp_embedding = self.dropout(drug_temp_embedding, )
            drug_temp_embedding = F.relu(self.drug_nn2(drug_temp_embedding))
            drug_temp_embedding = self.dropout(drug_temp_embedding)
            drug_temp_embedding = F.relu(self.drug_nn3(drug_temp_embedding))
            drug_embedding.append(drug_temp_embedding)
        drug_embedding = torch.stack(drug_embedding)
        # all protein nodes
        for i in range(pro_x.size()[0]):
            pro_temp_embedding = self.embedding_xt(pro_x[i])
            pro_temp_embedding = self.conv_xt_1(pro_temp_embedding)
            # flatten
            pro_temp_embedding = pro_temp_embedding.view(-1)
            pro_temp_embedding = self.fc1_xt(pro_temp_embedding)
            pro_embedding.append(pro_temp_embedding)
        pro_embedding = torch.stack(pro_embedding)

        # concatenate the drug and protein to construct a whole graph
        x = torch.cat((drug_embedding, pro_embedding), dim=0)
        mu, logvar = self.encode(x, net_adj)  # mean and std of the enconder
        z = self.reparameterize(mu, logvar)  # latent embedding
        return self.dc(z), mu, logvar


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=F.relu):
        super(InnerProductDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.act = act

    def forward(self, z):
        z = self.dropout(z)
        # adj = self.act(torch.mm(z, z.t()))
        adj = torch.mm(z, z.t())
        return adj
