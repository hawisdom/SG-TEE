import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class TokenAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, args, concat=True):
        super(TokenAttentionLayer, self).__init__()
        self.dropout = args.dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = args.alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features))) # (D, D')
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h,adj,level):
        # h     (N,T,D)
        # adj   (N*T, N*T)
        # level (N,T)
        Wh = torch.matmul(h, self.W)  # (N,T,D) * (D, D') = (N,T, D')
        a_input = self._prepare_attentional_mechanism_input(Wh.view(-1,Wh.shape[2]))  # (N*T,N*T,2D')
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # (N*T, N*T)
        level = level.view(level.shape[0]*level.shape[1]).unsqueeze(1)  # (N*T,1)
        Wh = Wh.view(Wh.shape[0]*Wh.shape[1],Wh.shape[2])
        Wh = torch.mul(Wh, level)  # (N*T,D')
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)   # (N*T, D')
        h_prime = h_prime.view(h.shape[0],h.shape[1],h_prime.shape[1])  # (N,T,D')

        if self.concat:
            return self.leakyrelu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        with torch.no_grad():
            N = Wh.size()[0]  # number of nodes

            # Below, two matrices are created that contain embeddings in their rows in different orders.
            # (e stands for embedding)
            # These are the rows of the first matrix (Wh_repeated_in_chunks):
            # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
            # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
            #
            # These are the rows of the second matrix (Wh_repeated_alternating):
            # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
            # '----------------------------------------------------' -> N times
            #
            Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
            Wh_repeated_alternating = Wh.repeat(N, 1)

            # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
            # e1 || e1
            # e1 || e2
            # e1 || e3
            # ...
            # e1 || eN
            # e2 || e1
            # e2 || e2
            # e2 || e3
            # ...
            # e2 || eN
            # ...
            # eN || e1
            # eN || e2
            # eN || e3
            # ...
            # eN || eN
            all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
            return all_combinations_matrix.view(N, N, 2 * self.out_features)

class TokenDepAttentionLayer(nn.Module):
    def __init__(self, in_dim = 300,out_dim = 64,concat=True):
        super(TokenDepAttentionLayer, self).__init__()
        self.fc1 = nn.Linear(in_dim,out_dim)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(out_dim,1)
        self.a = nn.Parameter(torch.empty(size=(1, out_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.concat = concat

    def forward(self,feature,dep_tags_v):
        '''
        feature           [N,T,D]
        Q dep_tags_v      [N,T Q]
        '''
        Q = self.fc1(dep_tags_v)  # (N,T,D')
        Q = self.relu(Q)
        Q = self.fc2(Q)  # (N, T, 1)

        out = torch.bmm(feature.transpose(1, 2), Q)  # (N, D, 1)
        out = torch.matmul(out, self.a)  # (N, D, D')
        out = torch.bmm(feature, out)  # (N,T,D')

        if (self.concat):
            return self.relu(out)
        else:
            return out

class TokenPOSAttentionLayer(nn.Module):
    def __init__(self, in_dim = 300,out_dim = 64,concat=True):
        super(TokenPOSAttentionLayer, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(out_dim, 1)
        self.a = nn.Parameter(torch.empty(size=(1, out_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.concat = concat

    def forward(self,feature,pos_tags_v):
        '''
        feature           [L,T,D]
        P dep_tags_v      [L,T Q]
        '''
        P = self.fc1(pos_tags_v)
        P = self.relu(P)
        P = self.fc2(P)  # (N, T, 1)

        out = torch.bmm(feature.transpose(1, 2), P)  # (N, D, 1)
        out = torch.matmul(out, self.a)  # (N, D, D')
        out = torch.bmm(feature, out)  # (N,T,D')

        if (self.concat):
            return self.relu(out)
        else:
            return out


class EventAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, args, concat=True):
        # in_dim   33D
        # out_dim  D'
        super(EventAttentionLayer, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.in_features = in_dim
        self.out_features = out_dim
        self.alpha = args.alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_dim, out_dim))) # (D, D')
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_dim,1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, level):
        # h     (N, 33D)
        # adj   (N, N)
        # level (N, 1)
        Wh = torch.matmul(h, self.W)  # (N, 33D) * (33D, D') = (N, D')
        a_input = self._prepare_attentional_mechanism_input(Wh) # (N, N, 2D')
        e = self.leakyrelu(torch.matmul(a_input,self.a).squeeze(2))  # (N, N)
        Wh = torch.mul(Wh, level.unsqueeze(1))  # (N, D')
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)  # (N, D')

        if self.concat:
            return self.leakyrelu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        with torch.no_grad():
            N = Wh.size()[0]  # number of nodes

            # Below, two matrices are created that contain embeddings in their rows in different orders.
            # (e stands for embedding)
            # These are the rows of the first matrix (Wh_repeated_in_chunks):
            # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
            # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
            #
            # These are the rows of the second matrix (Wh_repeated_alternating):
            # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
            # '----------------------------------------------------' -> N times
            #
            Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
            Wh_repeated_alternating = Wh.repeat(N, 1)

            # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
            # e1 || e1
            # e1 || e2
            # e1 || e3
            # ...
            # e1 || eN
            # e2 || e1
            # e2 || e2
            # e2 || e3
            # ...
            # e2 || eN
            # ...
            # eN || e1
            # eN || e2
            # eN || e3
            # ...
            # eN || eN
            all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
            # gc.collect()
            return all_combinations_matrix.view(N, N, 2 * self.out_features)

class EventDepAttentionLayer(nn.Module):
    def __init__(self, in_dim,out_dim,hidden_dim,concat=True):
        super(EventDepAttentionLayer, self).__init__()
        self.fc1 = nn.Linear(in_dim,out_dim)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(out_dim,1)
        self.a = nn.Parameter(torch.empty(size=(hidden_dim, out_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.concat = concat

    def forward(self,feature,dep_tags_v):
        '''
        feature           [N,T,D]
        Q dep_tags_v      [N,T,Q]
        '''
        Q = self.fc1(dep_tags_v)
        Q = self.relu(Q)   # (N,D')
        Q = self.fc2(Q)  # (N,T,1)

        out = torch.bmm(feature.transpose(1, 2), Q)
        out = out.squeeze(2)  # (N,D)
        out = torch.matmul(out, self.a)  # (N, D')

        if (self.concat):
            return self.relu(out)
        else:
            return out

class EventPOSAttentionLayer(nn.Module):
    def __init__(self, in_dim,out_dim,hidden_dim,concat=True):
        super(EventPOSAttentionLayer, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.relu = nn.LeakyReLU()
        self.fc2 = nn.Linear(out_dim, 1)
        self.a = nn.Parameter(torch.empty(size=(hidden_dim, out_dim)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.concat = concat

    def forward(self,feature,pos_tags_v):
        '''
        feature           [L,T,D]
        P dep_tags_v      [L,T Q]
        '''
        P = self.fc1(pos_tags_v)
        P = self.relu(P)
        P = self.fc2(P)  # (N, L, 1)

        out = torch.bmm(feature.transpose(1, 2), P)
        out = out.squeeze(2)  # (N,D)
        out = torch.matmul(out, self.a)  # (N, D')

        if (self.concat):
            return self.relu(out)
        else:
            return out
