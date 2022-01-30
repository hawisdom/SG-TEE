import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import TokenAttentionLayer,TokenDepAttentionLayer,TokenPOSAttentionLayer,EventAttentionLayer,EventDepAttentionLayer,EventPOSAttentionLayer
torch.set_printoptions(profile="full")

class TEE_GAT_Event(nn.Module):
    def __init__(self, args,dep_tag_num,pos_tag_num,sen_order_num):
        super(TEE_GAT_Event, self).__init__()
        self.args = args

        num_embeddings, embed_dim = args.word2vec_embedding.shape
        self.embed = nn.Embedding(num_embeddings, embed_dim)
        self.embed.weight = nn.Parameter(args.word2vec_embedding, requires_grad=False)

        self.dropout = nn.Dropout(args.dropout)

        self.bilstm = nn.LSTM(input_size=args.embedding_dim, hidden_size=args.hidden_size,
                              bidirectional= True, batch_first=True, num_layers=args.num_layers_bilstm)

        self.word_gats = []
        self.dep_event_gats = []
        self.share_event_gats = []
        self.org_event_gats = []

        # token gat
        token_in_dim = 2 * args.hidden_size
        for k in range(args.num_token_gat_layers - 1):
            self.word_gats.append([TokenAttentionLayer(token_in_dim,token_in_dim,args,concat=True).to(args.device) for _ in range(args.num_heads)])
            # token_in_dim = token_out_dim
        self.word_gat_out = TokenAttentionLayer(token_in_dim, token_in_dim, args, concat=False).to(args.device)

        self.word_dep_gat = [TokenDepAttentionLayer(args.dep_embedding_dim, args.dep_embedding_dim, concat=False).to(args.device)
                        for i in range(args.num_heads)]
        self.word_pos_gat = [TokenPOSAttentionLayer(args.dep_embedding_dim, args.dep_embedding_dim, concat=False).to(args.device)
                        for i in range(args.num_heads)]

        event_in_dim = 3 * 2 * args.hidden_size + args.sen_order_embedding_dim
        for k in range(args.num_dep_event_gat_layers - 1):
            self.dep_event_gats.append([EventAttentionLayer(event_in_dim, event_in_dim, args,concat=True).to(args.device) for _ in range(args.num_heads)])
        for k in range(args.num_share_event_gat_layers - 1):
            self.share_event_gats.append([EventAttentionLayer(event_in_dim, event_in_dim, args,concat=True).to(args.device) for _ in range(args.num_heads)])

        for k in range(args.num_org_event_gat_layers - 1):
            self.org_event_gats.append(
                [EventAttentionLayer(event_in_dim, event_in_dim, args, concat=True).to(args.device) for _ in range(args.num_heads)])

        self.dep_event_gat_out = EventAttentionLayer(event_in_dim, event_in_dim, args,concat=False).to(args.device)
        self.share_event_gat_out = EventAttentionLayer(event_in_dim, event_in_dim, args,concat=False).to(args.device)
        self.org_event_gat_out = EventAttentionLayer(event_in_dim, event_in_dim, args, concat=False).to(
                args.device)
        self.event_dep_gat = [EventDepAttentionLayer(args.dep_embedding_dim, args.dep_embedding_dim,2*args.hidden_size,concat=False).to(args.device) for i in range(args.num_heads)]
        self.event_pos_gat = [EventPOSAttentionLayer(args.dep_embedding_dim, args.dep_embedding_dim,2*args.hidden_size,concat=False).to(args.device) for i in range(args.num_heads)]

        self.dep_embed = nn.Embedding(dep_tag_num, args.dep_embedding_dim)
        self.pos_embed = nn.Embedding(pos_tag_num, args.pos_embedding_dim)
        self.sen_order_embed = nn.Embedding(sen_order_num,args.sen_order_embedding_dim)

        # (N, 3*2D'+ 2*3*D''+3*3*2D'+3*D''+2D'')
        event_in_dim = 24 * args.hidden_size  + 11 * args.dep_embedding_dim
        last_hidden_size = event_in_dim

        layers = [nn.Linear(last_hidden_size, args.final_hidden_size), nn.LeakyReLU()]
        for _ in range(args.num_mlps - 1):
            layers += [nn.Linear(args.final_hidden_size,
                                 args.final_hidden_size), nn.LeakyReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(args.final_hidden_size, args.num_classes)


    def forward(self, token_level,token_adj,event_ids,event_dep_ids,event_pos_ids,
                event_sen_ids, event_level,dep_e_adj,share_e_adj,org_e_adj):
        # N     numbers of events
        # T     numbers of event node
        # D     dim of tokens/dep
        # D'    out dim of unidirectional bilstm
        # D''   out dim of dep/pos

        event_feature = self.embed(event_ids)   # (N,T,D)
        event_feature = self.dropout(event_feature)
        event_out_bilstm, _ = self.bilstm(event_feature)  # (N,T,2D')

        # token aggregation information
        # (N,T,2D')
        token_in_feature = event_out_bilstm
        for i in range(len(self.word_gats)):
            token_out = [g(token_in_feature, token_adj,token_level).unsqueeze(1) for g in self.word_gats[i]]
            token_out = torch.cat(token_out, dim = 1)
            token_out = token_out.mean(dim=1)    # (N,T,2D')
            token_out = self.dropout(token_out)
            token_in_feature = token_out
        token_out = F.relu(self.word_gat_out(token_in_feature,token_adj,token_level))  # (N,T,2D')
        token_out = token_out.view(token_out.shape[0],-1) # (N,T*2D')

        token_dep_feature = self.dep_embed(event_dep_ids)  # (N,T,Q)
        token_dep_out = [g(event_out_bilstm, token_dep_feature).unsqueeze(1) for g in self.word_dep_gat]
        token_dep_out = torch.cat(token_dep_out, dim=1)
        token_dep_out = token_dep_out.mean(dim=1)  # (N,T,D'')
        token_dep_out = F.relu(token_dep_out)
        token_dep_out = token_dep_out.view(token_dep_out.shape[0],-1)

        token_pos_feature = self.pos_embed(event_pos_ids)  # (N,T,P)
        token_pos_out = [g(event_out_bilstm, token_pos_feature).unsqueeze(1) for g in self.word_pos_gat]
        token_pos_out = torch.cat(token_pos_out, dim=1)
        token_pos_out = token_pos_out.mean(dim=1)  # (N,T,D'')
        token_pos_out = F.relu(token_pos_out)
        token_pos_out = token_pos_out.view(token_pos_out.shape[0],-1)


        # event aggregation information
        event_dep_feature = self.dep_embed(event_dep_ids)  # (N,T,Q)
        event_dep_feature = self.dropout(event_dep_feature)

        event_dep_out = [g(event_out_bilstm, event_dep_feature).unsqueeze(1) for g in self.event_dep_gat]
        event_dep_out = torch.cat(event_dep_out, dim=1)
        event_dep_out = event_dep_out.mean(dim=1)  # (N,D'')
        event_dep_out = F.relu(event_dep_out)

        event_pos_feature = self.pos_embed(event_pos_ids)  # (N,T,P)
        event_pos_feature = self.dropout(event_pos_feature)

        event_pos_out = [g(event_out_bilstm, event_pos_feature).unsqueeze(1) for g in self.event_pos_gat]
        event_pos_out = torch.cat(event_pos_out, dim=1)
        event_pos_out = event_pos_out.mean(dim=1)  # (N,D'')
        event_pos_out = F.relu(event_pos_out)


        sen_order_feature = self.sen_order_embed(event_sen_ids)  # (N,D'')
        sen_order_feature = self.dropout(sen_order_feature)

        event_out_bilstm = event_out_bilstm.reshape(event_out_bilstm.shape[0], -1)  # (N,3*2D')
        event_out_bilstm = torch.cat([event_out_bilstm,sen_order_feature],dim=1) # (N,3*2D'+D'')
        event_in_feature = event_out_bilstm
        
        for i in range(len(self.dep_event_gats)):
            dep_event_out = [g(event_in_feature, dep_e_adj,event_level).unsqueeze(1) for g in self.dep_event_gats[i]]
            dep_event_out = torch.cat(dep_event_out, dim = 1)
            dep_event_out = dep_event_out.mean(dim=1)    # (N, 3*2D')
            dep_event_out = self.dropout(dep_event_out)
            event_in_feature = dep_event_out
        dep_event_out = F.relu(self.dep_event_gat_out(event_in_feature,dep_e_adj,event_level))  # (N, 3*2D')

        for i in range(len(self.share_event_gats)):
            share_event_out = [g(event_in_feature, share_e_adj,event_level).unsqueeze(1) for g in self.share_event_gats[i]]
            share_event_out = torch.cat(share_event_out, dim = 1)
            share_event_out = share_event_out.mean(dim=1)    # (N,3*2D')
            share_event_out = self.dropout(share_event_out)
            event_in_feature = share_event_out
        share_event_out = F.relu(self.share_event_gat_out(event_in_feature,share_e_adj,event_level))  # (N,3*2D')

        for i in range(len(self.org_event_gats)):
            org_event_out = [g(event_in_feature, org_e_adj,event_level).unsqueeze(1) for g in self.org_event_gats[i]]
            org_event_out = torch.cat(org_event_out, dim=1)
            org_event_out = org_event_out.mean(dim=1)    # (N,3*2D')
            org_event_out = self.dropout(org_event_out)
            event_in_feature = org_event_out
        org_event_out = F.relu(self.org_event_gat_out(event_in_feature,org_e_adj,event_level))  # (N,3*2D')

        # (N, 3*2D'+ 2*3*D''+2*3*2D'+2*D''+2D'')
        event_all_feature_out = torch.cat([token_out,token_dep_out,token_pos_out,dep_event_out,share_event_out,org_event_out,event_dep_out,event_pos_out],dim=1)
        event_all_feature_out = self.dropout(event_all_feature_out)

        out = self.fcs(event_all_feature_out)
        logit = self.fc_final(out)   # (N, 2)
        return logit





