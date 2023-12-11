from transformers import BertPreTrainedModel, BertModel
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class AutoEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dims):
        super(AutoEncoder, self).__init__()
        self.encoder_layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            if i < len(dims) - 2:
                layer = nn.Sequential(nn.Linear(dims[i], dims[i + 1]), nn.ReLU())
            else:
                layer = nn.Linear(dims[i], dims[i+1])
            self.encoder_layers.append(layer)
        self.encoder = nn.Sequential(*self.encoder_layers)

        self.decoder_layers = []
        hidden_dims.reverse()
        dims = hidden_dims + [input_dim]
        for i in range(len(dims) - 1):
            if i < len(dims) - 2:
                layer = nn.Sequential(nn.Linear(dims[i], dims[i+1]), nn.ReLU())
            else:
                layer = nn.Linear(dims[i], dims[i+1])
            self.decoder_layers.append(layer)
        self.decoder = nn.Sequential(*self.decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        z = F.normalize(z, dim=-1)
        x_bar = self.decoder(z)
        return x_bar, z

    def decode(self, z):
        z = F.normalize(z, dim=-1)
        return self.decoder(z)


class TopClusModel(BertPreTrainedModel):

    def __init__(self, config, input_dim, hidden_dims, n_clusters, kappa):
        super().__init__(config)
        self.init_weights()
        self.topic_emb = Parameter(torch.Tensor(n_clusters, hidden_dims[-1]))
        self.sub_topic_emb = Parameter(torch.Tensor(n_clusters, n_clusters, hidden_dims[-1]))
        self.bert = BertModel(config, add_pooling_layer=False)
        self.ae = AutoEncoder(input_dim, hidden_dims)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.kappa = kappa
        self.v = Parameter(torch.rand(config.hidden_size))
        torch.nn.init.xavier_normal_(self.topic_emb.data)
        for s in self.sub_topic_emb:
            torch.nn.init.xavier_normal_(s.data) 
        self.init_weights()
        for param in self.bert.parameters():
            param.requires_grad = False

    def cluster_assign(self, z, detach=False, softmax=True):
        self.topic_emb.data = F.normalize(self.topic_emb.data, dim=-1)
        if detach:
            sim = torch.matmul(z, self.topic_emb.detach().t()) * self.kappa
        else:
            sim = torch.matmul(z, self.topic_emb.t()) * self.kappa
        if not softmax:
            return sim
        p = F.softmax(sim, dim=-1)
        return p

    # # TODO: sub cluster assign
    # def subcluster_assign(self, z, p):
    #     self.sub_topic_emb.data = F.normalize(self.sub_topic_emb.data, dim=-1)
    #     # print('sub topic emb shape: ', self.sub_topic_emb.shape)
    #     # chosen_sub_topic_emb = self.sub_topic_emb[topic_nums]
    #     sub_sim = torch.matmul(self.sub_topic_emb, z.t()).transpose(0, 2)
    #     # print('sub sim shape: ', sub_sim.shape)
    #     p = p.unsqueeze(2)
    #     # print('p shape: ', p.shape)
    #     sim = (torch.matmul(sub_sim, p)*self.kappa).squeeze(2)
    #     # print('sim shape: ', sim.shape)

    #     p = F.softmax(sim, dim=-1)
    #     return p
    
    def subcluster_assign(self, z):
        self.sub_topic_emb.data = F.normalize(self.sub_topic_emb.data, dim=-1)
        sub_topic_emb = self.sub_topic_emb.view(-1, self.sub_topic_emb.shape[-1])
        # print("sub topic emb shape:", sub_topic_emb.shape)

        # sim = torch.matmul(z, sub_topic_emb.t()) * self.kappa
        sim = torch.matmul(z.detach(), sub_topic_emb.t()) * self.kappa

        p = F.softmax(sim, dim=-1)
        # print("p shape:", p.shape)
        return p


    def topic_sim(self, z, sub=-1):
        if sub == -1:
            self.topic_emb.data = F.normalize(self.topic_emb.data, dim=-1)
            sim = torch.matmul(z, self.topic_emb.t())
        else:
            self.sub_topic_emb.data = F.normalize(self.sub_topic_emb.data, dim=-1)
            # self.sub_topic_emb[sub].data = F.normalize(self.sub_topic_emb.data, dim=-1)
            sim = torch.matmul(z, self.sub_topic_emb[sub].t())
        # self.topic_emb.data = F.normalize(self.topic_emb.data, dim=-1)
        return sim

    # return initialized latent word embeddings
    def init_emb(self, input_ids, attention_mask, valid_pos):
        self.bert.eval()
        bert_outputs = self.bert(input_ids,
                                 attention_mask=attention_mask)
        last_hidden_states = bert_outputs[0]
        attention_mask[:, 0] = 0
        attn_mask = valid_pos != 0
        input_embs = last_hidden_states[attn_mask]
        _, z = self.ae(input_embs)
        return z

    def forward(self, input_ids, attention_mask, valid_pos=None, pretrain=False):
        self.bert.eval()
        bert_outputs = self.bert(input_ids,
                                 attention_mask=attention_mask)
        last_hidden_states = bert_outputs[0]

        if pretrain:
            attn_mask = attention_mask != 0
            input_embs = last_hidden_states[attn_mask]
            output_embs, _ = self.ae(input_embs)
            return input_embs, output_embs
        else:
            assert valid_pos is not None, "valid_pos should not be None in clustering mode!"
        attention_mask[:, 0] = 0
        sum_emb = (last_hidden_states * attention_mask.unsqueeze(-1)).sum(dim=1)
        valid_num = attention_mask.sum(dim=-1, keepdim=True)
        avg_doc_emb = sum_emb / valid_num               # h bar (L rec)
        trans_states = self.dense(last_hidden_states)
        trans_states = self.activation(trans_states)
        attn_logits = torch.matmul(trans_states, self.v)
        attention_mask[:, 0] = 0
        attn_mask = attention_mask == 0
        attn_logits.masked_fill_(attn_mask, float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=-1)
        doc_emb = (last_hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)
        
        attn_mask = valid_pos != 0
        input_embs = last_hidden_states[attn_mask]      # hi(w)
        output_embs, z_word = self.ae(input_embs)       # zi(w)
        _, z_doc = self.ae(doc_emb)
        p_doc = self.cluster_assign(z_doc)              # p(tk|z(d))
        p_word = self.cluster_assign(z_word)            # p(tk|zi(w))

        # topic_nums_p = torch.argmax(p_word, dim=1)
        topic_nums_d = torch.argmax(p_doc, dim=1)
        
        sub_p_word = self.subcluster_assign(z_word)
        sub_p_doc = self.subcluster_assign(z_doc)

        dec_topic = self.ae.decode(self.topic_emb)      # tk hat
        sub_topic_emb = self.sub_topic_emb.view(-1, self.sub_topic_emb.shape[-1])
        # dec_sub_topic = torch.stack([self.ae.decode(s) for s in self.sub_topic_emb]).transpose(1, 2)
        dec_sub_topic = self.ae.decode(sub_topic_emb)
        # print('================================')
        # print('dec sub topic shape: ', dec_sub_topic.shape)
        # print(sub_p_doc.shape)


        rec_doc_emb = torch.matmul(p_doc, dec_topic)    # h(d) hat 
        # sub_rec_doc_emb = torch.stack([torch.matmul(sub_p_doc[i], dec_sub_topic[i]) for i in topic_nums_d])
        sub_rec_doc_emb = torch.matmul(sub_p_doc, dec_sub_topic)
        # print('sub sim shape: ', sub_sim.shape)
        # print('sub rec doc emb shape: ', sub_rec_doc_emb.shape)
        # p_doc = p_doc.unsqueeze(2)
        # print('p doc shape: ', p_doc.shape)
        # sub_rec_doc_emb = torch.matmul(sub_rec_doc_emb, p_doc).squeeze(2)
        # print('sub rec doc emb shape: ', sub_rec_doc_emb.shape)
        p_sub_topic = self.cluster_assign(sub_topic_emb, detach=True, softmax=False)
        # print('===============================')
        # print(sub_p_doc.shape)
        # print(dec_sub_topic[0].shape)
        return avg_doc_emb, input_embs, output_embs, rec_doc_emb, p_word, sub_p_word, sub_rec_doc_emb, p_sub_topic

    def inference(self, input_ids, attention_mask, sub=-1):
        self.bert.eval()
        bert_outputs = self.bert(input_ids,
                                 attention_mask=attention_mask)
        # bert embeddings
        # shape: (32, 512, 768) - 32 documents, each document has 512 words (include paddings), each word has 768 dimensions
        last_hidden_states = bert_outputs[0]
        attention_mask[:, 0] = 0
        trans_states = self.dense(last_hidden_states)
        trans_states = self.activation(trans_states)
        attn_logits = torch.matmul(trans_states, self.v)
        attention_mask[:, 0] = 0
        attn_mask = attention_mask == 0
        attn_logits.masked_fill_(attn_mask, float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=-1)
        # shape: (32, 768) - 32 document embeddings 
        doc_emb = (last_hidden_states * attn_weights.unsqueeze(-1)).sum(dim=1)
        
        valid_word_embs = last_hidden_states[~attn_mask]
        valid_word_ids = input_ids[~attn_mask]
        # shape: (x, 100) - x valid words in latent space, x < 32 * 512
        _, z_word = self.ae(valid_word_embs)

        if sub == -1:
            sim = self.topic_sim(z_word)
        else:
            sim = self.topic_sim(z_word, sub=sub)

        # shape (32, 100) - 32 docs in latent space
        _, z_doc = self.ae(doc_emb)
        return z_doc, z_word, valid_word_ids, sim
