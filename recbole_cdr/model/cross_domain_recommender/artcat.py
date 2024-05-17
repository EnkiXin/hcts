import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import EmbLoss
from recbole.utils import InputType
import torch.nn.functional as F
import math

class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):

        scores = torch.einsum('ij,kj->ik',query,key) / math.sqrt(query.size(-1))
        p_attn = F.softmax(scores, dim=-1)
        #[1850,1850]   [1850,64]
        return torch.einsum('ij,jk->ik',p_attn,value)

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        x = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        return self.output_linear(x)


class ARTCAT(CrossDomainRecommender):
    input_type = InputType.POINTWISE
    def __init__(self, config, dataset):
        super(ARTCAT, self).__init__(config, dataset)

        # load dataset info
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field
        # load parameters info
        self.device = config['device']
        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.drop_rate = config['drop_rate']  # float32 type: the dropout rate
        # define layers and loss
        self.source_user_embedding = torch.nn.Embedding(num_embeddings=self.total_num_users, embedding_dim=self.latent_dim)
        self.target_user_embedding = torch.nn.Embedding(num_embeddings=self.total_num_users, embedding_dim=self.latent_dim)
        self.source_item_embedding = torch.nn.Embedding(num_embeddings=self.total_num_items, embedding_dim=self.latent_dim)
        self.target_item_embedding = torch.nn.Embedding(num_embeddings=self.total_num_items, embedding_dim=self.latent_dim)
        with torch.no_grad():
            self.source_user_embedding.weight[self.overlapped_num_users: self.target_num_users].fill_(0)
            self.source_item_embedding.weight[self.overlapped_num_items: self.target_num_items].fill_(0)
            self.target_user_embedding.weight[self.target_num_users:].fill_(0)
            self.target_item_embedding.weight[self.target_num_items:].fill_(0)
        self.dropout = nn.Dropout(p=self.drop_rate)
        self.loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.reg_loss = EmbLoss()
        # generate intermediate data
        self.attention = MultiHeadedAttention(h=1, d_model=self.latent_dim)

        self.target_restore_user_e = None
        self.target_restore_item_e = None
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ['target_restore_user_e', 'target_restore_item_e']

    def get_ego_embeddings(self, domain='source'):
        if domain == 'source':
            user_embeddings = self.source_user_embedding.weight
            item_embeddings = self.source_item_embedding.weight
        else:
            user_embeddings = self.target_user_embedding.weight
            item_embeddings = self.target_item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings


    def getUserEmbed(self,source_commen_embedding,target_commen_embedding):
        user_embed = self.attention(target_commen_embedding, source_commen_embedding,source_commen_embedding)
        return user_embed+target_commen_embedding



    def forward(self):
        source_all_embeddings = self.get_ego_embeddings(domain='source')
        target_all_embeddings = self.get_ego_embeddings(domain='target')
        source_user_all_embeddings, source_item_all_embeddings = torch.split(source_all_embeddings,
                                                                   [self.total_num_users, self.total_num_items])
        target_user_all_embeddings, target_item_all_embeddings = torch.split(target_all_embeddings,
                                                                   [self.total_num_users, self.total_num_items])
        target_user_all_embeddings=self.transfer_layer(source_user_all_embeddings,target_user_all_embeddings)
        return source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings



    def transfer_layer(self,source_user_embeddings,target_user_embeddings):
        source_overlap_user_embeddings = source_user_embeddings[:self.overlapped_num_users]
        target_overlap_user_embeddings = target_user_embeddings[:self.overlapped_num_users]
        target_overlap_user_embeddings=self.getUserEmbed(source_overlap_user_embeddings,target_overlap_user_embeddings)
        target_specific_user_embeddings=target_user_embeddings[self.overlapped_num_users:]
        target_transfer_user_embeddings = torch.cat([target_overlap_user_embeddings, target_specific_user_embeddings],dim=0)
        return target_transfer_user_embeddings

    def calculate_loss(self, interaction):
        self.init_restore_e()
        losses = []
        source_user = interaction[self.SOURCE_USER_ID]
        source_item = interaction[self.SOURCE_ITEM_ID]
        source_label = interaction[self.SOURCE_LABEL]
        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_label = interaction[self.TARGET_LABEL]
        source_user_embedding,source_item_embedding,target_user_embedding,target_item_embedding=self.forward()
        source_u_embeddings = source_user_embedding[source_user]
        source_i_embeddings = source_item_embedding[source_item]
        target_u_embeddings = target_user_embedding[target_user]
        target_i_embeddings = target_item_embedding[target_item]
        # calculate BCE Loss in source domain
        source_output = self.sigmoid(torch.mul(source_u_embeddings, source_i_embeddings).sum(dim=1))
        source_bce_loss = self.loss(source_output, source_label)
        # calculate Reg Loss in source domain
        u_ego_embeddings = self.source_user_embedding(source_user)
        i_ego_embeddings = self.source_item_embedding(source_item)
        source_reg_loss = self.reg_loss(u_ego_embeddings, i_ego_embeddings)
        source_loss = source_bce_loss + self.reg_weight * source_reg_loss
        losses.append(source_loss)
        # calculate BCE Loss in target domain
        target_output = self.sigmoid(torch.mul(target_u_embeddings, target_i_embeddings).sum(dim=1))
        target_bce_loss = self.loss(target_output, target_label)
        # calculate Reg Loss in target domain
        u_ego_embeddings = self.target_user_embedding(target_user)
        i_ego_embeddings = self.target_item_embedding(target_item)
        target_reg_loss = self.reg_loss(u_ego_embeddings, i_ego_embeddings)
        target_loss = target_bce_loss + self.reg_weight * target_reg_loss
        losses.append(target_loss)
        return tuple(losses)

    def predict(self, interaction):
        result = []
        _, _, target_user_all_embeddings, target_item_all_embeddings = self.forward()
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        u_embeddings = target_user_all_embeddings[user]
        i_embeddings = target_item_all_embeddings[item]
        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        restore_user_e, restore_item_e = self.get_restore_e()
        u_embeddings = restore_user_e[user]
        i_embeddings = restore_item_e[:self.target_num_items]
        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores.view(-1)

    def init_restore_e(self):
        # clear the storage variable when training
        if self.target_restore_user_e is not None or self.target_restore_item_e is not None:
            self.target_restore_user_e, self.target_restore_item_e = None, None

    def get_restore_e(self):
        if self.target_restore_user_e is None or self.target_restore_item_e is None:
            _, _, self.target_restore_user_e, self.target_restore_item_e = self.forward()
        return self.target_restore_user_e, self.target_restore_item_e