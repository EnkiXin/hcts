import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import torch.nn as nn
from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import EmbLoss
from recbole.utils import InputType


class CDAE(CrossDomainRecommender):
    r"""BiTGCF uses feature propagation and feature transfer to achieve bidirectional
        knowledge transfer between the two domains.
        We extend the basic BiTGCF model in a symmetrical way to support those datasets that have overlapped items.
    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(CDAE, self).__init__(config, dataset)

        # load dataset info
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field
        # load parameters info
        self.device = config['device']
        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.domain_lambda_source = config['lambda_source']  # float32 type: the weight of source embedding in transfer function
        self.domain_lambda_target = config['lambda_target']  # float32 type: the weight of target embedding in transfer function
        self.drop_rate = config['drop_rate']  # float32 type: the dropout rate
        self.connect_way = config['connect_way']  # str type: the connect way for all layers
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

        self.encoder_x = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
            )
        self.decoder_x = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.NUM_MOVIE)
            )
        self.encoder_y = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
            )
        self.decoder_y = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.NUM_BOOK)
            )
        self.orthogonal_w = nn.Parameter(nn.init.xavier_uniform(torch.Tensor(self.latent_dim, self.latent_dim).type(
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor), gain=np.sqrt(2.0)),
                              requires_grad=True)

        self.dropout = nn.Dropout(p=self.drop_rate)
        self.loss = nn.BCELoss()
        self.reg_loss = EmbLoss()
        # generate intermediate data
        self.target_restore_user_e = None
        self.target_restore_item_e = None
        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ['target_restore_user_e', 'target_restore_item_e']

    def orthogonal_map(self, z_x, z_y):
        mapped_z_x = torch.matmul(z_x, self.orthogonal_w)
        mapped_z_y = torch.matmul(z_y, torch.transpose(self.orthogonal_w, 1, 0))
        return mapped_z_x, mapped_z_y

    def forward(self, batch_user, batch_user_x, batch_user_y):
        h_user_x = self.encoder_x(self.dropout(batch_user_x))
        h_user_y = self.encoder_y(self.dropout(batch_user_y))
        h_user = self.user_embeddings(batch_user)
        feature_x = torch.add(h_user_x, h_user)
        feature_y = torch.add(h_user_y, h_user)
        z_x = F.relu(feature_x)
        z_y = F.relu(feature_y)
        preds_x = self.decoder_x(z_x)
        preds_y = self.decoder_y(z_y)
        mapped_z_x, mapped_z_y = self.orthogonal_map(z_x, z_y)
        preds_x2y = self.decoder_y(mapped_z_x)
        preds_y2x = self.decoder_x(mapped_z_y)
        # define orthogonal constraint loss
        z_x_ = torch.matmul(mapped_z_x, torch.transpose(self.orthogonal_w, 1, 0))
        z_y_ = torch.matmul(mapped_z_y, self.orthogonal_w)
        z_x_reg_loss = torch.norm(z_x - z_x_, p=1, dim=1)
        z_y_reg_loss = torch.norm(z_y - z_y_, p=1, dim=1)
        return preds_x, preds_y, preds_x2y, preds_y2x, feature_x, feature_y, z_x_reg_loss, z_y_reg_loss

    def get_ego_embeddings(self, domain='source'):
        if domain == 'source':
            user_embeddings = self.source_user_embedding.weight
            item_embeddings = self.source_item_embedding.weight
            norm_adj_matrix = self.source_norm_adj_matrix
        else:
            user_embeddings = self.target_user_embedding.weight
            item_embeddings = self.target_item_embedding.weight
            norm_adj_matrix = self.target_norm_adj_matrix
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings, norm_adj_matrix






    def calculate_loss(self, interaction):
        self.init_restore_e()
        source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings = self.forward()
        losses = []


        source_user = interaction[self.SOURCE_USER_ID]
        source_item = interaction[self.SOURCE_ITEM_ID]
        source_label = interaction[self.SOURCE_LABEL]
        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_label = interaction[self.TARGET_LABEL]



        source_u_embeddings = source_user_all_embeddings[source_user]
        source_i_embeddings = source_item_all_embeddings[source_item]
        target_u_embeddings = target_user_all_embeddings[target_user]
        target_i_embeddings = target_item_all_embeddings[target_item]
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