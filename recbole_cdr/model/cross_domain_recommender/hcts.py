import numpy as np
import scipy.sparse as sp
from scipy.sparse import coo_matrix
import torch
import torch.nn as nn
from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
from hyperbolic_gnn.model.hgcn.layers.gnn import GAT,GCN,GATv2,GraphSAGE
from hyperbolic_gnn.model.hgcn.layers.hyp_layers import HyperbolicGraphConvolution,HNNLayer,HypLinear,LorentzLinear,FermiDiracDecoder
from hyperbolic_gnn.model.hgcn.layers.hyperbolic_contrastive_learning import HyperbolicGraphHyperbolicContrastive
import hyperbolic_gnn.model.hgcn.manifolds.hyperboloid as manifolds
from hyperbolic_gnn.model.hgcn.manifolds.base import ManifoldParameter

import dgl
import random
from recbole.utils import init_logger, init_seed, set_color



class HCTS(CrossDomainRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(HCTS, self).__init__(config, dataset)
        init_seed(config['seed'], config['reproducibility'])
        # load dataset info
        self.config=config
        self.dataset=dataset
        self.reg_weight = config['reg_weight']
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field
        # load parameters info
        self.device = config['device']
        # load parameters info
        self.latent_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.source_curve = nn.Parameter(torch.tensor(config['curve2']))
        self.target_curve = nn.Parameter(torch.tensor(config['curve1']))
        self.num_neg_samples = config['num_neg_samples']
        # define layers and loss
        self.manifold = getattr(manifolds, "Hyperboloid")()
        self.ManifoldParameter = ManifoldParameter
        self.margin = config['margin']
        self.ce_loss = nn.CrossEntropyLoss()
        self.batch_size=config['train_batch_size']
        self.lap_batch_num=config['lap_batch_num']
        self.dist_gate=config['dist_gate']
        self.cts_lamda=config['cts_lamda']
        self.temp=config['temp']
        self.num_neg=config['num_neg']
        self.ireg_lambda=config['ireg_lambda']
        self.ireg=config['ireg']
        self.activation = lambda x: x


        # total_num_users和total_num_items的数量比真实的数量多1，0号embedding是假的
        # source+target全部的user个数：total_num_users
        self.source_user_embedding = torch.nn.Embedding(num_embeddings=self.total_num_users, embedding_dim=self.latent_dim)
        self.source_user_embedding.state_dict()['weight'].uniform_(-config['scale'], config['scale'])



        self.target_user_embedding = torch.nn.Embedding(num_embeddings=self.total_num_users, embedding_dim=self.latent_dim)
        self.target_user_embedding.state_dict()['weight'].uniform_(-config['scale'], config['scale'])

        self.source_item_embedding = torch.nn.Embedding(num_embeddings=self.total_num_items, embedding_dim=self.latent_dim)
        self.source_item_embedding.state_dict()['weight'].uniform_(-config['scale'], config['scale'])

        self.target_item_embedding = torch.nn.Embedding(num_embeddings=self.total_num_items, embedding_dim=self.latent_dim)
        self.target_item_embedding.state_dict()['weight'].uniform_(-config['scale'], config['scale'])
        self.conv=config['conv']

        self.loss = nn.CrossEntropyLoss()
        # 固定住不需要的部分
        with torch.no_grad():
            # 第一部分：overlapped user+第二部分：没有overlapped的target user+第三部分：没有overlapped的source user
            # overlapped_num_users：两边都有的user的个数
            self.source_user_embedding.weight[self.overlapped_num_users: self.target_num_users].fill_(0)
            self.source_item_embedding.weight[self.overlapped_num_items: self.target_num_items].fill_(0)
            self.target_user_embedding.weight[self.target_num_users:].fill_(0)
            self.target_item_embedding.weight[self.target_num_items:].fill_(0)
        # 下面两个邻接矩阵是一整个图
        self.source_interaction_matrix = dataset.inter_matrix(form='coo', value_field=None, domain='source').astype(np.float32)
        self.target_interaction_matrix = dataset.inter_matrix(form='coo', value_field=None, domain='target').astype(np.float32)
        self.source_norm_adj_matrix = self.get_norm_adj_mat(self.source_interaction_matrix, self.total_num_users,self.total_num_items).to(self.device)
        self.target_norm_adj_matrix = self.get_norm_adj_mat(self.target_interaction_matrix, self.total_num_users,self.total_num_items).to(self.device)

        self.source_u , self.source_i = dataset.interactions(domain='source')
        self.target_u , self.target_i = dataset.interactions(domain='target')

        self.s2t_linear=HypLinear(self.manifold,self.latent_dim,self.latent_dim,self.source_curve,self.target_curve)
        self.t2s_linear=HypLinear(self.manifold,self.latent_dim,self.latent_dim,self.source_curve,self.target_curve)
        self.t2t_linear=HypLinear(self.manifold,self.latent_dim,self.latent_dim,self.target_curve,self.target_curve)
        self.s2s_linear=HypLinear(self.manifold,self.latent_dim,self.latent_dim,self.source_curve,self.source_curve)
        self.num_neg_samples=config['num_neg_samples']

        # storage variables for full sort evaluation acceleration
        self.target_restore_user_e = None
        self.target_restore_item_e = None
        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ['target_restore_user_e', 'target_restore_item_e']
        # q：source到target
        # k：target到target
        # v：target到target
        self.source_user_degree_count = torch.from_numpy(self.source_interaction_matrix.sum(axis=1)).to(self.device)
        self.target_user_degree_count = torch.from_numpy(self.target_interaction_matrix.sum(axis=1)).to(self.device)
        self.source_item_degree_count = torch.from_numpy(self.source_interaction_matrix.sum(axis=0)).transpose(0, 1).to(self.device)
        self.target_item_degree_count = torch.from_numpy(self.target_interaction_matrix.sum(axis=0)).transpose(0, 1).to(self.device)
        if self.conv=='graphconv':
           self.target_gnn = GCN(
            latent_dim=self.latent_dim,
            num_layers=self.n_layers,
            src=torch.cat((self.target_u,torch.tensor([torch.max(self.source_u)])),dim=0),
            dst=torch.cat((self.target_i,torch.tensor([torch.max(self.source_i)])),dim=0)+torch.max(self.source_u)+1,
            )
           self.source_gnn = GCN(
            latent_dim=self.latent_dim,
            num_layers=self.n_layers,
            src=torch.cat((self.source_u,torch.tensor([torch.max(self.source_u)])),dim=0),
            dst=torch.cat((self.source_i,torch.tensor([torch.max(self.source_i)])),dim=0)+max(self.source_u)+1,
            )
        elif self.conv=='gat':
            self.target_gnn = GAT(
                latent_dim=self.latent_dim,
                num_layers=self.n_layers,
                num_heads=2,
                src=torch.cat((self.target_u, torch.tensor([torch.max(self.source_u)])), dim=0),
                dst=torch.cat((self.target_i, torch.tensor([torch.max(self.source_i)])), dim=0) + torch.max(
                    self.source_u) + 1,
            )
            self.source_gnn = GAT(
                latent_dim=self.latent_dim,
                num_layers=self.n_layers,
                num_heads=2,
                src=torch.cat((self.source_u, torch.tensor([torch.max(self.source_u)])), dim=0),
                dst=torch.cat((self.source_i, torch.tensor([torch.max(self.source_i)])), dim=0) + max(
                    self.source_u) + 1,
            )
        elif self.conv=='gatv2':
            self.target_gnn = GATv2(
                latent_dim=self.latent_dim,
                num_layers=self.n_layers,
                num_heads=2,
                src=torch.cat((self.target_u, torch.tensor([torch.max(self.source_u)])), dim=0),
                dst=torch.cat((self.target_i, torch.tensor([torch.max(self.source_i)])), dim=0) + torch.max(
                    self.source_u) + 1,
            )
            self.source_gnn = GATv2(
                latent_dim=self.latent_dim,
                num_layers=self.n_layers,
                num_heads=2,
                src=torch.cat((self.source_u, torch.tensor([torch.max(self.source_u)])), dim=0),
                dst=torch.cat((self.source_i, torch.tensor([torch.max(self.source_i)])), dim=0) + max(
                    self.source_u) + 1,
            )
        elif self.conv == 'sage':
            self.target_gnn = GraphSAGE(
                latent_dim=self.latent_dim,
                num_layers=self.n_layers,
                src=torch.cat((self.target_u, torch.tensor([torch.max(self.source_u)])), dim=0),
                dst=torch.cat((self.target_i, torch.tensor([torch.max(self.source_i)])), dim=0) + torch.max(
                    self.source_u) + 1,
            )
            self.source_gnn = GraphSAGE(
                latent_dim=self.latent_dim,
                num_layers=self.n_layers,
                src=torch.cat((self.source_u, torch.tensor([torch.max(self.source_u)])), dim=0),
                dst=torch.cat((self.source_i, torch.tensor([torch.max(self.source_i)])), dim=0) + max(
                    self.source_u) + 1,
            )
        else:
            self.target_gnn = HyperbolicGraphConvolution(latent_dim=self.latent_dim,
                                                         num_layers=self.n_layers,
                                                         manifold=self.manifold,
                                                         curve=self.target_curve)
            self.source_gnn = HyperbolicGraphConvolution(latent_dim=self.latent_dim,
                                                         num_layers=self.n_layers,
                                                         manifold=self.manifold,
                                                         curve=self.source_curve)

        # 初始化时的变量source_u，source_i,target_u,target_i是整个interactions中两个interaction图的节点
        self.hyperbolic_contrastive_learning=HyperbolicGraphHyperbolicContrastive(latent_dim=self.latent_dim,
                                                                                  source_curve=self.source_curve,
                                                                                  target_curve=self.target_curve,
                                                                                  manifold=self.manifold,
                                                                                  num_lapped_users=self.overlapped_num_users,
                                                                                  temp=self.temp,
                                                                                  cts_lamda=self.cts_lamda,
                                                                                  source_u=self.source_u ,
                                                                                  source_i= self.source_i,
                                                                                  target_u=self.target_u ,
                                                                                  target_i=self.target_i,
                                                                                  num_neg_samples=self.num_neg_samples,
                                                                                  config=config)



    def hir_loss(self, embeddings,domain):
        if domain=='source':
          c=self.source_curve
        else:
            c=self.target_curve


        embeddings_tan = self.manifold.logmap0(embeddings, c)
        # centering has been achieved before
        tangent_mean_norm = (1e-6 + embeddings_tan.pow(2).sum(dim=1).mean())
        tangent_mean_norm = self.activation(-tangent_mean_norm)
        return tangent_mean_norm



    def get_norm_adj_mat(self, interaction_matrix, n_users=None, n_items=None):
        if n_users == None or n_items == None:
            n_users, n_items = interaction_matrix.shape
        A = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL


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

    def forward(self):
        source_all_embeddings, source_norm_adj_matrix = self.get_ego_embeddings(domain='source')
        target_all_embeddings, target_norm_adj_matrix = self.get_ego_embeddings(domain='target')
        source_all_embeddings=self.manifold.proj(self.manifold.expmap0(source_all_embeddings, self.source_curve),self.source_curve)
        target_all_embeddings=self.manifold.proj(self.manifold.expmap0(target_all_embeddings, self.target_curve),self.target_curve)

        # 原本是在双曲空间上，在gnn中的操作是先映射回欧氏空间，然后做聚合
        if self.conv=='skip':
            source_all_embeddings = self.source_gnn(source_all_embeddings,source_norm_adj_matrix)
            target_all_embeddings = self.target_gnn(target_all_embeddings,target_norm_adj_matrix)
        else:
            source_all_embeddings = self.source_gnn(source_all_embeddings)
            target_all_embeddings = self.target_gnn(target_all_embeddings)
        source_all_embeddings=self.manifold.proj(self.manifold.expmap0(source_all_embeddings, self.source_curve),self.source_curve)
        target_all_embeddings=self.manifold.proj(self.manifold.expmap0(target_all_embeddings, self.target_curve),self.target_curve)

        source_structure_loss=self.hir_loss(source_all_embeddings,'source')
        target_structure_loss=self.hir_loss(target_all_embeddings,'target')

        source_user_all_embeddings, source_item_all_embeddings = torch.split(source_all_embeddings,[self.total_num_users, self.total_num_items])
        target_user_all_embeddings, target_item_all_embeddings = torch.split(target_all_embeddings,[self.total_num_users, self.total_num_items])


        return source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings,source_structure_loss,target_structure_loss


    def get_visualization_data(self):
        source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings,_,_=self.forward()
        source_u_id,_ = torch.sort(self.source_u)
        source_u_id,source_u_degree=torch.unique(source_u_id,return_counts=True)
        source_i_id,_ = torch.sort(self.source_i)
        source_i_id,source_i_degree=torch.unique(source_i_id,return_counts=True)
        target_u_id, _ = torch.sort(self.target_u)
        target_u_id,target_u_degree = torch.unique(target_u_id,return_counts=True)
        target_i_id,_ = torch.sort(self.target_i)
        target_i_id, target_i_degree = torch.unique(target_i_id,return_counts=True)
        return source_user_all_embeddings[source_u_id],source_item_all_embeddings[source_i_id],target_user_all_embeddings[target_u_id],target_item_all_embeddings[target_i_id]

    def source_decode(self, user_embedding,item_embedding,neg_item_embedding,label):
        pos_score = self.manifold.sqdist(user_embedding, item_embedding, self.source_curve)
        neg_score = self.manifold.sqdist(user_embedding, neg_item_embedding, self.source_curve)
        loss = pos_score - neg_score + self.margin
        loss[loss<0]=0
        loss=loss.squeeze()
        non_zero = torch.count_nonzero(loss).item()
        if non_zero > 0:
            loss = loss / non_zero
        loss=torch.dot(loss,label)
        return loss

    def target_decode(self, user_embedding,item_embedding,neg_item_embedding,label):
        pos_score = self.manifold.sqdist(user_embedding, item_embedding, self.target_curve)
        neg_score = self.manifold.sqdist(user_embedding, neg_item_embedding, self.target_curve)
        loss = pos_score - neg_score + self.margin
        loss[loss<0]=0
        loss=loss.squeeze()
        non_zero = torch.count_nonzero(loss).item()
        if non_zero > 0:
            loss = loss / non_zero
        loss=torch.dot(loss,label)
        return loss


    def generate_negative_samples(self,total_num_items, items, domain):
        if domain=='source':
           negative_samples_indices = torch.randint(self.target_num_items, total_num_items, (len(items), 1), dtype=torch.int64).to(items.device)
           negative_samples_indices = negative_samples_indices.view(-1, 1).squeeze()

        else:
           negative_samples_indices = torch.randint(1, self.target_num_items, (len(items), 1), dtype=torch.int64).to(items.device)
           negative_samples_indices = negative_samples_indices.view(-1, 1).squeeze()

        return negative_samples_indices


    def calculate_loss(self, interaction):
        self.get_visualization_data()
        self.init_restore_e()
        # forward是先过一个图神经网络，这里是将整个图直接处理一遍，然后得到所有user，item的embeddings
        source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings,source_structure_loss,target_structure_loss = self.forward()
        
        losses = []
        # 取出每个iteraction（伯乐的mini-batch）的user和item的ID
        source_user = interaction[self.SOURCE_USER_ID]
        source_item = interaction[self.SOURCE_ITEM_ID]
        # 在source中随机采样一个item作为训练的负样本
        source_neg_item = self.generate_negative_samples(self.total_num_items,source_item,'source')
        source_label = interaction[self.SOURCE_LABEL]
        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        # 在target中随机采样一个item作为训练的负样本
        target_neg_item=self.generate_negative_samples(self.total_num_items,target_item,'target')
        # 根据评分来给label，rating在阈值以下的为0，以上的为1
        # 训练跟message passing边不一样的问题
        target_label = interaction[self.TARGET_LABEL]
        # 从forward中取出每个user，item的embedding，这里是经过HGCF以后的user和item的embedding

        source_u_embeddings = source_user_all_embeddings[source_user]
        source_i_embeddings = source_item_all_embeddings[source_item]
        source_neg_i_embeddings = source_item_all_embeddings[source_neg_item]
        target_u_embeddings = target_user_all_embeddings[target_user]
        target_i_embeddings = target_item_all_embeddings[target_item]
        target_neg_i_embeddings= target_item_all_embeddings[target_neg_item]

        source_loss=self.source_decode(source_u_embeddings, source_i_embeddings, source_neg_i_embeddings, source_label)
        losses.append(source_loss)
        target_loss=self.target_decode(target_u_embeddings, target_i_embeddings, target_neg_i_embeddings, target_label)
        losses.append(target_loss)
        # 输入到cts的东西：source_user_all_embeddings是source_user全部的embeddings（包括target中不需要的部分），另外几个embeddings意义相同
        # source_user：一个batch中，source user的id，后面四个分别是一个batch的source_item,target_user,target_item


        if (self.config['s_t_transfer'] == False) & (self.config['t_s_transfer'] == False):
            pass
        else:
           cts_loss=self.hyperbolic_contrastive_learning(source_user_all_embeddings,
                                                      source_item_all_embeddings,
                                                      target_user_all_embeddings,
                                                      target_item_all_embeddings,
                                                      source_user,
                                                      source_item,
                                                      target_user,
                                                      target_item)
           losses.append(cts_loss)

        if self.ireg_lambda>0:
          structure_loss = self.ireg_lambda * (max(source_structure_loss, -10) + 10)+self.ireg_lambda * (max(target_structure_loss, -10) + 10)
          losses.append(structure_loss)

        return tuple(losses)

    def predict(self, interaction):
        result = []
        _, _, target_user_all_embeddings, target_item_all_embeddings = self.forward()
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        u_embeddings = target_user_all_embeddings[user]
        i_embeddings = target_item_all_embeddings[item]
        scores = -self.manifold.sqdist(u_embeddings, i_embeddings,self.c)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        restore_user_e, restore_item_e = self.get_restore_e()
        u_embeddings = restore_user_e[user]
        i_embeddings = restore_item_e[:self.target_num_items]

        scores = -self.manifold.sqdist(u_embeddings, i_embeddings,self.target_curve)

        return scores.view(-1)

    def init_restore_e(self):
        # clear the storage variable when training
        if self.target_restore_user_e is not None or self.target_restore_item_e is not None:
            self.target_restore_user_e, self.target_restore_item_e = None, None

    def get_restore_e(self):
        if self.target_restore_user_e is None or self.target_restore_item_e is None:
            _, _, self.target_restore_user_e, self.target_restore_item_e,_,_= self.forward()
        return self.target_restore_user_e, self.target_restore_item_e
