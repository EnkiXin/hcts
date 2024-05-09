import torch
import torch.nn as nn
import copy
import dgl
import numpy as np
import random
from hyperbolic_gnn.model.hgcn.layers.hyp_layers import HyperbolicGraphConvolution,HNNLayer,HypLinear,LorentzLinear,FermiDiracDecoder
dgl.seed(2023)
class HyperbolicGraphHyperbolicContrastive(nn.Module):
    """
    HCTS
    """
    def __init__(self,
                 latent_dim,
                 source_curve,
                 target_curve,
                 manifold,
                 num_lapped_users,
                 temp,
                 cts_lamda,
                 source_u,
                 source_i,
                 target_u,
                 target_i,
                 num_neg_samples,
                 config
                 ):
        super(HyperbolicGraphHyperbolicContrastive, self).__init__()
        dgl.seed(config['seed'])
        self.latent_dim = latent_dim
        self.source_curve = source_curve
        self.target_curve = target_curve
        self.manifold = manifold
        self.s2t_linear = HypLinear(self.manifold,
                                    self.latent_dim,
                                    self.latent_dim,
                                    self.source_curve,
                                    self.target_curve)

        self.t2s_linear = HypLinear(self.manifold,
                                    self.latent_dim,
                                    self.latent_dim,
                                    self.source_curve,
                                    self.target_curve)

        self.t2t_linear = HypLinear(self.manifold,
                                    self.latent_dim,
                                    self.latent_dim,
                                    self.target_curve,
                                    self.target_curve)

        self.s2s_linear = HypLinear(self.manifold,
                                    self.latent_dim,
                                    self.latent_dim,
                                    self.source_curve,
                                    self.source_curve)

        self.ce_loss = nn.CrossEntropyLoss()
        self.num_lapped_users = num_lapped_users
        self.temp = temp
        self.cts_lamda = cts_lamda
        self.source_u = source_u
        self.source_i = source_i
        self.target_u = target_u
        self.target_i = target_i
        self.num_target_u = max(self.target_u)
        self.num_target_i = max(self.target_i)
        self.num_source_u = max(self.source_u)-self.num_target_u+self.num_lapped_users
        self.num_source_i = max(self.source_i)-self.num_target_i
        self.target_g = self.generate_graph(target_u, target_i ,'target').to('cuda:0')
        self.source_g = self.generate_graph(source_u,source_i,'soruce').to('cuda:0')
        self.source_neg_g = self.generate_graph_neg(source_u,source_i,'source').to('cuda:0')
        self.target_neg_g = self.generate_graph_neg(target_u, target_i,'target').to('cuda:0')
        self.num_neg_samples=num_neg_samples
        self.config=config


    def inter2graph_id(self,src,dst,domain):
        # user_id不变，item_id变成item_id+max(user_id)
        u_id=copy.deepcopy(src)
        i_id=copy.deepcopy(dst)
        if domain=='target':
            u_id = u_id-1
            i_id = i_id-1+self.num_target_u
        else:
            u_id[u_id<=self.num_lapped_users]=u_id[u_id<=self.num_lapped_users]-1
            u_id[u_id>self.num_lapped_users]=u_id[u_id>self.num_lapped_users]-self.num_target_u+self.num_lapped_users-1
            i_id=i_id-self.num_target_i-1+self.num_source_u
        return u_id,i_id

    def graph_id2inter_id(self,src,dst,domain):
        u_id = copy.deepcopy(src)
        i_id = copy.deepcopy(dst)
        if domain == 'target':
           u_id = u_id + 1
           i_id = i_id + 1 - self.num_target_u
        else:
            u_id[u_id > (self.num_lapped_users-1)] = u_id[u_id > (self.num_lapped_users-1)] + self.num_target_u-self.num_lapped_users + 1
            u_id[u_id <= (self.num_lapped_users - 1)] = u_id[u_id <= (self.num_lapped_users - 1)]+1
            i_id = i_id + self.num_target_i + 1 - self.num_source_u
        return u_id , i_id

    def generate_graph(self,src,dst,domain):
        u_id = copy.deepcopy(src)
        i_id = copy.deepcopy(dst)
        u_id,i_id=self.inter2graph_id(u_id,i_id,domain)
        # u-i的graph
        g = dgl.graph((u_id,i_id))
        return g

    def generate_graph_neg(self,u_id,i_id,domain):
        if domain == 'target':
           t_u_id = copy.deepcopy(u_id)
           t_i_id = copy.deepcopy(i_id)
           t_u_id, t_i_id = self.inter2graph_id(t_u_id, t_i_id, domain)
           g = dgl.heterograph({
            ('u', 'inter', 'game'): (t_u_id, t_i_id)
           })
        else:
            s_u_id=copy.copy(u_id)
            s_i_id=copy.copy(i_id)
            s_u_id, s_i_id = self.inter2graph_id(s_u_id, s_i_id, domain)
            g = dgl.heterograph({
                ('u', 'inter', 'game'): (s_u_id, s_i_id)
            })

        neg_g=dgl.sampling.global_uniform_negative_sampling(g, 10000000, etype='inter')
        neg_g=dgl.graph(neg_g)
        return neg_g

    def graph_sample(self,overlap_user_id):
        # 为source和target分别生成一个negative graph
        # 在source_g和target_g中，为每个overlapped节点采样1条边
        # 在source_g上，为每个overlapped user，sample出一个item，作为正样本


        # sample_neighbors：从source_g上sample边，为overlap_user_id来sample，每个节点sample一条边
        sampled_source_pos_g = dgl.sampling.sample_neighbors(self.source_g,
                                                           overlap_user_id,
                                                           1,
                                                           edge_dir = 'out')

        # 在target_g上，为每个overlapped user，sample出一个item，作为正样本
        sampled_target_pos_g = dgl.sampling.sample_neighbors(self.target_g,
                                                           overlap_user_id,
                                                           1,
                                                           edge_dir = 'out')


        # 在neg_source_g和neg_target_g中，为每个overlapped节点采样10条边
        sampled_source_neg_g = dgl.sampling.sample_neighbors(self.source_neg_g,
                                                           overlap_user_id,
                                                           self.num_neg_samples,
                                                           edge_dir = 'out')


        sampled_target_neg_g = dgl.sampling.sample_neighbors(self.target_neg_g,
                                                           overlap_user_id,
                                                           self.num_neg_samples,
                                                           edge_dir = 'out')

        sampled_source_i_i_pos_g = dgl.sampling.sample_neighbors(self.source_g,
                                                             overlap_user_id,
                                                             self.num_neg_samples,
                                                             edge_dir='out')

        sampled_target_i_i_pos_g = dgl.sampling.sample_neighbors(self.target_g,
                                                             overlap_user_id,
                                                             self.num_neg_samples,
                                                             edge_dir='out')

        # source_graph中的overlapped_users的id
        source_pos_u_id = sampled_source_pos_g.edges()[0]
        # source_graph中overlapeed_users的邻居的id
        source_pos_i_id = sampled_source_pos_g.edges()[1]
        # target_graph中的overlapped_users的id
        target_pos_u_id = sampled_target_pos_g.edges()[0]
        # target_graph中overlaped_users的邻居的id
        target_pos_i_id = sampled_target_pos_g.edges()[1]
        source_neg_u_id = sampled_source_neg_g.edges()[0]
        source_neg_i_id = sampled_source_neg_g.edges()[1]
        target_neg_u_id = sampled_target_neg_g.edges()[0]
        target_neg_i_id = sampled_target_neg_g.edges()[1]

        source_i_i_neg_id_u = sampled_source_i_i_pos_g.edges()[0]
        source_i_i_neg_id=sampled_source_i_i_pos_g.edges()[1]
        target_i_i_neg_id_u = sampled_target_i_i_pos_g.edges()[0]
        target_i_i_neg_id = sampled_target_i_i_pos_g.edges()[1]

        source_pos_u_id,source_pos_i_id=self.graph_id2inter_id(source_pos_u_id,source_pos_i_id, 'source')
        source_neg_u_id,source_neg_i_id=self.graph_id2inter_id(source_neg_u_id,source_neg_i_id, 'source')
        target_pos_u_id,target_pos_i_id=self.graph_id2inter_id(target_pos_u_id,target_pos_i_id, 'target')
        target_neg_u_id,target_neg_i_id=self.graph_id2inter_id(target_neg_u_id,target_neg_i_id, 'target')
        _,source_i_i_neg_id=self.graph_id2inter_id(source_i_i_neg_id_u,source_i_i_neg_id, 'source')
        _,target_i_i_neg_id=self.graph_id2inter_id(target_i_i_neg_id_u,target_i_i_neg_id, 'source')



        # source_pos_u_id,source_pos_i_id：source graph上sample出来的边
        # source_neg_u_id,source_neg_i_id：neg source graph上sample出来的边
        return source_pos_u_id,source_pos_i_id,source_neg_u_id,source_neg_i_id,target_pos_u_id,target_pos_i_id,target_neg_u_id,target_neg_i_id,source_i_i_neg_id,target_i_i_neg_id

    def get_dist_matrix(self,
                        temp,
                        pos_a,
                        pos_b,
                        neg_a,
                        neg_b,
                        domain):
        # pos_a和pos_b是同维度的embeddings，需要拉近而这距离；neg_a，neg_b也是同纬度，要推远neg_a，neg_b的距离。
        # domain：在哪个domain的双曲空间进行对比学习
        if domain == 'source':
        # 这时，是将source_domain的embeddings与target_domain的item_embeddings进行对比学习
        #   pos_dist = 2*torch.sigmoid(-self.manifold.hyper_dist(self.source_curve,pos_user_embeddings, pos_item_embeddings))/temp
            pos_dist = 2*torch.sigmoid(-self.manifold.hyper_dist(self.source_curve,pos_a, pos_b))/temp
            neg_dist = 2*torch.sigmoid(-self.manifold.hyper_dist(self.source_curve,neg_a, neg_b))/temp



        else:
        # 这时，是将target_domain的embeddings与source_domain的item_embeddings进行对比学习
            pos_dist = 2*torch.sigmoid(-self.manifold.hyper_dist(self.target_curve,pos_a, pos_b))/temp
            neg_dist = 2*torch.sigmoid(-self.manifold.hyper_dist(self.target_curve,neg_a, neg_b))/temp
        pos_dist=pos_dist.view(-1, 1)
        neg_dist=neg_dist.view(pos_dist.shape[0], -1)
        dist_matrix = torch.cat([pos_dist, neg_dist], dim=1)
        return dist_matrix

    def mask_correlated_samples1(self, labels_dis):
        mask = torch.ones((labels_dis.shape[0], labels_dis.shape[1]),device=labels_dis.device)
        mask = torch.tensor(mask-labels_dis,dtype=bool)
        return mask

    def mask_correlated_samples2(self, labels_dis):
        mask =torch.tensor(labels_dis,dtype=bool,device=labels_dis.device)
        return mask


    # 给两个domain的overlapped user的embeddings，同一个user两个domain的embedding，距离拉近；不同user
    def u_u_hyperbolic_cts_loss(self, z_i, z_j, c, temp):
        len = z_i.shape[0]
        # 向量之间两两计算距离
        hyper_dist = 2*torch.sigmoid(-self.manifold.matrix_sqdist(z_i,z_j,c)/temp)
        # hyper_dist = -self.manifold.matrix_sqdist(z_i, z_j, c) / temp
        # 将向量自己与自己的距离mask掉
        labels_dis = torch.eye(hyper_dist.size(0), device=hyper_dist.device)
        filter_hyper_dist= hyper_dist
        mask1 = self.mask_correlated_samples1(labels_dis)
        mask2 = self.mask_correlated_samples2(labels_dis)
        negative_samples = filter_hyper_dist[mask1].reshape(len, -1)
        positive_samples = filter_hyper_dist[mask2].reshape(len, -1)
        labels = torch.zeros(len).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.ce_loss(logits, labels)
        return loss




    # 下面，输入进来的embeddings是已经进行了manifold alignment的向量
    def u_i_hyperbolic_cts_loss(self,
                                user_all_embeddings,
                                item_all_embeddings,
                                overlap_user_id,
                                temp,
                                way):
        # 输入：user_embedding,item_embedding,overlap_id,temp,way
        # 从overlapped user去sample其正负样本

        source_pos_u_id,source_pos_i_id, source_neg_u_id, source_neg_i_id, target_pos_u_id, target_pos_i_id, target_neg_u_id, target_neg_i_id,_,_ = self.graph_sample(overlap_user_id-1)

        if way=='su-ti':
           pos_u_embeddings = user_all_embeddings[source_pos_u_id]
           pos_i_embeddings = item_all_embeddings[target_pos_i_id]
           neg_u_embeddings = user_all_embeddings[source_neg_u_id]
           neg_i_embeddings = item_all_embeddings[target_neg_i_id]
           dist_matrix = self.get_dist_matrix(temp,
                                                  pos_u_embeddings,
                                                  pos_i_embeddings,
                                                  neg_u_embeddings,
                                                  neg_i_embeddings,
                                                     'target')
        elif way=='si-tu':
          pos_u_embeddings = user_all_embeddings[target_pos_u_id]
          pos_i_embeddings = item_all_embeddings[source_pos_i_id]
          neg_u_embeddings = user_all_embeddings[target_neg_u_id]
          neg_i_embeddings = item_all_embeddings[source_neg_i_id]
          dist_matrix = self.get_dist_matrix(temp,
                                                      pos_u_embeddings,
                                                      pos_i_embeddings,
                                                      neg_u_embeddings,
                                                      neg_i_embeddings,
                                                      'target')
        elif way == 'tu-si':
            pos_u_embeddings = user_all_embeddings[target_pos_u_id]
            pos_i_embeddings = item_all_embeddings[source_pos_i_id]
            neg_u_embeddings = user_all_embeddings[target_neg_u_id]
            neg_i_embeddings = item_all_embeddings[source_neg_i_id]
            dist_matrix = self.get_dist_matrix(temp,
                                                      pos_u_embeddings,
                                                      pos_i_embeddings,
                                                      neg_u_embeddings,
                                                      neg_i_embeddings,
                                                      'source')
        else:
           pos_u_embeddings = user_all_embeddings[source_pos_u_id]
           pos_i_embeddings = item_all_embeddings[target_pos_i_id]
           neg_u_embeddings = user_all_embeddings[source_neg_u_id]
           neg_i_embeddings = item_all_embeddings[target_neg_i_id]
           dist_matrix = self.get_dist_matrix(temp,
                                                  pos_u_embeddings,
                                                  pos_i_embeddings,
                                                  neg_u_embeddings,
                                                  neg_i_embeddings,
                                                     'source')
        source_labels = torch.zeros(dist_matrix.shape[0]).to(dist_matrix.device).long()
        loss = self.ce_loss(dist_matrix, source_labels)
        return loss

    def i_i_hyperbolic_cts_loss(self,
                                source_i_all_embeddings,
                                target_i_all_embeddings,
                                overlap_user_id,
                                temp,
                                way):
        # 输入：user_embedding,item_embedding,overlap_id,temp,way
        # 从overlapped user去sample其正负样本

        source_pos_u_id, source_pos_i_id, source_neg_u_id, source_neg_i_id, target_pos_u_id, target_pos_i_id, target_neg_u_id, target_neg_i_id ,source_i_i_neg_id,target_i_i_neg_id= self.graph_sample(
            overlap_user_id - 1)









        if way == 'si-ti':

            source_i_i_neg_id=source_pos_i_id.repeat_interleave(self.config['num_neg_samples'])
            pos_a_embeddings = source_i_all_embeddings[source_pos_i_id]
            pos_b_embeddings = target_i_all_embeddings[target_pos_i_id]
            neg_a_embeddings = source_i_all_embeddings[source_i_i_neg_id]
            neg_b_embeddings = target_i_all_embeddings[target_neg_i_id]
            dist_matrix = self.get_dist_matrix(temp,
                                               pos_a_embeddings,
                                               pos_b_embeddings,
                                               neg_a_embeddings,
                                               neg_b_embeddings,
                                               'target')
        elif way == 'ti-si':
            target_i_i_neg_id = target_pos_i_id.repeat_interleave(self.config['num_neg_samples'])
            pos_a_embeddings = target_i_all_embeddings[target_pos_i_id]
            pos_b_embeddings = source_i_all_embeddings[source_pos_i_id]
            neg_a_embeddings = target_i_all_embeddings[target_i_i_neg_id]
            neg_b_embeddings = source_i_all_embeddings[source_neg_i_id]
            dist_matrix = self.get_dist_matrix(temp,
                                               pos_a_embeddings,
                                               pos_b_embeddings,
                                               neg_a_embeddings,
                                               neg_b_embeddings,
                                               'source')

        source_labels = torch.zeros(dist_matrix.shape[0]).to(dist_matrix.device).long()
        loss = self.ce_loss(dist_matrix, source_labels)
        return loss

    def _concat_and_remove_duplicates(self,tensor1, tensor2):
        # 将两个张量拼接在一起
        concatenated_tensor = torch.cat((tensor1, tensor2))
        # 使用unique函数去掉重复元素
        unique_tensor, _ = torch.unique(concatenated_tensor, sorted=True, return_inverse=True)
        return unique_tensor


    def transfer_source_to_target(self,
                                  s_t_user_embeddings,
                                  s_t_item_embeddings,
                                  t_t_user_embeddings,
                                  t_t_item_embeddings,
                                  overlap_user_id,
                                  curve,
                                  temp):

        if self.config['u_u_cts']:
           u_u_loss=self.u_u_hyperbolic_cts_loss(s_t_user_embeddings[overlap_user_id],t_t_user_embeddings[overlap_user_id],curve,temp)
        else:
            u_u_loss=0

        # 从source的user传递给target的item
        # 输入：user_embedding,item_embedding,overlap_id,curve,temp,way
        if self.config['u_i_cts']:
           u_i_loss=self.u_i_hyperbolic_cts_loss(s_t_user_embeddings,t_t_item_embeddings,overlap_user_id,temp,'su-ti')
        else:
            u_i_loss=0

        if self.config['i_u_cts']:
           i_u_loss=self.u_i_hyperbolic_cts_loss(t_t_user_embeddings,s_t_item_embeddings,overlap_user_id,temp,'si-tu')
        else:
            i_u_loss=0

        if self.config['i_i_cts']:
            i_i_loss = self.i_i_hyperbolic_cts_loss(t_t_item_embeddings, s_t_item_embeddings, overlap_user_id, temp,'si-ti')
        else:
            i_i_loss=0
        return u_u_loss+u_i_loss+i_u_loss+i_i_loss

    def transfer_target_to_source(self,
                                  t_s_user_embeddings,
                                  t_s_item_embeddings,
                                  s_s_user_embeddings,
                                  s_s_item_embeddings,
                                  overlap_user_id ,
                                  curve,
                                  temp):
        if self.config['u_u_cts']:
           u_u_loss=self.u_u_hyperbolic_cts_loss(t_s_user_embeddings[overlap_user_id],s_s_user_embeddings[overlap_user_id],curve,temp)
        else:
            u_u_loss=0


        if self.config['u_i_cts']:
        # 从target的user传递给source的item
           u_i_loss=self.u_i_hyperbolic_cts_loss(t_s_user_embeddings,s_s_item_embeddings,overlap_user_id,temp,'tu-si')
        else:
            u_i_loss=0


        if self.config['i_u_cts']:
        # 从target的item传递给source的user
           i_u_loss=self.u_i_hyperbolic_cts_loss(s_s_user_embeddings,t_s_item_embeddings,overlap_user_id,temp,'ti-su')
        else:
            i_u_loss=0

        if self.config['i_i_cts']:
            i_i_loss = self.i_i_hyperbolic_cts_loss(s_s_item_embeddings, t_s_item_embeddings, overlap_user_id, temp,'ti-si')
        else:
            i_i_loss=0
        return u_u_loss+u_i_loss+i_u_loss+i_i_loss

    def forward(self,
                # batch_source_u，batch_target_u：是recbole中一个batch的user和item的id
                source_user_all_embeddings,
                source_item_all_embeddings,
                target_user_all_embeddings,
                target_item_all_embeddings,
                batch_source_u,
                batch_source_i,
                batch_target_u,
                batch_target_i):
        # batch_source_u，batch_target_u:输入进来一个batch的user，overlap的部分，下面将两个部分取并集，即两个domain一个batch的全部overlap user
        # 将这部分overlapped user取并集，然后去掉重复的
        overlap_user_id=self._concat_and_remove_duplicates(batch_source_u[batch_source_u<self.num_lapped_users],batch_target_u[batch_target_u<self.num_lapped_users])


        source_u_transfer = source_user_all_embeddings.clone().detach()
        source_i_transfer = source_item_all_embeddings.clone().detach()
        target_u_transfer = target_user_all_embeddings.clone().detach()
        target_i_transfer = target_item_all_embeddings.clone().detach()

        s_t_user_embeddings = self.s2t_linear(source_u_transfer)
        s_t_item_embeddings = self.s2t_linear(source_i_transfer)
        t_s_user_embeddings = self.t2s_linear(target_u_transfer)
        t_s_item_embeddings = self.t2s_linear(target_i_transfer)

        s_s_user_embeddings = self.s2s_linear(source_user_all_embeddings)
        s_s_item_embeddings = self.s2s_linear(source_item_all_embeddings)
        t_t_user_embeddings = self.t2t_linear(target_user_all_embeddings)
        t_t_item_embeddings = self.t2t_linear(target_item_all_embeddings)



        if (self.config['s_t_transfer']==True)&(self.config['t_s_transfer']==True):

           loss1 = self.transfer_source_to_target(s_t_user_embeddings,s_t_item_embeddings,t_t_user_embeddings,t_t_item_embeddings,overlap_user_id,self.target_curve,self.temp)
           loss2 = self.transfer_target_to_source(t_s_user_embeddings, t_s_item_embeddings,s_s_user_embeddings,s_s_item_embeddings,overlap_user_id ,self.source_curve,self.temp)
           loss = self.cts_lamda*(loss1+loss2)

        if (self.config['s_t_transfer'] == True) & (self.config['t_s_transfer'] == False):
            loss1 = self.transfer_source_to_target(s_t_user_embeddings, s_t_item_embeddings, t_t_user_embeddings,
                                                   t_t_item_embeddings, overlap_user_id, self.target_curve, self.temp)
            loss = self.cts_lamda * loss1

        if (self.config['s_t_transfer'] == False) & (self.config['t_s_transfer'] == True):
            loss2 = self.transfer_target_to_source(t_s_user_embeddings, t_s_item_embeddings, s_s_user_embeddings,
                                                   s_s_item_embeddings, overlap_user_id, self.source_curve, self.temp)

            loss = self.cts_lamda * loss2





        return loss


