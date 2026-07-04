import os
from statistics import mean
import torch as t
from torch import nn
import torch.nn.functional as F
from config.params import args
from Utils.utils import *
import numpy as np
import scipy
import torch_sparse as ts
from data.data_handler import temHandler

init = nn.init.xavier_uniform_
uniform_init = nn.init.uniform_


import networkx as nx



class SpanningTree(nn.Module):
    def __init__(self, adj):
        super(SpanningTree, self).__init__()
        self.old_adj = adj

    def to_graph_list(self, adj):
        self.graph_list = []
        rows, cols, vals = adj.coo()    
        for i in range(rows.shape[0]):
            r = rows[i].item()
            c = cols[i].item()
            v = vals[i].item()
            if r <= c:
                self.graph_list.append((r,c,v))
        return self.graph_list

    def to_sparse_adj(self, shape, T):
        rows = []
        cols = []
        vals = []

        for tup in T:
            rows.append(tup[0])
            cols.append(tup[1])
            vals.append(tup[2]['weight'])

            rows.append(tup[1])
            cols.append(tup[0])
            vals.append(tup[2]['weight'])
        rows = t.tensor(rows)
        cols = t.tensor(cols)
        vals = t.tensor(vals)

        return ts.SparseTensor(row=rows, col=cols, value = vals, sparse_sizes= shape).cuda()

    def forward(self, adj):
        if adj == self.old_adj:
            return self.new_adj

        self.old_adj = adj
        self.to_graph_list(adj)
        G = nx.Graph()
        G.add_weighted_edges_from(self.graph_list)  
        T = nx.minimum_spanning_tree(G)    
        self.new_adj =  self.to_sparse_adj(adj.sizes(), T.edges(data=True))

        return self.new_adj



class SpAdjDropEdge(nn.Module):
    def __init__(self):
        super(SpAdjDropEdge, self).__init__()

    def forward(self, adj, keepRate):
        if keepRate == 1.0:
            return adj

        row, col, val = adj.coo()
        edgeNum = val.size()

        mask = ((t.rand(edgeNum) + keepRate).floor()).type(t.bool)
        newVals = val[mask] / keepRate  #  v1
        # newVals = val[mask]   #  v2

        newRow = row[mask]
        newCol = col[mask]

        return ts.SparseTensor(row=newRow, col=newCol, value = newVals, sparse_sizes= adj.sizes())


class HGNNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=False, act=None):
        super(HGNNLayer, self).__init__()
        # self.act = nn.LeakyReLU(negative_slope=args.leaky)
        self.W1 = nn.Parameter(t.eye(in_feat, out_feat).cuda() )
        self.bias1 = nn.Parameter(t.zeros( 1, out_feat).cuda() )

        self.W2 = nn.Parameter(t.eye(out_feat, in_feat).cuda())
        self.bias2 = nn.Parameter(t.zeros( 1, in_feat).cuda())


        if act == 'identity' or act is None:
            self.act = None
        elif act == 'leaky':
            self.act = nn.LeakyReLU(negative_slope=args.leaky)
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'relu6':
            self.act = nn.ReLU6()
        else:
            raise ValueError(f"Unsupported activation '{act}', expected one of: identity, leaky, relu, relu6")

    def forward(self, embeds):
        if self.act is None:
            return embeds @ self.W1
        out1 = self.act(  embeds @ self.W1 + self.bias1 )
        out2 = self.act(  out1 @ self.W2 + self.bias2  )
        return out2


    
    
class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()
		# self.act = nn.LeakyReLU(negative_slope=args.leaky)

	def forward(self, adj, embeds):
		# return (t.spmm(adj, embeds))
		return adj.matmul(embeds)


class GraphUnlearning(nn.Module):
    def __init__(self, handler, model, ini_embeds, fnl_embeds):
        super(GraphUnlearning, self).__init__()        
        edges_num = handler.ts_ori_adj.nnz()
        self.edge_embeds1 = nn.Parameter(t.zeros(args.user+ args.item, args.latdim).cuda())
        # self.edge_embeds2 = nn.Parameter(t.zeros(args.user+ args.item, args.latdim).cuda())

        self.mlp_layers = nn.Sequential(*[FeedForwardLayer(args.latdim, args.latdim, act=args.act) for i in range(args.layer_mlp)])
        # self.layer_norm = nn.LayerNorm(args.latdim)
        self.act = nn.LeakyReLU(negative_slope=args.leaky)

        if args.withdraw_rate_init == 1:
            self.withdraw_rate = nn.Parameter(t.ones(args.user+args.item, 1) * args.lr * 2)            
        else:
            self.withdraw_rate = nn.Parameter(t.zeros(args.user+args.item, 1) * args.lr * 10)

        self.edgeDropper = SpAdjDropEdge()
        self.gcnLayer = GCNLayer()

        self.ini_embeds = ini_embeds.detach()
        self.fnl_embeds = fnl_embeds.detach()

        self.ini_embeds.requires_grad = False
        self.fnl_embeds.requires_grad = False
        self.model  = model


        if hasattr(self.model, "uEmbeds") and  hasattr(self.model, "iEmbeds"):  
            self.model.uEmbeds.detach()
            self.model.uEmbeds.requires_grad = False    
            self.model.iEmbeds.detach()
            self.model.iEmbeds.requires_grad = False                
        else:
            self.model.ini_embeds.detach()
            self.model.ini_embeds.requires_grad = False            


    def forward(self, ori_adj, ts_pk_adj ,mask, ts_drp_adj):        
        lats = [self.edge_embeds1 ]
        gnnLats = []        
        hyperLats = []           

        for _ in range(args.gnn_layer):            
            temEmbeds = self.gcnLayer(self.edgeDropper(ts_drp_adj, 1.0), lats[-1])
            hyperemb = self.gcnLayer(self.edgeDropper(ts_drp_adj, 0.95), lats[-1])

            gnnLats.append(temEmbeds)
            hyperLats.append(hyperemb)
            lats.append(  temEmbeds )
        edge_embed = sum(lats)
                
        edges_embeddings = [edge_embed]
        for _ in range(args.unlearn_layer):
            edges_embeddings.append(ts_pk_adj.matmul(edges_embeddings[-1]))
        
        withdraw =  [ self.fnl_embeds * self.withdraw_rate ]        
        for _ in range(args.gnn_layer):
            withdraw.append(ts_drp_adj.matmul(withdraw[-1]))

        delta_emb = - args.overall_withdraw_rate* withdraw[-1] +  edges_embeddings[-1]
        
        for i, layer in enumerate(self.mlp_layers):
            delta_emb = layer(delta_emb)

        tuned_emb = self.ini_embeds + delta_emb

        return tuned_emb, gnnLats, hyperLats  

    def outforward(self, ori_adj, ts_pk_adj ,mask, ts_drp_adj):
        self.model.training = False
        tuned_emb , _ , _ = self.forward(ori_adj, ts_pk_adj ,mask, ts_drp_adj)
        out_emb = self.model.forward(ts_pk_adj, tuned_emb, keepRate=1.0)        
        usr_embeds, itm_embeds = out_emb[:2]
        
        return usr_embeds, itm_embeds
    

    def out_all_layer(self, ori_adj, ts_pk_adj ,mask, ts_drp_adj, layer=0):
        self.model.training = False
        tuned_emb, _, _ = self.forward(ori_adj, ts_pk_adj ,mask, ts_drp_adj)
        all_embs, out_emb = self.model.forward(ts_pk_adj, tuned_emb, all_layer=True)  

        if layer == -2:
            return tuned_emb[:args.user], tuned_emb[args.user:]
        elif layer == -1:
            return out_emb[:args.user], out_emb[args.user:]
        else:
            return out_emb[layer][:args.user], out_emb[layer][args.user:]
            

    def cal_loss(self, batch_data, ori_adj, ts_pk_adj , mask, ts_drp_adj, drp_edges,  pk_edges=None):
        ancs, poss, negs = list(map(lambda x: x.long(), batch_data))

        self.model.training = True
        tuned_emb,  gcnEmbedsLst, hyperEmbedsLst  = self.forward(ori_adj, ts_pk_adj, mask, ts_drp_adj)
        out_emb = self.model.forward(ts_pk_adj, tuned_emb)
        
        usr_embeds, itm_embeds = out_emb[:2]


        base_loss, loss_dict =  self.model.cal_loss(batch_data,  tuned_emb=tuned_emb , ori_adj=ori_adj, ts_pk_adj=ts_pk_adj , mask=mask, ts_drp_adj=ts_drp_adj, drp_edges=drp_edges,  pk_edges=None)

        if args.unlearn_type =='v1':
            unlearn_loss = cal_neg_aug_v1(usr_embeds[drp_edges[0]],  itm_embeds[drp_edges[1]])
        elif args.unlearn_type =='v2':
            unlearn_loss = cal_neg_aug_v2(usr_embeds[drp_edges[0]],  itm_embeds[drp_edges[1]])
        else:
            raise ValueError(f"Unknown unlearn_type '{args.unlearn_type}', expected 'v1' or 'v2'")

        tar_fnl_uEmbeds, tar_fnl_iEmbeds  = self.fnl_embeds[ :args.user].detach(), self.fnl_embeds[args.user: ].detach()
            
        if args.fineTune:
            loss_dict['unlearn_loss'] = unlearn_loss
            # loss_dict['align_loss'] = align_loss
            align_loss= cal_positive_pred_align_v2(usr_embeds[ancs], tar_fnl_uEmbeds[ancs], itm_embeds[poss], tar_fnl_iEmbeds[poss],   cal_l2_distance, temp=args.align_temp)
            loss_dict['align_loss'] =  align_loss
            loss_dict['unlearn_ssl'] = t.tensor(0.)
            return base_loss + args.unlearn_wei * unlearn_loss + args.align_wei*align_loss ,    loss_dict
            # return base_loss + args.unlearn_wei * unlearn_loss  ,    loss_dict


        

        if not args.fineTune:
            if args.align_type == 'v2':
                align_loss = cal_positive_pred_align_v2(usr_embeds[ancs], tar_fnl_uEmbeds[ancs], itm_embeds[poss], tar_fnl_iEmbeds[poss],   cal_l2_distance, temp=args.align_temp)
            elif args.align_type == 'v3':
                align_loss = cal_positive_pred_align_v3(usr_embeds[ancs], tar_fnl_uEmbeds[ancs], itm_embeds[poss], tar_fnl_iEmbeds[poss],   cal_l2_distance, temp=args.align_temp)
            else:
                raise ValueError(f"Unknown align_type '{args.align_type}', expected 'v2' or 'v3'")

        sslLoss = 0
        for i in range(args.gnn_layer):
            embeds1 = gcnEmbedsLst[i].detach()
            embeds2 = hyperEmbedsLst[i]
            sslLoss += contrastLoss(embeds1[:args.user], embeds2[:args.user], t.unique(ancs), args.hyper_temp) + contrastLoss(embeds1[args.user:], embeds2[args.user:], t.unique(poss), args.hyper_temp)
  
        loss = args.unlearn_wei * unlearn_loss  + args.align_wei*align_loss + base_loss + args.unlearn_ssl*sslLoss
        loss_dict['unlearn_loss'] = unlearn_loss
        loss_dict['align_loss'] = align_loss
        loss_dict['unlearn_ssl'] = sslLoss
                

        return loss, loss_dict

    # (self.handler.ts_ori_adj, self.handler.ts_pk_adj, self.handler.mask,  self.handler.ts_drp_adj, usrs, trn_mask)
    def full_predict(self, ori_adj, ts_pk_adj, mask, ts_drp_adj , usrs, trn_mask):

        self.model.training = False
        usr_embeds, itm_embeds = self.outforward(ori_adj, ts_pk_adj, mask, ts_drp_adj)

        pck_usr_embeds = usr_embeds[usrs]
        full_preds = pck_usr_embeds @ itm_embeds.T
        full_preds = full_preds * (1 - trn_mask) - trn_mask * 1e8
        return full_preds


class GAIEEncoder(nn.Module):
    """GNN encoder that maps the influence graph into a latent space."""
    def __init__(self, in_dim, hidden_dim, latent_dim, num_layers):
        super(GAIEEncoder, self).__init__()
        self.gcnLayers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.gcnLayers.append(nn.Linear(in_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))
        for _ in range(num_layers - 2):
            self.gcnLayers.append(nn.Linear(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, adj, node_feats):
        h = node_feats
        for layer, norm in zip(self.gcnLayers, self.norms):
            h = self.act(norm(layer(adj.matmul(h))))
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar


class GAIEDecoder(nn.Module):
    """Decodes latent Z back to reconstruct influence adjacency values."""
    def __init__(self):
        super(GAIEDecoder, self).__init__()

    def forward(self, z, drp_rows, drp_cols):
        # inner product decoder for the dropped edges
        return (z[drp_rows] * z[drp_cols]).sum(-1)


class GAIE(nn.Module):
    """Graph Autoencoder Influence Encoder for machine unlearning."""
    def __init__(self, handler, model, ini_embeds, fnl_embeds):
        super(GAIE, self).__init__()
        num_nodes = args.user + args.item
        self.node_feats = nn.Parameter(init(t.empty(num_nodes, args.latdim)))

        latent_dim = args.latdim
        hidden_dim = args.latdim
        encoder_layers = max(args.gnn_layer, 2)

        self.encoder = GAIEEncoder(args.latdim, hidden_dim, latent_dim, encoder_layers)
        self.decoder = GAIEDecoder()

        # Shift generator: maps latent Z to embedding correction
        self.shift_mlp = nn.Sequential(*[
            FeedForwardLayer(args.latdim, args.latdim, act=args.act)
            for _ in range(args.layer_mlp)
        ])

        self.edgeDropper = SpAdjDropEdge()
        self.gcnLayer = GCNLayer()

        self.ini_embeds = ini_embeds.detach()
        self.fnl_embeds = fnl_embeds.detach()
        self.ini_embeds.requires_grad = False
        self.fnl_embeds.requires_grad = False
        self.model = model

        # Freeze the pretrained recommendation model
        if hasattr(self.model, "uEmbeds") and hasattr(self.model, "iEmbeds"):
            self.model.uEmbeds.detach()
            self.model.uEmbeds.requires_grad = False
            self.model.iEmbeds.detach()
            self.model.iEmbeds.requires_grad = False
        else:
            self.model.ini_embeds.detach()
            self.model.ini_embeds.requires_grad = False

    def reparameterize(self, mu, logvar):
        if self.training:
            std = t.exp(0.5 * logvar)
            eps = t.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, ori_adj, ts_pk_adj, mask, ts_drp_adj):
        # Encode the influence graph
        mu, logvar = self.encoder(ts_drp_adj, self.node_feats)
        logvar = t.clamp(logvar, min=-20, max=20)  # prevent exp overflow in reparameterize/KL
        z = self.reparameterize(mu, logvar)

        # Shift generator (MLP maps z to embedding correction)
        delta_emb = z
        for layer in self.shift_mlp:
            delta_emb = layer(delta_emb)
        # No raw +z skip: avoids uncontrolled shift magnitudes that hurt Recall

        # Embedding correction
        tuned_emb = self.ini_embeds + delta_emb

        return tuned_emb, mu, logvar

    def outforward(self, ori_adj, ts_pk_adj, mask, ts_drp_adj):
        self.model.training = False
        tuned_emb, _, _ = self.forward(ori_adj, ts_pk_adj, mask, ts_drp_adj)
        out_emb = self.model.forward(ts_pk_adj, tuned_emb, keepRate=1.0)
        usr_embeds, itm_embeds = out_emb[:2]
        return usr_embeds, itm_embeds

    def cal_reconstruction_loss(self, mu, logvar, ts_drp_adj, drp_edges):
        z = self.reparameterize(mu, logvar)

        drp_rows = drp_edges[0]
        drp_cols = [c + args.user for c in drp_edges[1]]

        # Positive: dropped edges should reconstruct to 1
        pos_preds = self.decoder(z, drp_rows, drp_cols)
        pos_labels = t.ones_like(pos_preds)

        # Negative sampling: random non-edges
        num_neg = len(drp_rows)
        neg_rows = t.randint(0, args.user, (num_neg,)).tolist()
        neg_cols = t.randint(args.user, args.user + args.item, (num_neg,)).tolist()
        neg_preds = self.decoder(z, neg_rows, neg_cols)
        neg_labels = t.zeros_like(neg_preds)

        all_preds = t.cat([pos_preds, neg_preds])
        all_labels = t.cat([pos_labels, neg_labels])
        rec_loss = F.binary_cross_entropy_with_logits(all_preds, all_labels)

        # KL divergence
        kl_loss = -0.5 * t.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return rec_loss + 0.01 * kl_loss

    def cal_loss(self, batch_data, ori_adj, ts_pk_adj, mask, ts_drp_adj, drp_edges, pk_edges=None):
        ancs, poss, negs = list(map(lambda x: x.long(), batch_data))

        self.model.training = True
        tuned_emb, mu, logvar = self.forward(ori_adj, ts_pk_adj, mask, ts_drp_adj)
        out_emb = self.model.forward(ts_pk_adj, tuned_emb)
        usr_embeds, itm_embeds = out_emb[:2]

        # Base recommendation loss
        base_loss, loss_dict = self.model.cal_loss(
            batch_data, tuned_emb=tuned_emb, ori_adj=ori_adj,
            ts_pk_adj=ts_pk_adj, mask=mask, ts_drp_adj=ts_drp_adj,
            drp_edges=drp_edges, pk_edges=None
        )

        # Unlearning loss
        if args.unlearn_type == 'v1':
            unlearn_loss = cal_neg_aug_v1(usr_embeds[drp_edges[0]], itm_embeds[drp_edges[1]])
        elif args.unlearn_type == 'v2':
            unlearn_loss = cal_neg_aug_v2(usr_embeds[drp_edges[0]], itm_embeds[drp_edges[1]])
        else:
            raise ValueError(f"Unknown unlearn_type '{args.unlearn_type}', expected 'v1' or 'v2'")

        # Alignment loss (preserve prediction for non-deleted edges)
        tar_fnl_uEmbeds = self.fnl_embeds[:args.user].detach()
        tar_fnl_iEmbeds = self.fnl_embeds[args.user:].detach()

        if args.align_type == 'v2':
            align_loss = cal_positive_pred_align_v2(
                usr_embeds[ancs], tar_fnl_uEmbeds[ancs],
                itm_embeds[poss], tar_fnl_iEmbeds[poss],
                cal_l2_distance, temp=args.align_temp
            )
        elif args.align_type == 'v3':
            align_loss = cal_positive_pred_align_v3(
                usr_embeds[ancs], tar_fnl_uEmbeds[ancs],
                itm_embeds[poss], tar_fnl_iEmbeds[poss],
                cal_l2_distance, temp=args.align_temp
            )
        else:
            raise ValueError(f"Unknown align_type '{args.align_type}', expected 'v2' or 'v3'")

        # Reconstruction loss
        rec_loss = self.cal_reconstruction_loss(mu, logvar, ts_drp_adj, drp_edges)

        # Total: L = L_M + λ_u L_u + λ_p L_p + λ_c L_c + λ_r L_rec
        loss = (base_loss
                + args.unlearn_wei * unlearn_loss
                + args.align_wei * align_loss
                + args.rec_wei * rec_loss)

        loss_dict['unlearn_loss'] = unlearn_loss
        loss_dict['align_loss'] = align_loss
        loss_dict['rec_loss'] = rec_loss
        loss_dict['unlearn_ssl'] = t.tensor(0.)

        return loss, loss_dict

    def full_predict(self, ori_adj, ts_pk_adj, mask, ts_drp_adj, usrs, trn_mask):
        self.model.training = False
        usr_embeds, itm_embeds = self.outforward(ori_adj, ts_pk_adj, mask, ts_drp_adj)
        pck_usr_embeds = usr_embeds[usrs]
        full_preds = pck_usr_embeds @ itm_embeds.T
        full_preds = full_preds * (1 - trn_mask) - trn_mask * 1e8
        return full_preds


class GATInfluenceLayer(nn.Module):
    """Single graph attention layer over the influence graph.
    α_ij = softmax_j( a^T [W h_i || W h_j] )
    h_i' = Σ_{j∈N(i)} α_ij W h_j
    """
    def __init__(self, in_dim, out_dim, negative_slope=0.2):
        super(GATInfluenceLayer, self).__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        # attention vector a ∈ R^{2*out_dim}
        self.a = nn.Parameter(init(t.empty(2 * out_dim, 1)))
        self.leaky = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, adj, h):
        """
        adj : torch_sparse.SparseTensor  (bipartite influence graph)
        h   : (N, in_dim) node features
        """
        Wh = self.W(h)                          # (N, out_dim)
        row, col, _ = adj.coo()                 # edges

        # a^T [Wh_i || Wh_j] for every edge
        cat_ij = t.cat([Wh[row], Wh[col]], dim=-1)  # (E, 2*out_dim)
        e_ij = self.leaky(cat_ij @ self.a).squeeze(-1)  # (E,)

        # sparse softmax per row (neighbor-wise)
        e_ij = e_ij - e_ij.max()                # numerical stability
        exp_e = t.exp(e_ij)
        # sum of exp per source node
        denom = t.zeros(h.size(0), device=h.device)
        denom.scatter_add_(0, row, exp_e)
        alpha = exp_e / (denom[row] + 1e-10)    # (E,)

        # weighted aggregation
        out = t.zeros_like(Wh)
        out.scatter_add_(0, row.unsqueeze(-1).expand_as(Wh[col]),
                         alpha.unsqueeze(-1) * Wh[col])
        return out


class AIE(nn.Module):
    """Attention-Based Influence Encoder for machine unlearning.

    Pipeline:
      Deleted edges → Attention GNN over A_Δ → Weighted influence →
      Aggregation → MLP → Embedding shift ΔE
    """
    def __init__(self, handler, model, ini_embeds, fnl_embeds):
        super(AIE, self).__init__()
        num_nodes = args.user + args.item

        # Learnable input features for influence propagation
        self.node_feats = nn.Parameter(init(t.empty(num_nodes, args.latdim)))

        # Multi-layer GAT encoder on the influence graph
        self.gat_layers = nn.ModuleList()
        for _ in range(args.gnn_layer):
            self.gat_layers.append(
                GATInfluenceLayer(args.latdim, args.latdim, negative_slope=0.2)
            )
        self.layer_act = nn.ELU()

        # MLP shift generator
        self.shift_mlp = nn.Sequential(*[
            FeedForwardLayer(args.latdim, args.latdim, act=args.act)
            for _ in range(args.layer_mlp)
        ])

        self.ini_embeds = ini_embeds.detach()
        self.fnl_embeds = fnl_embeds.detach()
        self.ini_embeds.requires_grad = False
        self.fnl_embeds.requires_grad = False
        self.model = model

        # Freeze the pretrained recommendation model
        if hasattr(self.model, "uEmbeds") and hasattr(self.model, "iEmbeds"):
            self.model.uEmbeds.detach()
            self.model.uEmbeds.requires_grad = False
            self.model.iEmbeds.detach()
            self.model.iEmbeds.requires_grad = False
        else:
            self.model.ini_embeds.detach()
            self.model.ini_embeds.requires_grad = False

    def forward(self, ori_adj, ts_pk_adj, mask, ts_drp_adj):
        h = self.node_feats
        layer_outs = [h]

        for gat in self.gat_layers:
            h_new = self.layer_act(gat(ts_drp_adj, h))
            h = h + h_new  # residual connection
            layer_outs.append(h)

        # Mean aggregation across layers
        h_bar = sum(layer_outs) / len(layer_outs)

        # Shift generator
        delta_emb = h_bar
        for layer in self.shift_mlp:
            delta_emb = layer(delta_emb)

        # Embedding correction
        tuned_emb = self.ini_embeds + delta_emb

        return tuned_emb

    def outforward(self, ori_adj, ts_pk_adj, mask, ts_drp_adj):
        self.model.training = False
        tuned_emb = self.forward(ori_adj, ts_pk_adj, mask, ts_drp_adj)
        out_emb = self.model.forward(ts_pk_adj, tuned_emb, keepRate=1.0)
        usr_embeds, itm_embeds = out_emb[:2]
        return usr_embeds, itm_embeds

    def cal_loss(self, batch_data, ori_adj, ts_pk_adj, mask, ts_drp_adj, drp_edges, pk_edges=None):
        ancs, poss, negs = list(map(lambda x: x.long(), batch_data))

        self.model.training = True
        tuned_emb = self.forward(ori_adj, ts_pk_adj, mask, ts_drp_adj)
        out_emb = self.model.forward(ts_pk_adj, tuned_emb)
        usr_embeds, itm_embeds = out_emb[:2]

        # Base recommendation loss (L_M)
        base_loss, loss_dict = self.model.cal_loss(
            batch_data, tuned_emb=tuned_emb, ori_adj=ori_adj,
            ts_pk_adj=ts_pk_adj, mask=mask, ts_drp_adj=ts_drp_adj,
            drp_edges=drp_edges, pk_edges=None
        )

        # Unlearning loss (L_u)
        if args.unlearn_type == 'v1':
            unlearn_loss = cal_neg_aug_v1(usr_embeds[drp_edges[0]], itm_embeds[drp_edges[1]])
        elif args.unlearn_type == 'v2':
            unlearn_loss = cal_neg_aug_v2(usr_embeds[drp_edges[0]], itm_embeds[drp_edges[1]])
        else:
            raise ValueError(f"Unknown unlearn_type '{args.unlearn_type}', expected 'v1' or 'v2'")

        # Alignment / preservation loss (L_p)
        tar_fnl_uEmbeds = self.fnl_embeds[:args.user].detach()
        tar_fnl_iEmbeds = self.fnl_embeds[args.user:].detach()

        if args.align_type == 'v2':
            align_loss = cal_positive_pred_align_v2(
                usr_embeds[ancs], tar_fnl_uEmbeds[ancs],
                itm_embeds[poss], tar_fnl_iEmbeds[poss],
                cal_l2_distance, temp=args.align_temp
            )
        elif args.align_type == 'v3':
            align_loss = cal_positive_pred_align_v3(
                usr_embeds[ancs], tar_fnl_uEmbeds[ancs],
                itm_embeds[poss], tar_fnl_iEmbeds[poss],
                cal_l2_distance, temp=args.align_temp
            )
        else:
            raise ValueError(f"Unknown align_type '{args.align_type}', expected 'v2' or 'v3'")

        # Total: L = L_M + λ_u L_u + λ_p L_p
        loss = (base_loss
                + args.unlearn_wei * unlearn_loss
                + args.align_wei * align_loss)

        loss_dict['unlearn_loss'] = unlearn_loss
        loss_dict['align_loss'] = align_loss
        loss_dict['unlearn_ssl'] = t.tensor(0.)

        return loss, loss_dict

    def full_predict(self, ori_adj, ts_pk_adj, mask, ts_drp_adj, usrs, trn_mask):
        self.model.training = False
        usr_embeds, itm_embeds = self.outforward(ori_adj, ts_pk_adj, mask, ts_drp_adj)
        pck_usr_embeds = usr_embeds[usrs]
        full_preds = pck_usr_embeds @ itm_embeds.T
        full_preds = full_preds * (1 - trn_mask) - trn_mask * 1e8
        return full_preds


class HyperNetwork(nn.Module):
    """Generates an update weight matrix W_u from a latent vector z.
    H(z) → W_u ∈ R^{N × d}  (one weight row per node).
    Internally: z (d,) → MLP → (N * d,) → reshape (N, d).
    Because N*d can be huge, we use a chunked low-rank factorisation:
        z → hidden → W_row ∈ R^{N × r}   and   W_col ∈ R^{r × d}
        W_u = W_row @ W_col   (N, d)
    where r (= hyper_rank) is a small bottleneck.
    """
    def __init__(self, latent_dim, num_nodes, embed_dim, hyper_rank=32):
        super(HyperNetwork, self).__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.hyper_rank = hyper_rank

        # z → hidden
        self.fc1 = nn.Linear(latent_dim, latent_dim)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        # hidden → row factor  (N × r)
        self.fc_row = nn.Linear(latent_dim, num_nodes * hyper_rank)
        # hidden → col factor  (r × d)
        self.fc_col = nn.Linear(latent_dim, hyper_rank * embed_dim)

    def forward(self, z):
        """
        z : (latent_dim,)  – global latent vector (mean-pooled from GNN).
        Returns W_u : (N, d).
        """
        h = self.act(self.fc1(z))
        W_row = self.fc_row(h).view(self.num_nodes, self.hyper_rank)  # (N, r)
        W_col = self.fc_col(h).view(self.hyper_rank, self.embed_dim)  # (r, d)
        W_u = W_row @ W_col  # (N, d)
        return W_u


class HIE(nn.Module):
    """Hypernetwork-Based Influence Encoder for machine unlearning.

    Pipeline:
      Influence graph A_Δ → GNN encoder → node embeddings →
      attention-pool → latent z → Hypernetwork H(z) → W_u →
      ΔE = W_u * E → corrected embeddings
    """
    def __init__(self, handler, model, ini_embeds, fnl_embeds, hyper_rank=32):
        super(HIE, self).__init__()
        num_nodes = args.user + args.item

        # Learnable input features for influence GNN
        self.node_feats = nn.Parameter(init(t.empty(num_nodes, args.latdim)))

        # GNN layers to encode the influence graph
        self.gnn_layers = nn.ModuleList()
        for _ in range(args.gnn_layer):
            self.gnn_layers.append(nn.Linear(args.latdim, args.latdim))
        self.layer_act = nn.ELU()

        # Attention pooling: learn which nodes matter for the global latent
        self.attn_gate = nn.Sequential(
            nn.Linear(args.latdim, args.latdim // 2),
            nn.Tanh(),
            nn.Linear(args.latdim // 2, 1)
        )

        # Hypernetwork: latent z → update weight matrix W_u
        self.hypernet = HyperNetwork(args.latdim, num_nodes, args.latdim, hyper_rank)

        # Learnable residual gate to control how much shift to apply
        self.gate = nn.Parameter(t.zeros(1))

        self.ini_embeds = ini_embeds.detach()
        self.fnl_embeds = fnl_embeds.detach()
        self.ini_embeds.requires_grad = False
        self.fnl_embeds.requires_grad = False
        self.model = model

        # Freeze the pretrained recommendation model
        if hasattr(self.model, "uEmbeds") and hasattr(self.model, "iEmbeds"):
            self.model.uEmbeds.detach()
            self.model.uEmbeds.requires_grad = False
            self.model.iEmbeds.detach()
            self.model.iEmbeds.requires_grad = False
        else:
            self.model.ini_embeds.detach()
            self.model.ini_embeds.requires_grad = False

        # Store deleted-edge affected nodes for focused attention pooling.
        # Pooling from only these nodes makes z a targeted representation of the
        # deletion neighbourhood instead of a diluted whole-graph average.
        drp_u = t.as_tensor(handler.dropped_edges[0], dtype=t.long)
        drp_i = t.as_tensor(handler.dropped_edges[1], dtype=t.long) + args.user
        affected = t.cat([drp_u, drp_i]).unique()
        self.register_buffer('affected_nodes', affected)

    def forward(self, ori_adj, ts_pk_adj, mask, ts_drp_adj):
        # GNN encode the influence graph
        h = self.node_feats
        layer_outs = [h]
        for gnn in self.gnn_layers:
            h = self.layer_act(gnn(ts_drp_adj.matmul(h)))
            layer_outs.append(h)

        # Mean aggregation across layers
        h_bar = sum(layer_outs) / len(layer_outs)  # (N, d)

        # Attention pooling restricted to deleted-edge affected nodes (users + items).
        # z becomes a focused summary of what was deleted, not a diluted whole-graph mean.
        pool_h = h_bar[self.affected_nodes]           # (K, d)
        attn_scores = self.attn_gate(pool_h)           # (K, 1)
        attn_weights = t.softmax(attn_scores, dim=0)   # (K, 1)
        z = (attn_weights * pool_h).sum(dim=0)         # (d,)

        # Hypernetwork generates update weights
        W_u = self.hypernet(z)  # (N, d)

        # Embedding update with gated residual: ΔE = sigmoid(gate) * W_u ⊙ E
        delta_emb = t.sigmoid(self.gate) * W_u * self.ini_embeds

        # Corrected embeddings
        tuned_emb = self.ini_embeds + delta_emb

        return tuned_emb

    def outforward(self, ori_adj, ts_pk_adj, mask, ts_drp_adj):
        self.model.training = False
        tuned_emb = self.forward(ori_adj, ts_pk_adj, mask, ts_drp_adj)
        out_emb = self.model.forward(ts_pk_adj, tuned_emb, keepRate=1.0)
        usr_embeds, itm_embeds = out_emb[:2]
        return usr_embeds, itm_embeds

    def cal_loss(self, batch_data, ori_adj, ts_pk_adj, mask, ts_drp_adj, drp_edges, pk_edges=None):
        ancs, poss, negs = list(map(lambda x: x.long(), batch_data))

        self.model.training = True
        tuned_emb = self.forward(ori_adj, ts_pk_adj, mask, ts_drp_adj)
        out_emb = self.model.forward(ts_pk_adj, tuned_emb)
        usr_embeds, itm_embeds = out_emb[:2]

        # Base recommendation loss (L_M)
        base_loss, loss_dict = self.model.cal_loss(
            batch_data, tuned_emb=tuned_emb, ori_adj=ori_adj,
            ts_pk_adj=ts_pk_adj, mask=mask, ts_drp_adj=ts_drp_adj,
            drp_edges=drp_edges, pk_edges=None
        )

        # Unlearning loss (L_u)
        if args.unlearn_type == 'v1':
            unlearn_loss = cal_neg_aug_v1(usr_embeds[drp_edges[0]], itm_embeds[drp_edges[1]])
        elif args.unlearn_type == 'v2':
            unlearn_loss = cal_neg_aug_v2(usr_embeds[drp_edges[0]], itm_embeds[drp_edges[1]])
        else:
            raise ValueError(f"Unknown unlearn_type '{args.unlearn_type}', expected 'v1' or 'v2'")

        # Alignment / preservation loss (L_p)
        tar_fnl_uEmbeds = self.fnl_embeds[:args.user].detach()
        tar_fnl_iEmbeds = self.fnl_embeds[args.user:].detach()

        if args.align_type == 'v2':
            align_loss = cal_positive_pred_align_v2(
                usr_embeds[ancs], tar_fnl_uEmbeds[ancs],
                itm_embeds[poss], tar_fnl_iEmbeds[poss],
                cal_l2_distance, temp=args.align_temp
            )
        elif args.align_type == 'v3':
            align_loss = cal_positive_pred_align_v3(
                usr_embeds[ancs], tar_fnl_uEmbeds[ancs],
                itm_embeds[poss], tar_fnl_iEmbeds[poss],
                cal_l2_distance, temp=args.align_temp
            )
        else:
            raise ValueError(f"Unknown align_type '{args.align_type}', expected 'v2' or 'v3'")

        # Total: L = L_M + λ_u L_u + λ_p L_p
        loss = (base_loss
                + args.unlearn_wei * unlearn_loss
                + args.align_wei * align_loss)

        loss_dict['unlearn_loss'] = unlearn_loss
        loss_dict['align_loss'] = align_loss
        loss_dict['unlearn_ssl'] = t.tensor(0.)

        return loss, loss_dict

    def full_predict(self, ori_adj, ts_pk_adj, mask, ts_drp_adj, usrs, trn_mask):
        self.model.training = False
        usr_embeds, itm_embeds = self.outforward(ori_adj, ts_pk_adj, mask, ts_drp_adj)
        pck_usr_embeds = usr_embeds[usrs]
        full_preds = pck_usr_embeds @ itm_embeds.T
        full_preds = full_preds * (1 - trn_mask) - trn_mask * 1e8
        return full_preds


class CIE(nn.Module):
    """Causal Influence Encoder for machine unlearning.

    Models unlearning as a causal intervention  do(e_ij = 0).
    Learns the causal effect of deleting edges by comparing with
    counterfactual embeddings E^cf (computed from the post-deletion graph).

    Pipeline:
      Deleted edges -> Influence graph A_delta ->
      GNN Influence Encoder -> Latent Z ->
      Causal Effect Predictor (MLP) -> Predicted delta_E ->
      E_corrected = E + delta_E

    Additional losses:
      L_c       = contrastive consistency (batch nodes vs counterfactual)
      L_causal  = ||output_corrected - E^cf||^2  (causal consistency)
    """
    def __init__(self, handler, model, ini_embeds, fnl_embeds, cf_embeds):
        super(CIE, self).__init__()
        num_nodes = args.user + args.item

        # Learnable input features for influence GNN
        self.node_feats = nn.Parameter(init(t.empty(num_nodes, args.latdim)))

        # GNN layers on the influence graph with LayerNorm for stability
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        for _ in range(args.gnn_layer):
            self.gnn_layers.append(nn.Linear(args.latdim, args.latdim))
            self.gnn_norms.append(nn.LayerNorm(args.latdim))
        self.layer_act = nn.ELU()

        # Causal Effect Predictor: maps latent Z to embedding shift
        self.causal_mlp = nn.Sequential(*[
            FeedForwardLayer(args.latdim, args.latdim, act=args.act)
            for _ in range(args.layer_mlp)
        ])

        self.ini_embeds = ini_embeds.detach()
        self.fnl_embeds = fnl_embeds.detach()
        self.cf_embeds = cf_embeds.detach()   # counterfactual embeddings
        self.ini_embeds.requires_grad = False
        self.fnl_embeds.requires_grad = False
        self.cf_embeds.requires_grad = False
        self.model = model

        # Freeze the pretrained recommendation model
        if hasattr(self.model, "uEmbeds") and hasattr(self.model, "iEmbeds"):
            self.model.uEmbeds.detach()
            self.model.uEmbeds.requires_grad = False
            self.model.iEmbeds.detach()
            self.model.iEmbeds.requires_grad = False
        else:
            self.model.ini_embeds.detach()
            self.model.ini_embeds.requires_grad = False

    def forward(self, ori_adj, ts_pk_adj, mask, ts_drp_adj):
        # GNN encode the influence graph with LayerNorm
        h = self.node_feats
        layer_outs = [h]
        for gnn, norm in zip(self.gnn_layers, self.gnn_norms):
            h = self.layer_act(norm(gnn(ts_drp_adj.matmul(h))))
            layer_outs.append(h)

        # Mean aggregation across layers
        z = sum(layer_outs) / len(layer_outs)  # (N, d)

        # Causal Effect Predictor -> predicted delta_E
        delta_emb = z
        for layer in self.causal_mlp:
            delta_emb = layer(delta_emb)

        # Embedding correction
        tuned_emb = self.ini_embeds + delta_emb

        return tuned_emb

    def outforward(self, ori_adj, ts_pk_adj, mask, ts_drp_adj):
        self.model.training = False
        tuned_emb = self.forward(ori_adj, ts_pk_adj, mask, ts_drp_adj)
        out_emb = self.model.forward(ts_pk_adj, tuned_emb, keepRate=1.0)
        usr_embeds, itm_embeds = out_emb[:2]
        return usr_embeds, itm_embeds

    def cal_loss(self, batch_data, ori_adj, ts_pk_adj, mask, ts_drp_adj, drp_edges, pk_edges=None):
        ancs, poss, negs = list(map(lambda x: x.long(), batch_data))

        self.model.training = True
        tuned_emb = self.forward(ori_adj, ts_pk_adj, mask, ts_drp_adj)
        out_emb = self.model.forward(ts_pk_adj, tuned_emb)
        usr_embeds, itm_embeds = out_emb[:2]

        # Base recommendation loss (L_M)
        base_loss, loss_dict = self.model.cal_loss(
            batch_data, tuned_emb=tuned_emb, ori_adj=ori_adj,
            ts_pk_adj=ts_pk_adj, mask=mask, ts_drp_adj=ts_drp_adj,
            drp_edges=drp_edges, pk_edges=None
        )

        # Unlearning loss (L_u)
        if args.unlearn_type == 'v1':
            unlearn_loss = cal_neg_aug_v1(usr_embeds[drp_edges[0]], itm_embeds[drp_edges[1]])
        elif args.unlearn_type == 'v2':
            unlearn_loss = cal_neg_aug_v2(usr_embeds[drp_edges[0]], itm_embeds[drp_edges[1]])
        else:
            raise ValueError(f"Unknown unlearn_type '{args.unlearn_type}', expected 'v1' or 'v2'")

        # Alignment / preservation loss (L_p)
        tar_fnl_uEmbeds = self.fnl_embeds[:args.user].detach()
        tar_fnl_iEmbeds = self.fnl_embeds[args.user:].detach()

        # True causal alignment:
        # Deleted-edge nodes must align to cf_embeds (counterfactual target = do(e=0)).
        # Applying fnl_embeds alignment to those same nodes would pull them BACK toward
        # the pre-deletion state, directly fighting the causal signal.
        # Solution: align_loss applies ONLY to retained (non-deleted-edge) anchor nodes.
        drp_u = drp_edges[0]
        drp_i = drp_edges[1]
        drp_u_flag = t.zeros(args.user, dtype=t.bool, device=ancs.device)
        drp_u_flag[drp_u] = True
        retained_mask = ~drp_u_flag[ancs]   # True for anchors NOT in deleted edges
        ret_ancs = ancs[retained_mask]
        ret_poss = poss[retained_mask]

        if len(ret_ancs) > 0:
            if args.align_type == 'v2':
                align_loss = cal_positive_pred_align_v2(
                    usr_embeds[ret_ancs], tar_fnl_uEmbeds[ret_ancs],
                    itm_embeds[ret_poss], tar_fnl_iEmbeds[ret_poss],
                    cal_l2_distance, temp=args.align_temp
                )
            elif args.align_type == 'v3':
                align_loss = cal_positive_pred_align_v3(
                    usr_embeds[ret_ancs], tar_fnl_uEmbeds[ret_ancs],
                    itm_embeds[ret_poss], tar_fnl_iEmbeds[ret_poss],
                    cal_l2_distance, temp=args.align_temp
                )
            else:
                raise ValueError(f"Unknown align_type '{args.align_type}', expected 'v2' or 'v3'")
        else:
            align_loss = t.tensor(0., device=ancs.device)

        # cf_embeds = pretrained model run on post-deletion graph ts_pk_adj.
        # This is the true do(e_ij=0) counterfactual for every node.
        cf_tuned_u = self.cf_embeds[:args.user].detach()
        cf_tuned_i = self.cf_embeds[args.user:].detach()
        tuned_u = tuned_emb[:args.user]
        tuned_i = tuned_emb[args.user:]

        # Contrastive consistency: ALL batch nodes align tuned_emb to cf (valid causal target)
        contrast_loss = (
            F.mse_loss(tuned_u[ancs], cf_tuned_u[ancs])
            + F.mse_loss(tuned_i[poss], cf_tuned_i[poss])
        )

        # Causal loss (ATT estimator): specifically targets deleted-edge nodes toward cf.
        # This provides a precise embedding-space unlearning target that complements
        # unlearn_loss (which only says "decrease score", not "reach this target").
        causal_loss = (
            F.mse_loss(tuned_u[drp_u], cf_tuned_u[drp_u])
            + F.mse_loss(tuned_i[drp_i], cf_tuned_i[drp_i])
        )

        # Total: L = L_M + lambda_u L_u + lambda_p L_p + lambda_c L_c + lambda_causal L_causal
        loss = (base_loss
                + args.unlearn_wei * unlearn_loss
                + args.align_wei * align_loss
                + args.contrast_wei * contrast_loss
                + args.causal_wei * causal_loss)

        loss_dict['unlearn_loss'] = unlearn_loss
        loss_dict['align_loss'] = align_loss
        loss_dict['contrast_loss'] = contrast_loss
        loss_dict['causal_loss'] = causal_loss
        loss_dict['unlearn_ssl'] = t.tensor(0.)

        return loss, loss_dict

    def full_predict(self, ori_adj, ts_pk_adj, mask, ts_drp_adj, usrs, trn_mask):
        self.model.training = False
        usr_embeds, itm_embeds = self.outforward(ori_adj, ts_pk_adj, mask, ts_drp_adj)
        pck_usr_embeds = usr_embeds[usrs]
        full_preds = pck_usr_embeds @ itm_embeds.T
        full_preds = full_preds * (1 - trn_mask) - trn_mask * 1e8
        return full_preds


class LightGCN(nn.Module):
    def __init__(self, handler):
        super(LightGCN, self).__init__()

        # self.adj = handler.torch_adj
        self.handler = handler
        self.adj = handler.ts_ori_adj
        self.ini_embeds = nn.Parameter(init(t.empty(args.user + args.item, args.latdim)))

    def forward(self, adj, ini_embeds=None, all_layer=False, keepRate=None):
        if ini_embeds is None:
            ini_embeds = self.ini_embeds

        embedsList = [ini_embeds]
        for _ in range(args.gnn_layer):
            embedsList.append(adj.matmul(embedsList[-1]))
        embeds = sum(embedsList)

        if all_layer:
            return (embedsList, embeds)

        return embeds[:args.user], embeds[args.user:]
    
    def cal_loss(self, batch_data,  tuned_emb=None , ori_adj=None, ts_pk_adj=None , mask=None, ts_drp_adj=None, drp_edges=None,  pk_edges=None):
        ancs, poss, negs = list(map(lambda x: x.long(), batch_data))
        usr_embeds, itm_embeds = self.forward( ts_pk_adj , tuned_emb)
        bpr_loss = cal_bpr(usr_embeds[ancs], itm_embeds[poss], itm_embeds[negs]) * args.bpr_wei

        reg_loss = cal_reg(self) * args.reg        
        loss = bpr_loss + reg_loss 
    
        loss_dict = {'bpr_loss': bpr_loss, 'reg_loss': reg_loss}
        return loss, loss_dict

    def full_predict(self, usrs, trn_mask, adj):
        usr_embeds, itm_embeds = self.forward(adj)
        pck_usr_embeds = usr_embeds[usrs]
        full_preds = pck_usr_embeds @ itm_embeds.T
        full_preds = full_preds * (1 - trn_mask) - trn_mask * 1e8
        return full_preds





init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class SimGCL(nn.Module):
    def __init__(self, handler):
        super(SimGCL, self).__init__()
        
        self.adj = handler.ts_ori_adj
        self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
        self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
        self.gcnLayers = nn.Sequential(*[SimGclGCNLayer() for i in range(args.gnn_layer)])
        self.perturbGcnLayers1 = nn.Sequential(*[SimGclGCNLayer(perturb=True) for i in range(args.gnn_layer)])
        self.perturbGcnLayers2 = nn.Sequential(*[SimGclGCNLayer(perturb=True) for i in range(args.gnn_layer)])


    def getEgoEmbeds(self, adj):
        uEmbeds, iEmbeds = self.forward(adj)
        return t.concat([uEmbeds, iEmbeds], axis=0)

    def forward(self, adj, iniEmbeds=None, all_layer=False, keepRate=None):
        if iniEmbeds is None:          
            iniEmbeds = t.concat([self.uEmbeds, self.iEmbeds], axis=0)        
                                            
        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        mainEmbeds = sum(embedsLst[1:]) / len(embedsLst[1:])
        if all_layer:
            return (embedsLst,mainEmbeds)

        if self.training:
            perturbEmbedsLst1 = [iniEmbeds]
            for gcn in self.perturbGcnLayers1:
                embeds = gcn(adj, perturbEmbedsLst1[-1])
                perturbEmbedsLst1.append(embeds)
            perturbEmbeds1 = sum(perturbEmbedsLst1[1:]) / len(embedsLst[1:])

            perturbEmbedsLst2 = [iniEmbeds]
            for gcn in self.perturbGcnLayers2:
                embeds = gcn(adj, perturbEmbedsLst2[-1])
                perturbEmbedsLst2.append(embeds)
            perturbEmbeds2 = sum(perturbEmbedsLst2[1:]) / len(embedsLst[1:])

            return mainEmbeds[:args.user], mainEmbeds[args.user:], perturbEmbeds1[:args.user], perturbEmbeds1[args.user:], perturbEmbeds2[:args.user], perturbEmbeds2[args.user:]
        return mainEmbeds[:args.user], mainEmbeds[args.user:]

    def cal_loss(self, batch_data,  tuned_emb=None , ori_adj=None, ts_pk_adj=None , mask=None, ts_drp_adj=None, drp_edges=None,  pk_edges=None):
        ancs, poss, negs = list(map(lambda x: x.long(), batch_data))
        # ancs, poss, negs = tem
        ancs = ancs.long().cuda()
        poss = poss.long().cuda()
        negs = negs.long().cuda()
        self.train()
        # print("###################cal_loss self.train########################")
        # print(self.training)
        usrEmbeds, itmEmbeds, pUsrEmbeds1, pItmEmbeds1, pUsrEmbeds2, pItmEmbeds2 = self.forward(ts_pk_adj, tuned_emb )

        ancEmbeds = usrEmbeds[ancs]
        posEmbeds = itmEmbeds[poss]
        negEmbeds = itmEmbeds[negs]

        scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        bprLoss = - (scoreDiff).sigmoid().log().mean()
        if args.reg_version == 'v1':
            regLoss = SimGCL_calcRegLoss(ancEmbeds, posEmbeds) 
        elif args.reg_version == 'v2':
            regLoss = SimGCL_calcRegLoss_v2(ancEmbeds, posEmbeds) 
        else:
            regLoss = SimGCL_calcRegLoss_v3(ancEmbeds, posEmbeds)                 

        contrastLoss = (contrast(pUsrEmbeds1, pUsrEmbeds2, ancs, args.temp) + contrast(pItmEmbeds1, pItmEmbeds2, poss, args.temp)) 
        # contrastLoss = 0


        loss = args.bpr_wei * bprLoss +  args.reg * regLoss + args.ssl_reg * contrastLoss   

        loss_dict = {'bpr_loss': bprLoss, 'reg_loss': regLoss, "contrast_loss": contrastLoss}
        return loss, loss_dict                

    def full_predict(self, usrs, trn_mask, adj):
        self.training = False
        usr_embeds, itm_embeds = self.forward(adj)
        pck_usr_embeds = usr_embeds[usrs]
        full_preds = pck_usr_embeds @ itm_embeds.T
        full_preds = full_preds * (1 - trn_mask) - trn_mask * 1e8
        return full_preds


class SimGclGCNLayer(nn.Module):
    def __init__(self, perturb=False):
        super(SimGclGCNLayer, self).__init__()
        self.perturb = perturb

    def forward(self, adj, embeds):
        # ret = t.spmm(adj, embeds)
        ret = adj.matmul(embeds)
        if not self.perturb:
            return ret
        # noise = (F.normalize(t.rand(ret.shape).cuda(), p=2) * t.sign(ret)) * args.eps
        random_noise = t.rand_like(ret).cuda()
        noise = t.sign(ret) * F.normalize(random_noise, dim=-1) * args.eps
        return ret + noise


def get_shape(adj):
    if isinstance(adj, ts.SparseTensor):
        return adj.sizes()
    else:
        return adj.shape


class FeedForwardLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=False, act=None):
        super(FeedForwardLayer, self).__init__()
        self.W = nn.Parameter(init(t.empty(args.latdim, args.latdim).cuda()))
        self.bias = nn.Parameter(t.zeros( 1, args.latdim).cuda())

        
        if act == 'identity' or act is None:
            self.act = None
        elif act == 'leaky':
            self.act = nn.LeakyReLU(negative_slope=args.leaky)
        elif act == 'relu':
            self.act = nn.ReLU()
        elif act == 'relu6':
            self.act = nn.ReLU6()
        else:
            raise ValueError(f"Unsupported activation '{act}', expected one of: identity, leaky, relu, relu6")
    
    def forward(self, embeds):
        if self.act is None:
            return  embeds @ self.W 
        return (self.act(  embeds @ self.W + self.bias )) + embeds  #  residual skip (v1)
        # return self.act(  embeds @ self.W + self.bias )  #  no-skip (v2)
    

class SGL(nn.Module):
    def __init__(self):
        super(SGL, self).__init__()

        self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
        self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

        self.edgeDropper = SpAdjDropEdge()

    def getEgoEmbeds(self, adj):
        uEmbeds, iEmbeds = self.forward(adj)
        return t.concat([uEmbeds, iEmbeds], axis=0)

    def forward(self, adj, iniEmbeds=None , keepRate=args.sglkeepRate):
        if iniEmbeds is None:
            iniEmbeds = t.concat([self.uEmbeds, self.iEmbeds], axis=0)

        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        mainEmbeds = sum(embedsLst) / len(embedsLst)

        if keepRate == 1.0 or self.training == False:
            return mainEmbeds[:args.user], mainEmbeds[args.user:]

        adjView1 = self.edgeDropper(adj, keepRate)
        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adjView1, embedsLst[-1])
            embedsLst.append(embeds)
        embedsView1 = sum(embedsLst)

        adjView2 = self.edgeDropper(adj, keepRate)
        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adjView2, embedsLst[-1])
            embedsLst.append(embeds)
        embedsView2 = sum(embedsLst)
        return mainEmbeds[:args.user], mainEmbeds[args.user:], embedsView1[:args.user], embedsView1[args.user:], embedsView2[:args.user], embedsView2[args.user:]		
     
    def cal_loss(self, batch_data,  tuned_emb=None , ori_adj=None, ts_pk_adj=None , mask=None, ts_drp_adj=None, drp_edges=None,  pk_edges=None):     
        # ancs, poss, negs = batch_data
        ancs, poss, negs = list(map(lambda x: x.long(), batch_data))
        ancs = ancs.long().cuda()
        poss = poss.long().cuda()
        negs = negs.long().cuda()
        usrEmbeds, itmEmbeds, usrEmbeds1, itmEmbeds1, usrEmbeds2, itmEmbeds2 = self.forward(ts_pk_adj, iniEmbeds=tuned_emb ,keepRate = args.sglkeepRate)
        ancEmbeds = usrEmbeds[ancs]
        posEmbeds = itmEmbeds[poss]
        negEmbeds = itmEmbeds[negs]

        clLoss = (contrastLoss(usrEmbeds1, usrEmbeds2, ancs, args.sgltemp) + contrastLoss(itmEmbeds1, itmEmbeds2, poss, args.sgltemp)) * args.sgl_ssl_reg

        scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        # bprLoss = - (scoreDiff).sigmoid().log().sum()
        bprLoss = - ((scoreDiff).sigmoid() + 1e-8 ).log().mean()
        regLoss = calcRegLoss([self.uEmbeds[ancs], self.iEmbeds[poss], self.iEmbeds[negs]]) * args.reg
        # regLoss = calcRegLoss(self.model) * args.reg
        loss = bprLoss + regLoss + clLoss

        loss_dict = {'bpr_loss': bprLoss, 'reg_loss': regLoss}

        return loss, loss_dict

    def full_predict(self, usrs, trn_mask, adj):
        self.training = False
        usr_embeds, itm_embeds = self.forward(adj, keepRate=1.0)
        pck_usr_embeds = usr_embeds[usrs]
        full_preds = pck_usr_embeds @ itm_embeds.T
        full_preds = full_preds * (1 - trn_mask) - trn_mask * 1e8
        return full_preds
