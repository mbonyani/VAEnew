#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on February 2023
@author: Elham Sadeghi
"""
import torch
import torch.nn as nn
import math
from utils.model import Model
from utils.PositionalEncoding import PositionalEncoding
import torch.nn.functional as F

class SequenceModel(Model):
    ALPHABET = ['A','C','G','T']
    def __init__(
        self,
        n_chars = 4,
        seq_len = 12,
        batch_size = 64,
        d_model = 64,
        num_heads = 8 ,
        emb_dim = 10,
        stack = 2,
        dim_feedforward=2048,
        dropout = 0,
        
    ):
        super(SequenceModel,self).__init__()
        self.n_chars = n_chars                      #Number of DNA Bases, 4
        self.seq_len = seq_len                      #Number of bases in a sequence, 10
        self.d_model = d_model*4                      #model dimension
        self.num_heads = num_heads                  #Number of Heads for multi-head attention, it must be divisable by d_model
        self.emb_dim = emb_dim                      #Width of the embeded dimention. we need the same emb_dim as d_model
        self.stack = stack                          #number of stack
        self.dim_feedforward = dim_feedforward      #number of dim_feedforward
        self.batch_size = batch_size                #Number of sequences in a batch
        
        self.pos_encoder = PositionalEncoding(self.d_model, dropout) #positional encoding
        # self.oneHot2Dmodel = torch.nn.Linear(4, self.d_model)
        self.embedding = nn.Embedding(4, d_model)
        self.embedding1 = nn.Embedding(4, d_model)

        #The Transformer encoder
        self.encode_transformer = torch.nn.TransformerEncoderLayer(
            d_model= self.d_model,nhead= self.num_heads,dim_feedforward= self.dim_feedforward,
             dropout=dropout,batch_first=True
            )
        self.transformer_encoder = nn.TransformerEncoder(self.encode_transformer, num_layers=self.stack) #stack of the encoder of transformer
        #fully connected layer after Transformer encoder
        self.latent_linear = torch.nn.Sequential(
            torch.nn.Linear(self.d_model*self.seq_len,self.emb_dim),
            torch.nn.ReLU()
        )
        #mean and Standard deviation
        self.latent_mean = torch.nn.Linear(self.emb_dim, emb_dim)
        # self.latent_mean = torch.nn.Linear(self.emb_dim, 1)

        self.latent_log_std = torch.nn.Linear(self.emb_dim, emb_dim)
        # self.latent_log_std = torch.nn.Linear(self.emb_dim, 1)

        
        self.dec_lin = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, self.d_model*self.seq_len),
            torch.nn.ReLU()
        )

        #the Transformer decoder
        self.dec_transformer = torch.nn.TransformerDecoderLayer(
            d_model= self.d_model,nhead= self.num_heads,dim_feedforward= self.dim_feedforward,
             dropout=dropout,batch_first=True
            
            )
        self.transformer_decoder = nn.TransformerDecoder(self.dec_transformer, num_layers=self.stack) #stack of the Decoder transformer
        #fully connected after transformer decoder
        self.dec_final = torch.nn.Linear(self.d_model, n_chars)
        
    
    #endoder forward pass, takes one hot encoded sequences x and returns q(x|z)
    def encode(self, x):
        # x = x.permute(1,0,2)
        # embed_out = self.embedding(x)
        emb = self.embedding(x.view(-1, x.size(2)).long())
        emb = emb.view(x.size(0),x.size(1),self.d_model)
        pose = self.pos_encoder(emb)
        # lin = self.oneHot2Dmodel(pose)
        hidden = self.transformer_encoder(pose) #applying transformer encoder on the input which has positional encoding
        # print("HiddenNNNNNN",hidden.shape)
        hidden = self.latent_linear(torch.flatten(hidden, 1))
        # print("222HiddenNNNNNN",hidden.shape)

        z_mean = self.latent_mean(hidden)
        z_log_std = self.latent_log_std(hidden)
        return torch.distributions.Normal(loc=z_mean, scale=torch.exp(z_log_std))
    
    #decoder forward pass, takes a latent sample z and returns x^hat encoded sequences
    def decode(self, z,x):
        emb = self.embedding1(x.view(-1, x.size(2)))
        emb = emb.view(x.size(0),x.size(1),self.d_model)
        dec_input = self.pos_encoder(emb.float())[:, 1:]
        # print("DDDDDD",dec_input.shape)                

        tgt_mask = self.get_mask(x.size(1))#.to(device)
        dec_input = torch.nn.functional.pad(dec_input, (0, 0, 1, 0),"constant", -2) #padding data because of shift right in the transformer decoder;
        # print("DDDDDD",dec_input.shape)                
        # print("333HiddenNNNNNN",z.shape)

        hidden = self.dec_lin(z)
        # print("444HiddenNNNNNN",hidden.shape)

        
        hidden = hidden.view(-1,self.seq_len,self.d_model)

        hidden = self.transformer_decoder(dec_input,hidden,tgt_mask = tgt_mask)

        out = self.dec_final(hidden)
        return out
    
    #reparameterization trick for backwards pass
    def reparametrize(self, dist):
        sample = dist.rsample()
        prior = torch.distributions.Normal(torch.zeros_like(dist.loc), torch.ones_like(dist.scale))
        prior_sample = prior.sample()
        return sample, prior_sample, prior
    
    #full forward pass for entire model
    def forward(self, x):
        latent_dist = self.encode(x)   
        latent_sample, prior_sample, prior = self.reparametrize(latent_dist)    #why are we performing reparameterization on the forward pass?
        output = self.decode(latent_sample,x).view(-1,self.seq_len,self.n_chars)
        return output, latent_dist, prior, latent_sample, prior_sample
    def get_tgt_mask(self, size):
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]
        
        return mask

    def get_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def compute_selfattention(self,transformer_encoder,x,mask,src_key_padding_mask,i_layer,d_model,num_heads):
        print("FFFFFFFF",x.shape)
        h = F.linear(x.float(), transformer_encoder.layers[i_layer].self_attn.in_proj_weight.float(), bias=transformer_encoder.layers[i_layer].self_attn.in_proj_bias.float())
        qkv = h.reshape(x.shape[0], x.shape[1], num_heads, 3 * d_model//num_heads)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1) # [Batch, Head, SeqLen, d_head=d_model//num_heads]
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) # [Batch, Head, SeqLen, SeqLen]
        d_k = q.size()[-1]
        attn_probs = attn_logits / math.sqrt(d_k)
        # combining src_mask e.g. upper triangular with src_key_padding_mask e.g. columns over each padding position
        combined_mask = torch.zeros_like(attn_probs)
        if mask is not None:
            combined_mask += mask.float() # assume mask of shape (seq_len,seq_len)
        if src_key_padding_mask is not None:
            combined_mask += src_key_padding_mask.float().unsqueeze(1).unsqueeze(1).repeat(1,num_heads,x.shape[1],1)
            # assume shape (batch_size,seq_len), repeating along head and line dimensions == "column" mask
        combined_mask = torch.where(combined_mask>0,torch.zeros_like(combined_mask)-float("inf"),torch.zeros_like(combined_mask))
        # setting masked logits to -inf before softmax
        attn_probs += combined_mask
        attn_probs = F.softmax(attn_probs, dim=-1)
        return attn_logits,attn_probs

    def extract_selfattention_maps(self,x,mask,src_key_padding_mask):
        attn_logits_maps = []
        attn_probs_maps = []
        num_layers = self.transformer_encoder.num_layers
        d_model = self.transformer_encoder.layers[0].self_attn.embed_dim
        num_heads = self.transformer_encoder.layers[0].self_attn.num_heads
        norm_first = self.transformer_encoder.layers[0].norm_first
        emb = self.embedding(x.view(-1, x.size(2)).long())
        emb = emb.view(x.size(0),x.size(1),self.d_model)
        x = self.pos_encoder(emb)
        with torch.no_grad():
            for i in range(num_layers):
                # compute attention of layer i
                h = x.clone()
                if norm_first:
                    h = self.transformer_encoder.layers[i].norm1(h)
                # attn = transformer_encoder.layers[i].self_attn(h, h, h,attn_mask=mask,key_padding_mask=src_key_padding_mask,need_weights=True)[1]
                # attention_maps.append(attn) # of shape [batch_size,seq_len,seq_len]
                attn_logits,attn_probs = self.compute_selfattention(self.transformer_encoder,h,mask,src_key_padding_mask,i,d_model,num_heads)
                attn_logits_maps.append(attn_logits) # of shape [batch_size,num_heads,seq_len,seq_len]
                attn_probs_maps.append(attn_probs)
                # forward of layer i
                x = self.transformer_encoder.layers[i](x,src_mask=mask,src_key_padding_mask=src_key_padding_mask)
        return attn_logits_maps,attn_probs_maps

    

    def __repr__(self):
        return 'SequenceVAE' + self.trainer_config
    
