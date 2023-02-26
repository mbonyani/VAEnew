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

    def __repr__(self):
        return 'SequenceVAE' + self.trainer_config
    
