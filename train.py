import torch
import os
import random

import numpy as np
import argparse

from data import MovieLensDataset
from model import SASRec

def set_seed(seed:int)-> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manula_seed_all(seed)


def sample_negative_items(pos_items:torch.tensor,item_num:int,device:torch.device)->torch.tensor:
    neg_items=torch.randint(1,item_num+1,size=pos_items.shape,device=device)
    neg_items[pos_items==0]=0 ## Keep padding aligned: if target position is padding (0), set negative to 0 too.
    return neg_items


def parse_args():
    parser=argparse.ArgumentParser("Train SASRec on MovieLens 1M dataset")

    parser.add_argument("--movies_file", type=str, default="movies.dat")
    parser.add_argument("--ratings_file", type=str, default="ratings.dat")
    
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--ckpt_name", type=str, default="sasrec_best.pt")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=5)
    
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--maxlen", type=int, default=50)
    parser.add_argument("--hidden_units", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    
    return parser.parse_args()

def train(args):
    set_seed(args.seed)

    #load data
    dataset=MovieLensDataset(
        movies_file=args.movies_file,
        rating_file=args.rating_file,
        maxlen=args.maxlen,
    )



    train_loader=dataset.get_loader("train",args.batch_size)
    val_loader=dataset.get_loader("val",args.batch_size)


    #number of items(excluding pading id 0)
    item_num=int(dataset.movies['MovieID'].max()) #larget movie id


    #build a model
    model=SASRec(
        item_num=item_num,
        maxlen=args.maxlen,
        hidden_units=args.hidden_units,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        lr=args.lr,

    )

    device=model.device
    _=(train_loader,val_loader,device)


#step7: add checkpoint +early stoping tracker

#prepare checpoint folder/file
os.makedirs(args.ckpt_dir,exist_ok=True)
ckpt_path=os.path.join(args.ckpt_dir,args.ckpt_name)

# track best validation NDCG@10
best_val_ncdg10=-1.0
wait=0


#step  train loop
for  epoch in range(1,args.epochs+1):
    model.train()
    epoch_loss=0.0
    num_batches=0

    for




    


    