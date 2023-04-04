from torch import Tensor
import torch.nn.functional as f

#inspired from 
# https://medium.com/the-dl/transformers-from-scratch-in-pytorch-8777e346ca51


def scaled_dot_product_attention(query: Tensor,key:Tensor,value:Tensor) -> Tensor:
    temp = query.bmm(key.transpose(1,2))
    scale = query.size(-1) **5
    softmax = f.softmax(temp/scale,dim=1)
    return softmax