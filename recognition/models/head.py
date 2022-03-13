import torch
from torch import nn
import torch.nn.functional as F

        
class CosFace(nn.Module):   
    def __init__(self, in_feature=128, out_feature=10575, s=30.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m                        
        self.weight = nn.Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label):
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        
        output = (cosine - one_hot * self.m)
        
        return output

        
class ArcFace(nn.Module):    
    def __init__(self, in_feature=128, out_feature=10575, s=32.0, m=0.50):
        super(ArcFace, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.Tensor(out_feature, in_feature))        
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label):
        # cos(theta)             
        x, weight = F.normalize(x), F.normalize(self.weight)
        cosine = F.linear(x, weight)        
        
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
                
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cosine.acos_()
        cosine[index] += m_hot
        cosine.cos_()
                
        return cosine


class UNPG(nn.Module):
    def __init__(self, s, wisk=1.0):
        super(UNPG, self).__init__()
        self.s = s
        self.wisk = wisk
        self.cross_entropy = nn.CrossEntropyLoss()
                         
    def forward(self, cosine, aux_sn, labels):                  
        # aux_sn = aux_sn.unsqueeze(0)                        
        one = torch.ones(cosine.size(0), device=cosine.device).unsqueeze(1)        
        aux_sn = one * aux_sn        
        cosine = torch.cat([cosine, aux_sn], dim=1)                      
        loss = self.cross_entropy(self.s * cosine, labels)
        return loss
   

