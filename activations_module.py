from __future__ import annotations
import random,math,torch,numpy as np,matplotlib.pyplot as plt
import fastcore.all as fc
from functools import partial
from n_framework import *
import torch.nn.functional as F,matplotlib as mpl
from pathlib import Path
from operator import attrgetter,itemgetter
from contextlib import contextmanager
import sys,gc,traceback
from torch import tensor,nn,optim
import torchvision.transforms.functional as TF
from datasets import load_dataset

from fastcore.test import test_close

import pickle,gzip,math,os,time,shutil,torch,matplotlib as mpl,numpy as np,matplotlib.pyplot as plt
import sys,gc,traceback
from collections.abc import Mapping
from copy import copy
from torch.utils.data import DataLoader,default_collate
from torch.nn import init
from torcheval.metrics import MulticlassAccuracy
from datasets import load_dataset,load_dataset_builder


torch.set_printoptions(precision=2, linewidth=140, sci_mode=False)
mpl.rcParams['figure.constrained_layout.use'] = True

import logging
logging.disable(logging.WARNING)
     

class Hook():
    def __init__(self, m, f): self.hook = m.register_forward_hook(partial(f, self))
    def remove(self): self.hook.remove()
    def __del__(self): self.remove()
   
class Hooks(list):
    def __init__(self, ms, f): super().__init__([Hook(m, f) for m in ms])
    def __enter__(self, *args): return self
    def __exit__ (self, *args): self.remove()
    def __del__(self): self.remove()
    def __delitem__(self, i):
        self[i].remove()
        super().__delitem__(i)
    def remove(self):
        for h in self: h.remove()

class HooksCallback(callback):
    def __init__(self, hookfunc, mod_filter=fc.noop, on_train=True, on_valid=False, mods=None):
        fc.store_attr()
        super().__init__()
    
    def before_fit(self, learn):
        if self.mods: mods=self.mods
        else: mods = fc.filter_ex(learn.model.modules(), self.mod_filter)
        self.hooks = Hooks(mods, partial(self._hookfunc, learn))

    def _hookfunc(self, learn, *args, **kwargs):
        if (self.on_train and learn.training) or (self.on_valid and not learn.training): self.hookfunc(*args, **kwargs)

    def after_fit(self, learn): self.hooks.remove()
    def __iter__(self): return iter(self.hooks)
    def __len__(self): return len(self.hooks)
     

def append_stats(hook, mod, inp, outp):
    if not hasattr(hook,'stats'): hook.stats = ([],[],[])
    acts = to_cpu(outp)
    hook.stats[0].append(acts.mean())
    hook.stats[1].append(acts.std())
    hook.stats[2].append(acts.abs().histc(40,0,10))
     
def get_hist(h): return torch.stack(h.stats[2]).t().float().log1p()
 
def get_min(h):
    h1 = torch.stack(h.stats[2]).t().float()
    return h1[0]/h1.sum(0)
  
class ActivationStats(HooksCallback):
    def __init__(self, mod_filter=fc.noop): super().__init__(append_stats, mod_filter)

    def color_dim(self, figsize=(11,5)):
        fig,axes = get_grid(len(self), figsize=figsize)
        for ax,h in zip(axes.flat, self):
            show_image(get_hist(h), ax, origin='lower')

    def dead_chart(self, figsize=(11,5)):
        fig,axes = get_grid(len(self), figsize=figsize)
        for ax,h in zip(axes.flatten(), self):
            ax.plot(get_min(h))
            ax.set_ylim(0,1)

    def plot_stats(self, figsize=(10,4)):
        fig,axs = plt.subplots(1,2, figsize=figsize)
        for h in self:
            for i in 0,1: axs[i].plot(h.stats[i])
        axs[0].set_title('Means')
        axs[1].set_title('Stdevs')
        plt.legend(fc.L.range(self))
     
#|export
def clean_mem():
    clean_tb()
    clean_ipython_hist()
    gc.collect()
    torch.cuda.empty_cache()
     
def clean_ipython_hist():
    # Code in this function mainly copied from IPython source
    if not 'get_ipython' in globals(): return
    ip = get_ipython()
    user_ns = ip.user_ns
    ip.displayhook.flush()
    pc = ip.displayhook.prompt_count + 1
    for n in range(1, pc): user_ns.pop('_i'+repr(n),None)
    user_ns.update(dict(_i='',_ii='',_iii=''))
    hm = ip.history_manager
    hm.input_hist_parsed[:] = [''] * pc
    hm.input_hist_raw[:] = [''] * pc
    hm._i = hm._ii = hm._iii = hm._i00 =  ''

def clean_tb():
   
    if hasattr(sys, 'last_traceback'):
        traceback.clear_frames(sys.last_traceback)
        delattr(sys, 'last_traceback')
    if hasattr(sys, 'last_type'): delattr(sys, 'last_type')
    if hasattr(sys, 'last_value'): delattr(sys, 'last_value')

class BatchTransformCB(callback):
    def __init__(self,tfm): self.tfm=tfm
    def before_batch(self,learn): learn.batch = self.tfm(learn.batch)

class GeneralReLU(nn.Module):
    def __init__(self,leak=None, sub=None, maxv=None):
        super().__init__()
        self.leak,self.sub,self.maxv = leak,sub,maxv
        
    def forward(self,x):
        x = F.leaky_relu(x,self.leak) if self.leak is not None else F.relu(x)
        if self.sub is not None: x -= self.sub
        if self.maxv is not None: x.clamp_max(self.maxv)
        return x
    
def plot_func(f, start=-5., end=5.,steps=100):
    x=torch.linspace(start,end,steps)
    plt.plot(x,f(x))
    plt.grid(True, which='both',ls = '--')
    plt.axhline(y=0,color='k',linewidth=0.7)
    plt.axvline(x=0,color='k', linewidth=0.7)

def init_weights(m,leaky=0.):
    if isinstance(m,(nn.Conv1d,nn.Conv2d,nn.Conv3d,nn.Linear)): init.kaiming_normal_(m.weight,a=leaky)

def _lsuv_stats(hook,mod,inp,outp):
    acts = to_cpu(outp)
    hook.mean = acts.mean()
    hook.std = acts.std()
    
def lsuv_init(m,m_in,xb):
    h = Hook(m,_lsuv_stats)
    with torch.no_grad():
        while model(xb) is not None and (abs(h.std -1)>1e-3 or abs(h.mean)>1e-3):
            #print('before: ',h.mean,h.std)
            m_in.bias -=h.mean
            m_in.weight.data/=h.std
        #print('after:',h.mean,h.std)
    h.remove()

class LayerNorm(nn.Module):
    def __init__(self,dummy, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mult = nn.Parameter(tensor(1.))
        self.add  = nn.Parameter(tensor(0.))
    def forward(self,x):
        m = x.mean((1,2,3),keepdim=True)
        v = x.var ((1,2,3), keepdim=True)
        x = (x-m)/((v+self.eps).sqrt())
        return x*self.mult + self.add
    
# def conv(ni,nf,ks=3,stride=2,act=nn.ReLU, norm = None, bias=True):
#     layers = [nn.Conv2d(ni,nf,stride=stride, kernel_size=ks,padding=ks//2, bias=bias)]
#     if norm: layers.append(norm(nf))
#     if act: layers.append(act())
#     return nn.Sequential(*layers)
def conv(ni,nf,ks=3,stride=2,act=nn.ReLU, norm = None, bias=None):
    layers = [nn.Conv2d(ni,nf,stride=stride, kernel_size=ks,padding=ks//2, bias=bias)]
    if norm: layers.append(norm(nf))
    if act: layers.append(act())
    return nn.Sequential(*layers)


# def get_model(act=nn.ReLU,nfs=None,norm=None):
#     if nfs is None: nfs = [1,8,16,32,64]
#     layers =[conv(nfs[i],nfs[i+1],act=act,norm=norm) for i in range(len(nfs)-1)]   
#     return nn.Sequential(*layers,conv(nfs[-1],10, act=None,norm=None,bias=False),nn.Flatten()).to(def_device)
def get_model(act=nn.ReLU,nfs=None,norm=None):
    if nfs is None: nfs = [1,8,16,32,64]
    layers =[conv(nfs[i],nfs[i+1],act=act,norm=norm) for i in range(len(nfs)-1)]   
    return nn.Sequential(*layers,conv(nfs[-1],10, act=None,norm=False,bias=True),nn.Flatten()).to(def_device)

class BaseschedulerCB(callback):
    def __init__(self,sched): self.sched = sched
    def before_fit(self,learn): self.schedo = self.sched(learn.opt)
    def step(self,learn):
        if learn.training: self.schedo.step()

class BatchschedCB(BaseschedulerCB):
    def after_batch(self,learn): self.step(learn)

class HasLearnCB(callback):
    def before_fit(self,learn): self.learn = learn
    def after_fit(self,learn): self.learn = None

class RecorderCB(callback):
    def __init__(self, **d): self.d = d
    def before_fit(self,learn):
            self.recs = {k:[] for k in self.d}
            self.pg = learn.opt.param_groups[0]
            
    def after_batch(self,learn):
        if not learn.training: return
        for k,v in self.d.items():
            self.recs[k].append(v(self))
    def plot(self):
        for k,v in self.recs.items():
            plt.plot(v,label=k)
            plt.legend()
            plt.show()

class EpochSchedCB(BaseschedulerCB):
    def after_epoch(self,learn): self.step(learn)
