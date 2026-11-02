#!/usr/bin/env python3
"""One row per ComfyUI node from a tools/comfy-trace.sh pair: BARE's true wall time beside TRACED's GPU busy, copies, CPU seconds, disk reads, dispatches
and top Python stacks. Tracing costs wall time, not kernel time: subtract across the two runs for the GPU's idle share. usage: comfy_join.py BARE TRACED"""
import collections
import csv
import glob
import sys

import numpy as np

B0, R = sys.argv[1:3]
f=glob.glob(R+'/att/hercule/*_kernel_trace.csv')[0]
names=[];ix={};S=[];E=[];K=[];D=[]
for x in csv.DictReader(open(f)):
    n=x['Kernel_Name']
    if n not in ix: ix[n]=len(names); names.append(n)
    S.append(int(x['Start_Timestamp']));E.append(int(x['End_Timestamp']));K.append(ix[n]);D.append(int(x['Dispatch_Id']))
S=np.array(S,dtype=np.int64);E=np.array(E,dtype=np.int64);K=np.array(K);D=np.array(D,dtype=np.int64)
o=np.argsort(S);S,E,K,D=S[o],E[o],K[o],D[o]
m=np.maximum.accumulate(E);br=np.nonzero(S[1:]>m[:-1])[0]+1
A=np.concatenate(([S[0]],S[br]));B=np.concatenate((m[br-1],[m[-1]]))
busy=lambda a,b:(np.clip(B,a,b).sum()-np.clip(A,a,b).sum())/1e9
def nodes(p):
    ev=[l.rstrip('\n').split('\t') for l in open(p)]
    x=[(int(e[0]),e[2],e[3]) for e in ev if e[1] in('executing','execution_success')]
    return [(a[0],b[0],a[2],a[1]) for a,b in zip(x,x[1:]) if a[1]]
bare={n:(b-a)/1e9 for a,b,c,n in nodes(B0+'/stages.tsv')}
cpu=[l.split('\t') for l in open(R+'/cpu.tsv').read().splitlines()]
ct=np.array([int(c[0]) for c in cpu]);tk=np.array([int(c[1]) for c in cpu]);rd=np.array([int(c[3] or 0) for c in cpu]);vr=np.array([int(c[5] or 0) for c in cpu]);mc=np.array([int(c[8] or 0) for c in cpu]);sc=np.array([int(c[7] or 0) for c in cpu]);run=np.array([int(c[2]) for c in cpu])
fr=[l.rstrip('\n').split('\t') for l in open(R+'/frames.tsv')]
ft=np.array([int(a) for a,_ in fr]);fs=[b for _,b in fr]
copy=glob.glob(R+'/att/hercule/*memory_copy_trace.csv')
r=list(csv.DictReader(open(copy[0]))) if copy else []
cS=np.array([int(x['Start_Timestamp']) for x in r],dtype=np.int64);cE=np.array([int(x['End_Timestamp']) for x in r],dtype=np.int64)
print('copies',len(r),'s',(cE-cS).sum()/1e9)
def fn(s):
    p=[q.split(':') for q in s.split(' < ')]
    return ' < '.join(f"{a[1]}" for a in p[:3])
rows=[]
for a,b,c,n in nodes(R+'/stages.tsv'):
    w=(b-a)/1e9; bw=bare.get(n,0)
    if max(w,bw)<0.3: continue
    i=np.searchsorted(ct,a);j=np.searchsorted(ct,b)-1
    cs=(tk[j]-tk[i])/100 if j>i else 0
    gb=(rd[j]-rd[i])/1e9 if j>i else 0
    ka,kb=np.searchsorted(S,a),np.searchsorted(S,b)
    q=np.searchsorted(ft,a),np.searchsorted(ft,b)
    top=collections.Counter(fn(s) for s in fs[q[0]:q[1]]).most_common(2)
    cp=((np.minimum(cE,b)-np.maximum(cS,a)).clip(0)).sum()/1e9 if len(cS) else 0
    print(f"{c[:26]:26s} {n:>3s} bare {bw:6.2f} traced {w:6.2f} busy {busy(a,b):6.2f} copy {cp:5.2f} cpu {cs:6.1f}s read {gb:5.1f}GB disp {kb-ka:8d} mclk {int(mc[i:j+1].max()) if j>=i else 0} | {top[:2]}")
    rows.append((c,n,bw,w,busy(a,b),cs))
print('sum bare',sum(bare.values()),'busy',sum(q[4] for q in rows),'cpu',sum(q[5] for q in rows))
