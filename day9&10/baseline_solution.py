import numpy as np, pandas as pd
np.random.seed(42)
tr=pd.read_csv('train.csv'); te=pd.read_csv('test.csv')
feats=[c for c in tr.columns if c not in ['id','sleep_stage']]
def prep(df):
    X=df[feats].copy()
    X['eog_missing']=X['eog_burst_index'].isna().astype(float)
    return X
Xtr=prep(tr); Xte=prep(te); y=tr['sleep_stage'].values
med=Xtr['eog_burst_index'].median()
Xtr['eog_burst_index']=Xtr['eog_burst_index'].fillna(med)
Xte['eog_burst_index']=Xte['eog_burst_index'].fillna(med)
mu=Xtr.mean(); sd=Xtr.std().replace(0,1)
Xs=((Xtr-mu)/sd).values; Xts=((Xte-mu)/sd).values
n=len(Xs); idx=np.random.permutation(n); cut=int(n*0.8)
trI,vaI=idx[:cut],idx[cut:]
def f1macro(yt,yp,K=4):
    fs=[]
    for k in range(K):
        tp=np.sum((yp==k)&(yt==k)); fp=np.sum((yp==k)&(yt!=k)); fn=np.sum((yp!=k)&(yt==k))
        p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
        fs.append(2*p*r/(p+r) if p+r else 0)
    return np.mean(fs)
# --- Softmax regression ---
def softmax_fit(X,y,K=4,lr=0.5,ep=400,l2=1e-3):
    m,d=X.shape; W=np.zeros((d,K)); b=np.zeros(K)
    Y=np.eye(K)[y]
    for _ in range(ep):
        Z=X@W+b; Z-=Z.max(1,keepdims=True); P=np.exp(Z); P/=P.sum(1,keepdims=True)
        gW=X.T@(P-Y)/m+l2*W; gb=(P-Y).mean(0)
        W-=lr*gW; b-=lr*gb
    return W,b
W,b=softmax_fit(Xs[trI],y[trI])
def softmax_pred(X,W,b):
    Z=X@W+b; return Z.argmax(1)
vp=softmax_pred(Xs[vaI],W,b); print('Softmax val macro-F1:',round(f1macro(y[vaI],vp),4))
# --- Gaussian NB ---
def gnb_fit(X,y,K=4):
    means=[];vars=[];pri=[]
    for k in range(K):
        Xk=X[y==k]; means.append(Xk.mean(0)); vars.append(Xk.var(0)+1e-6); pri.append(len(Xk)/len(X))
    return np.array(means),np.array(vars),np.log(np.array(pri))
def gnb_pred(X,means,vars,lp):
    K=len(lp); ll=np.zeros((len(X),K))
    for k in range(K):
        ll[:,k]=lp[k]-0.5*np.sum(np.log(2*np.pi*vars[k]))-0.5*np.sum((X-means[k])**2/vars[k],1)
    return ll.argmax(1)
m_,v_,lp_=gnb_fit(Xs[trI],y[trI]); gp=gnb_pred(Xs[vaI],m_,v_,lp_)
print('GNB val macro-F1:',round(f1macro(y[vaI],gp),4))
# pick best, refit on full, predict test
best='softmax' if f1macro(y[vaI],vp)>=f1macro(y[vaI],gp) else 'gnb'
print('BEST:',best)
if best=='softmax':
    W,b=softmax_fit(Xs,y); pred=softmax_pred(Xts,W,b)
else:
    m_,v_,lp_=gnb_fit(Xs,y); pred=gnb_pred(Xts,m_,v_,lp_)
sub=pd.DataFrame({'id':te['id'],'sleep_stage':pred.astype(int)})
sub.to_csv('submission.csv',index=False)
print('saved submission.csv',sub.shape); print(sub['sleep_stage'].value_counts().sort_index().to_dict())
