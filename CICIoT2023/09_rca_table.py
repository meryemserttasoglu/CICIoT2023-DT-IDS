import pickle, joblib, numpy as np
import shap
d=pickle.load(open("data/processed/preprocessed.pkl","rb"))
Xte=d["X_test"]; yte=d["y_test_category"]; feats=list(d["feature_cols"]); le=d["le_category"]
classes=list(le.classes_)
rf=joblib.load("models/rf_multiclass.joblib")
rng=np.random.RandomState(42)
# sınıf-dengeli örnek: her sınıftan en çok 400
idx=[]
for c in range(len(classes)):
    ci=np.where(yte==c)[0]
    if len(ci)>0: idx.extend(rng.choice(ci,size=min(400,len(ci)),replace=False))
idx=np.array(idx); Xs=Xte[idx]; ys=yte[idx]
print("SHAP örnek:",Xs.shape)
expl=shap.TreeExplainer(rf)
sv=expl.shap_values(Xs)
# format normalize -> list[class] of (n,feat)
if isinstance(sv,list):
    svc=sv
else:  # (n,feat,class)
    svc=[sv[:,:,c] for c in range(sv.shape[2])]
benign=[i for i,c in enumerate(classes) if c in ("Benign","BenignTraffic")][0]
trname={"BruteForce":"Kaba-Kuvvet","DDoS":"DDoS","DoS":"DoS","Mirai":"Mirai","Recon":"Keşif","Spoofing":"Kimlik Sahtekarlığı","Web":"Web Tabanlı"}
rows=[]
for c in range(len(classes)):
    if c==benign: continue
    m=(ys==c)
    if m.sum()==0: continue
    imp=np.abs(svc[c][m]).mean(axis=0)
    top=np.argsort(-imp)[:4]
    rows.append((trname.get(classes[c],classes[c]), [feats[t] for t in top]))
for name,tf in rows: print(f"{name:18s}: {', '.join(tf)}")
pickle.dump(rows, open("results/09_rca_table.pkl","wb"))
print("kaydedildi.")
