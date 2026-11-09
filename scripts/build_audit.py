"""Deterministically build retained audit cases from original and synthetic data."""
import json
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vectorlite.audit import audit
from vectorlite.vectors import load_vectors
root=Path(__file__).resolve().parents[1]
original,_=load_vectors(root/'examples/data/vectors.jsonl')
ring=[{'id':name,'vector':v} for name,v in [('east',[1,0]),('near',[1,.25]),('north-east',[1,1]),('north',[0,1]),('west',[-1,0]),('south',[0,-1]),('twin',[1,0]),('zero',[0,0])]]
rng=random.Random(97);cases=[]
for i in range(32):
 items=original if i%4==0 else ring if i%4 in [1,2] else [{'id':'v%02d'%j,'vector':[rng.randint(-8,8) for _ in range(5)]} for j in range(12)]
 query=items[0]['vector'] if i%4 in [0,1] else [.97,.35] if i%4==2 else [rng.randint(-8,8) for _ in range(5)]
 spec=dict(items=items,query=query,bits=[8,16,32][i%3],seed=i,k=[1,3,5][i%3])
 spec['threshold']=[spec['bits'],spec['bits']*3//4,0][i%3]
 if i%4==1:spec['exclude_ids']=[items[0]['id']]
 if i==30:spec['query']=[0,0]
 report=audit(spec)
 cases.append(dict(name=('Original8D' if i%4==0 else 'Unit-circle' if i%4 in [1,2] else 'Synthetic5D')+' / '+str(i),spec=spec,report=report))
# Explicit boundary sign and self-retrieval controls.
for name,spec in [('Boundary signs',dict(items=ring,query=[1,0],bits=8,k=3,hyperplanes=[[1,0],[0,1],[-1,0],[0,-1]]*2)),('Self included',dict(items=ring[:1]+ring[3:4],query=[1,0],bits=8,seed=0,k=1,threshold=8)),('Self excluded',dict(items=ring[:1]+ring[3:4],query=[1,0],bits=8,seed=0,k=1,threshold=8,exclude_ids=['east']))]:
 cases.append(dict(name=name,spec=spec,report=audit(spec)))
(root/'docs/audit.json').write_text(json.dumps({'cases':cases},indent=2,allow_nan=False)+'\n')
print(json.dumps({'cases':len(cases),'self_controls':[x['report']['selected']['recall'] for x in cases[-2:]],'threshold_rows':sum(len(x['report']['curve']) for x in cases)}))
