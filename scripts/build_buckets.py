"""Native witnesses; browser ships only signature inputs, never precomputed reports."""
from pathlib import Path
import random,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vectorlite.bucket_cli import audit
from vectorlite.lsh import LSHIndex
from vectorlite.vectors import load_vectors
P=Path(__file__).resolve().parents[1];out=[]
def add(name,codes,bits,radius,mode='auto',excluded=None,max_blocks=32,kind='constructed-signatures'):
 v={'bits':bits,'radius':radius,'signatures':[format(x,'x')for x in codes],'query':'0','exclude_indices':excluded or[],'mode':mode,'max_blocks':max_blocks};out.append({'name':name,'kind':kind,**audit(v)})
rng=random.Random(13);codes=[rng.getrandbits(24)for _ in range(512)];codes[0]=0
add('sparse-512-auto',codes,24,2);add('sparse-512-scan',codes,24,2,'scan');add('dense-duplicates-auto',[0]*512,24,2);add('dense-duplicates-forced-blocks',[0]*512,24,2,'blocks')
add('false-candidate-boundary',[0,1,8,64,73,11,255],8,2,'blocks');add('radius-covers-all',[0,1,255],8,8);add('block-cap',[0,1,1<<127],128,64);add('exclude-duplicate-id-positions',[0,0,1,2],8,0,'blocks',[0]);add('empty-table',[],8,1)
items,_=load_vectors(P/'examples/data/vectors.jsonl');idx=LSHIndex(items,24,17,22);q=idx._signature(items[0]['vector']);add('original-corpus-query-relative',[s^q for s in idx.signatures],24,2,kind='original12-vector-relative-signatures')
for j in range(24):
 bits=rng.randint(1,128);n=rng.randint(0,80);r=rng.randrange(bits+1);add('seeded-'+str(j),[rng.getrandbits(bits)for _ in range(n)],bits,r,['auto','blocks','scan'][j%3],rng.sample(range(n),rng.randint(0,n)),rng.randint(1,64),'seeded-signature-table')
(P/'examples/buckets.json').write_text(json.dumps(out,separators=(',',':'),sort_keys=True)+'\n');(P/'docs/buckets-data.js').write_text('window.BUCKET_INPUTS='+json.dumps([{k:v for k,v in w.items()if k!='report'}for w in out],separators=(',',':'))+';\n')
print(len(out),'full bucket reports',sum(len(w['report']['checks'])for w in out),'full Hamming checks')
