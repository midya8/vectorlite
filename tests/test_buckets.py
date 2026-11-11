import copy,itertools,json,random,subprocess,sys,tempfile,unittest
from pathlib import Path
from vectorlite.buckets import HammingBuckets,BucketLSHIndex
from vectorlite.bucket_cli import audit,parse
from vectorlite.lsh import LSHIndex
class BucketTests(unittest.TestCase):
 def test_exhaustive_complete_cube_all_radii_queries_modes(self):
  comparisons=0
  for bits in range(1,8):
   codes=list(range(1<<bits))
   for radius in range(bits+1):
    h=HammingBuckets(codes,bits,radius)
    for query in codes:
     expected=[i for i,c in enumerate(codes)if sum(a!=b for a,b in zip(f'{query:0{bits}b}',f'{c:0{bits}b}'))<=radius]
     for mode in ('auto','blocks','scan'):
      r=h.lookup(query,mode=mode);self.assertEqual(r['admitted_indices'],expected);comparisons+=1
  print('complete cube query/radius/mode comparisons',comparisons)
 def test_random_codes_exclusions_and_storage_conservation(self):
  rng=random.Random(793);count=0
  for _ in range(160):
   bits=rng.randint(1,128);n=rng.randint(0,100);codes=[rng.getrandbits(bits)for _ in range(n)];radius=rng.randrange(bits+1);ex=sorted(rng.sample(range(n),rng.randint(0,n)));h=HammingBuckets(codes,bits,radius,max_blocks=rng.randint(1,64));query=rng.getrandbits(bits)
   expected=[i for i,c in enumerate(codes)if i not in ex and sum((query>>j&1)!=(c>>j&1)for j in range(bits))<=radius]
   for mode in ('auto','blocks','scan'):
    r=h.lookup(query,exclude_indices=ex,mode=mode);self.assertEqual(r['admitted_indices'],expected);self.assertEqual(r['work']['full_hamming_checks'],len(r['checks']));self.assertEqual(r['construction']['posting_references'],n*len(r['construction']['blocks']));count+=1
  print('random excluded-query mode comparisons',count)
 def test_sparse_and_dense_cost_witness(self):
  rng=random.Random(13);codes=[rng.getrandbits(24)for _ in range(512)];codes[0]=0;h=HammingBuckets(codes,24,2);a=h.lookup(0);b=h.lookup(0,mode='scan');self.assertEqual(a['strategy'],'blocks');self.assertLess(a['work']['full_hamming_checks'],512);self.assertEqual(a['admitted_indices'],b['admitted_indices'])
  dense=HammingBuckets([0]*512,24,2);r=dense.lookup(0);self.assertEqual(r['fallback_reason'],'posting-cost');self.assertEqual(r['work']['posting_references_visited'],0);self.assertEqual(r['work']['posting_lengths_total'],1536);r=dense.lookup(0,mode='blocks');self.assertEqual(r['work']['posting_references_visited'],1536);self.assertEqual(r['work']['unique_posting_candidates'],512)
 def test_lsh_exact_output_and_detached_inputs(self):
  rng=random.Random(831);comparisons=0
  for trial in range(20):
   items=[{'id':f'i{i:03d}','vector':[rng.uniform(-1,1)for _ in range(5)]}for i in range(30)];query=[rng.uniform(-1,1)for _ in range(5)];items[1]['vector']=list(items[0]['vector']);items[2]['vector']=[0]*5
   for threshold in (0,1,4,6,7,8):
    scan=LSHIndex(items,8,7,threshold);bucket=BucketLSHIndex(items,8,7,threshold);expected=scan.query(query,k=17,exclude_ids=['i000','unknown'])
    for mode in ('auto','blocks','scan'):self.assertEqual(bucket.query(query,17,exclude_ids=['i000','unknown'],mode=mode),expected);comparisons+=1
  source=[{'id':'a','vector':[1,0]},{'id':'b','vector':[0,1]}];h=BucketLSHIndex(source,8);before=h.query([1,0]);source[0]['vector'][0]=100;source[0]['id']='changed';self.assertEqual(h.query([1,0]),before);self.assertEqual(h.query([1,0],0),[])
  print('original scan versus bucket cosine/id rankings',comparisons)
 def test_partition_boundary_and_false_candidates(self):
  # bits8 radius2 partitions3,3,2; one mismatch per block is a false candidate only if a block still matches.
  h=HammingBuckets([0,1,8,64,1|8|64,1|2|8,255],8,2);r=h.lookup(0,mode='blocks');self.assertEqual(r['admitted_indices'],[0,1,2,3]);self.assertIn(5,[x['index']for x in r['checks']]);self.assertNotIn(4,[x['index']for x in r['checks']]);self.assertEqual([(x['offset'],x['length'])for x in r['construction']['blocks']],[(0,3),(3,3),(6,2)])
 def test_empty_radius_all_and_cap(self):
  self.assertEqual(HammingBuckets([],8,1).lookup(0)['admitted_indices'],[])
  r=HammingBuckets([0,255],8,8).lookup(0);self.assertEqual(r['fallback_reason'],'radius-covers-all');self.assertEqual(r['admitted_indices'],[0,1]);self.assertEqual(r['construction']['posting_references'],0)
  self.assertEqual(HammingBuckets([0],128,64).lookup(0)['fallback_reason'],'block-cap')
 def test_immutable_signature_snapshot_and_detached_reports(self):
  codes=[0,1];h=HammingBuckets(codes,8,0);codes[0]=255;self.assertEqual(h.signatures,(0,1));r=h.lookup(0);r['construction']['blocks'][0]['length']=1;r['admitted_indices'].clear();self.assertEqual(h.lookup(0)['admitted_indices'],[0])
 def test_strict_bounds_and_cli(self):
  for args in [([True],8,0),([-1],8,0),([256],8,0),([0],True,0),([0],8,True),([0],8,9)]:
   with self.assertRaises(ValueError):HammingBuckets(*args)
  h=HammingBuckets([0],8,0)
  for kwargs in [{'exclude_indices':[True]},{'exclude_indices':[1]},{'exclude_indices':[0,0]},{'mode':'bad'}]:
   with self.assertRaises(ValueError):h.lookup(0,**kwargs)
  v={'bits':8,'radius':1,'signatures':['0','ff'],'query':'0','exclude_indices':[],'mode':'auto','max_blocks':32}
  self.assertEqual(parse(json.dumps(v)),v)
  for text in ['{"bits":1,"bits":2}','['*1000,' '.ljust(131073)]:
   with self.assertRaises(ValueError):parse(text)
  for change in [{'query':'-1'},{'signatures':['FF']},{'bits':129},{'extra':1},{'exclude_indices':True}]:
   with self.assertRaises(ValueError):audit({**v,**change})
  with tempfile.TemporaryDirectory()as td:
   p=Path(td)/'v.json';p.write_text(json.dumps(v));r=subprocess.run([sys.executable,'-m','vectorlite.bucket_cli','--input',str(p)],capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout),audit(v));p.write_text('{}');r=subprocess.run([sys.executable,'-m','vectorlite.bucket_cli','--input',str(p)],capture_output=True,text=True);self.assertEqual(r.returncode,2)

 def test_all_legacy_specs_and_new_full_reports(self):
  root=Path(__file__).resolve().parents[1];legacy=json.loads((root/'docs/audit.json').read_text())['cases'];n=0
  for w in legacy:
   v=w['spec'];scan=LSHIndex(v['items'],v.get('bits',16),v.get('seed',17),v.get('threshold'),hyperplanes=v.get('hyperplanes'));h=BucketLSHIndex(v['items'],v.get('bits',16),v.get('seed',17),v.get('threshold'),hyperplanes=v.get('hyperplanes'))
   for mode in ('auto','blocks','scan'):self.assertEqual(h.query(v['query'],v.get('k',3),exclude_ids=v.get('exclude_ids',[]),mode=mode),scan.query(v['query'],v.get('k',3),exclude_ids=v.get('exclude_ids',[])));n+=1
  ws=json.loads((root/'examples/buckets.json').read_text());self.assertEqual(len(ws),34)
  for w in ws:self.assertEqual(audit(w['input']),{'input':w['input'],'report':w['report']})
  print('all legacy complete-spec mode comparisons',n,'new full reports',len(ws))
