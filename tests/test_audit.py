import copy
from decimal import Decimal, localcontext
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from vectorlite import ExactIndex, LSHIndex, cosine_similarity, l2_distance, load_vectors
from vectorlite.audit import audit, admission_probability, load_spec
from vectorlite.recall import evaluate_recall

ITEMS=[{"id":"a","vector":[1,0]},{"id":"b","vector":[1,1]},
       {"id":"c","vector":[0,1]},{"id":"d","vector":[-1,0]}]
class AuditTests(unittest.TestCase):
    def spec(self,**kw):return dict(items=copy.deepcopy(ITEMS),query=[1,0],bits=8,k=2,seed=7,**kw)
    def test_scale_safe_cosine(self):
        for scale in [1e-300,1,1e300]:
            self.assertAlmostEqual(cosine_similarity([scale,scale],[scale,0]),math.sqrt(.5),14)
            self.assertAlmostEqual(cosine_similarity([scale,-scale],[scale,-scale]),1,14)
        self.assertAlmostEqual(l2_distance([0,0],[3e200,4e200])/5e200,1,14)
    def test_decimal_reference(self):
        rng=random.Random(5)
        with localcontext() as ctx:
            ctx.prec=90
            for _ in range(120):
                a=[rng.randint(-1000,1000) for _ in range(5)]
                b=[rng.randint(-1000,1000) for _ in range(5)]
                ref=Decimal(sum(x*y for x,y in zip(a,b)))/(Decimal(sum(x*x for x in a))*Decimal(sum(x*x for x in b))).sqrt()
                self.assertAlmostEqual(cosine_similarity(a,b),float(ref),14)
    def test_bad_vectors(self):
        for a,b in [([1],[1,2]),([float('nan')],[1]),([True],[1]),([],[]),([float('inf')],[1])]:
            with self.assertRaises(ValueError):cosine_similarity(a,b)
        for cls in [ExactIndex,LSHIndex]:
            for items in [[ITEMS[0],ITEMS[0]],[{"id":"a","vector":[True]}], [{"id":"a","vector":[]}]]:
                with self.assertRaises(ValueError):cls(items)
            for k in [-1,True,1.2]:
                with self.assertRaises(ValueError):cls(ITEMS).query([1,0],k)
    def test_index_owns_copies(self):
        items=copy.deepcopy(ITEMS);a=LSHIndex(items);e=ExactIndex(items)
        items[0]['vector'][0]=-1;items[0]['id']='zz'
        self.assertEqual(a.query([1,0],1)[0][0],'a');self.assertEqual(e.query([1,0],1)[0][0],'a')
    def test_configuration(self):
        for kw in [dict(num_bits=True),dict(num_bits=9.5),dict(num_bits=5000),dict(min_shared_bits=-1),dict(min_shared_bits=65),dict(seed=True),dict(hyperplanes=4),dict(hyperplanes=[[0,0]]*64)]:
            with self.assertRaises(ValueError):LSHIndex(ITEMS,**kw)
    def test_full_curve_canonical_subset_property(self):
        for seed in range(20):
            r=audit(dict(self.spec(),seed=seed))
            prev=1.0;count=len(ITEMS)
            for row in r['curve']:
                self.assertLessEqual(row['candidate_count'],count);count=row['candidate_count']
                self.assertLessEqual(row['recall'],prev);prev=row['recall']
                admitted=set(row['candidate_ids'])
                self.assertEqual(row['recall'],len(admitted & {x[0] for x in r['exact']})/len(r['exact']))
                idx=LSHIndex(ITEMS,num_bits=8,seed=seed,min_shared_bits=row['threshold'])
                self.assertEqual(row['results'],[list(x) for x in idx.query([1,0],2)])
    def test_exclusion_and_zero_k(self):
        for cls in [ExactIndex,LSHIndex]:
            self.assertNotIn('a',[x[0] for x in cls(ITEMS).query([1,0],4,exclude_ids=['a'])])
            self.assertEqual(cls(ITEMS).query([1,0],0),[])
        r=audit(self.spec(exclude_ids=['a','b','c','d']))
        self.assertEqual(r['exact'],[])
        self.assertTrue(all(x['recall']==1 and x['candidate_count']==0 for x in r['curve']))
    def test_zero_convention_and_boundary(self):
        spec=self.spec();spec['items'].append({'id':'zero','vector':[0,0]})
        spec['hyperplanes']=[[1,0],[0,1],[-1,0],[0,-1]]*2
        r=audit(spec);z=next(x for x in r['rows'] if x['id']=='zero')
        self.assertEqual(z['signature'],'11111111');self.assertIsNone(z['bit_agreement_probability'])
        self.assertEqual(r['query_signature'],'10111011')
        self.assertTrue(all(x['theory_expected_candidates'] is None for x in r['curve']))
    def test_binomial_fraction_reference(self):
        for bits in [8,16,32]:
            for p in [Fraction(0),Fraction(1,4),Fraction(1,2),Fraction(3,4),Fraction(1)]:
                for t in range(bits+1):
                    ref=sum(Fraction(math.comb(bits,j))*p**j*(1-p)**(bits-j) for j in range(t,bits+1))
                    self.assertAlmostEqual(admission_probability(bits,t,float(p)),float(ref),14)
    def test_binomial_full_enumeration(self):
        counts=[0]*9
        for signs in itertools.product([0,1],repeat=8):counts[sum(signs)]+=1
        for t in range(9):self.assertEqual(admission_probability(8,t,.5),sum(counts[t:])/256)
    def test_angle_full_circle(self):
        for theta in [0,math.pi/6,math.pi/2,math.pi]:
            agree=0
            for i in range(36000):
                angle=(i+.5)*2*math.pi/36000
                agree+=(math.cos(angle)>=0)==(math.cos(angle-theta)>=0)
            self.assertAlmostEqual(agree/36000,1-theta/math.pi,places=4)
    def test_expected_marginal_rebuilds(self):
        spec=self.spec(threshold=6);ref=audit(spec)['selected']
        counts=[];recalls=[]
        for seed in range(1200):
            idx=LSHIndex(ITEMS,num_bits=8,seed=seed,min_shared_bits=6)
            qsig=idx._signature([1,0]);counts.append(sum(8-(qsig^s).bit_count()>=6 for s in idx.signatures))
            recalls.append(len({x[0] for x in idx.query([1,0],2)} & {'a','b'})/2)
        self.assertLess(abs(sum(counts)/len(counts)-ref['theory_expected_candidates']),.04)
        self.assertLess(abs(sum(recalls)/len(recalls)-ref['theory_expected_recall']),.025)
    def test_strict_spec(self):
        for spec in [dict(self.spec(),junk=1),dict(self.spec(),k=True),dict(self.spec(),bits=256),dict(self.spec(),exclude_ids=['missing']),dict(self.spec(),query=[1,float('nan')])]:
            with self.assertRaises(ValueError):audit(spec)
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'s.json'
            for text in ['{"a":1,"a":2}','{"a":NaN}',' '*131073]:
                p.write_text(text)
                with self.assertRaises(ValueError):load_spec(p)
    def test_full_precision_recall(self):
        class Index:
            def __init__(self,out):self.out=out
            def query(self,q,k):return self.out
        r=evaluate_recall(Index([('a',1),('b',0),('c',0)]),Index([('a',1)]),[[1]],3)
        self.assertEqual(r['mean_recall'],1/3)
    def test_loader_duplicate_nonfinite(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'v.jsonl'
            for text in ['{"id":"a","vector":[1]}\n{"id":"a","vector":[2]}','{"vector":[NaN]}','{"vector":[true]}']:
                p.write_text(text)
                with self.assertRaises(ValueError):load_vectors(p)
    def test_cli_audit_and_errors(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'s.json';p.write_text(json.dumps(self.spec()))
            call=lambda *args:subprocess.run([sys.executable,'-m','vectorlite',*args],capture_output=True,text=True)
            r=call('audit','--spec',str(p));self.assertEqual(r.returncode,0,r.stderr)
            self.assertEqual(json.loads(r.stdout),audit(self.spec()))
            out=Path(t)/'out.json';r=call('audit','--spec',str(p),'--json',str(out));self.assertEqual(r.returncode,0)
            self.assertEqual(json.loads(out.read_text()),audit(self.spec()))
            p.write_text('{"items":[]}');r=call('audit','--spec',str(p));self.assertEqual(r.returncode,2);self.assertNotIn('Traceback',r.stderr)

class ProtocolTests(unittest.TestCase):
    def test_self_and_external_cli(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'items.jsonl';q=Path(t)/'q.jsonl';out=Path(t)/'out.json'
            p.write_text('{"id":"a","vector":[1,0]}\n{"id":"b","vector":[0,1]}\n')
            q.write_text('{"id":"unseen","vector":[1,1]}\n')
            call=lambda *args:subprocess.run([sys.executable,'-m','vectorlite','benchmark','--vectors',str(p),'--bits','8','--seed','0','--min-shared-bits','8','-k','1','--json',str(out),*args],capture_output=True,text=True)
            for flags,protocol,recall in [([], 'self-inclusive',1),(['--exclude-self'],'leave-one-id-out',0)]:
                r=call(*flags);self.assertEqual(r.returncode,0,r.stderr);d=json.loads(out.read_text());self.assertEqual(d['protocol'],protocol);self.assertEqual(d['mean_recall'],recall)
            r=call('--queries',str(q));self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(out.read_text())['protocol'],'external-supplied')
            self.assertEqual(call('--queries',str(q),'--exclude-self').returncode,2)
            self.assertEqual(call('--min-recall','nan').returncode,2)
    def test_bad_approximate_metric(self):
        r=subprocess.run([sys.executable,'-m','vectorlite','query','--vectors','examples/data/vectors.jsonl','--query','1,0,0,0,0,0,0,0','--approximate','--metric','l2'],capture_output=True,text=True)
        self.assertEqual(r.returncode,2);self.assertIn('cosine only',r.stderr)
    def test_binomial_guards(self):
        for x in [(8,9,.5),(129,2,.5),(8,3,float('nan')),(True,0,.5),(8,0,True),(8,-1,.5)]:
            with self.assertRaises(ValueError):admission_probability(*x)
