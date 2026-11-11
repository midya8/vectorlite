"""Strict bounded hexadecimal signature-table audit."""
import argparse,json,sys
from .buckets import HammingBuckets

def parse(text):
    if len(text.encode())>131072:raise ValueError('input exceeds128KiB')
    def pairs(xs):
        d={}
        for k,v in xs:
            if k in d:raise ValueError('duplicate JSON key')
            d[k]=v
        return d
    try:v=json.loads(text,object_pairs_hook=pairs)
    except(RecursionError,json.JSONDecodeError)as e:raise ValueError('invalid JSON')from e
    if type(v)is not dict or set(v)!={'bits','radius','signatures','query','exclude_indices','mode','max_blocks'}:raise ValueError('expected exact documented keys')
    if type(v['bits'])is not int or not 1<=v['bits']<=128 or type(v['signatures'])is not list or len(v['signatures'])>1024:raise ValueError('bounded audit requires bits1..128, items0..1024')
    import re
    def code(x):
        if type(x)is not str or not re.fullmatch('[0-9a-f]{1,32}',x)or int(x,16)>=1<<v['bits']:raise ValueError('bounded lowercase hexadecimal signature required')
        return int(x,16)
    signatures=[code(x)for x in v['signatures']];query=code(v['query'])
    if type(v['exclude_indices'])is not list:raise ValueError('excluded indices must be a list')
    h=HammingBuckets(signatures,v['bits'],v['radius'],max_blocks=v['max_blocks']);h.lookup(query,exclude_indices=v['exclude_indices'],mode=v['mode']);return v

def audit(v):
    v=parse(json.dumps(v));h=HammingBuckets([int(x,16)for x in v['signatures']],v['bits'],v['radius'],max_blocks=v['max_blocks']);return {'input':v,'report':h.lookup(int(v['query'],16),exclude_indices=v['exclude_indices'],mode=v['mode'])}

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);a=p.parse_args(argv)
    try:
        with open(a.input,encoding='utf-8')as f:v=parse(f.read(131073))
        print(json.dumps(audit(v),indent=2,allow_nan=False));return 0
    except(ValueError,OSError,OverflowError)as e:print('ERROR: '+str(e),file=sys.stderr);return 2
if __name__=='__main__':sys.exit(main())
