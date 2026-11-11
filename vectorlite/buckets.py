"""Exact Hamming radius retrieval via disjoint-block postings, not cosine NN."""
from types import MappingProxyType
from .vectors import positive_integer,cosine_similarity,checked_vector
from .lsh import LSHIndex

class HammingBuckets:
    """Detached immutable signature snapshot. Build once at fixed radius.

    Pigeonhole specialization of multi-index hashing: r+1 exact blocks,
    followed by full-code validation. Dense cases can still require a scan.
    """
    def __init__(self, signatures, bits, radius, *, max_blocks=32):
        positive_integer(bits,'bits');positive_integer(radius,'radius',zero=True);positive_integer(max_blocks,'max_blocks')
        if bits>4096 or radius>bits or max_blocks>64:raise ValueError('bits<=4096, radius<=bits, max_blocks<=64 required')
        values=tuple(signatures)
        if any(type(x)is not int or not 0<=x<(1<<bits)for x in values):raise ValueError('signatures must be bounded nonboolean integers')
        self._signatures=values;self._bits=bits;self._radius=radius;self._max_blocks=max_blocks
        count=radius+1;self._reason='radius-covers-all' if radius==bits else 'block-cap' if count>max_blocks else None
        blocks=[];tables=[];offset=0
        if self._reason is None:
            q,remainder=divmod(bits,count)
            for b in range(count):
                length=q+(b<remainder);mask=(1<<length)-1;table={}
                for i,value in enumerate(values):table.setdefault((value>>offset)&mask,[]).append(i)
                tables.append(MappingProxyType({k:tuple(v)for k,v in table.items()}));blocks.append((offset,length));offset+=length
            assert offset==bits
        self._blocks=tuple(blocks);self._tables=tuple(tables)
    @property
    def bits(self):return self._bits
    @property
    def radius(self):return self._radius
    @property
    def signatures(self):return self._signatures
    def construction(self):
        return {'items':len(self._signatures),'bits':self._bits,'radius':self._radius,'max_blocks':self._max_blocks,'fallback_reason':self._reason,'blocks':[{'offset':o,'length':n,'occupied_buckets':len(t)}for(o,n),t in zip(self._blocks,self._tables)],'posting_references':len(self._signatures)*len(self._blocks),'occupied_buckets':sum(map(len,self._tables))}
    def lookup(self, query, *, exclude_indices=(), mode='auto'):
        if type(query)is not int or not 0<=query<(1<<self._bits):raise ValueError('query must be a bounded nonboolean integer')
        if mode not in ('auto','blocks','scan'):raise ValueError('mode must be auto, blocks or scan')
        excluded=tuple(exclude_indices)
        if any(type(i)is not int or not 0<=i<len(self._signatures)for i in excluded)or len(set(excluded))!=len(excluded):raise ValueError('excluded indices must be unique and in range')
        ex=set(excluded);lookups=[];lists=[]
        if mode!='scan' and self._reason is None:
            for b,((offset,length),table)in enumerate(zip(self._blocks,self._tables)):
                key=(query>>offset)&((1<<length)-1);postings=table.get(key,());lists.append(postings);lookups.append({'block':b,'key':format(key,'x'),'posting_length':len(postings)})
        posting_total=sum(len(x)for x in lists)
        reason='requested-scan' if mode=='scan' else self._reason or ('posting-cost' if mode=='auto' and posting_total>=len(self._signatures)else None)
        strategy='scan' if reason else 'blocks'
        union=set() if strategy=='blocks' else None
        if union is not None:
            for posting in lists:union.update(posting)
            pool=sorted(union)
        else:pool=range(len(self._signatures))
        checks=[];admitted=[]
        for i in pool:
            if i in ex:continue
            distance=(query^self._signatures[i]).bit_count();passed=distance<=self._radius;checks.append({'index':i,'distance':distance,'admitted':passed})
            if passed:admitted.append(i)
        return {'schema':'vectorlite-buckets/1','query_hex':format(query,'x'),'excluded_indices':sorted(ex),'requested_mode':mode,'strategy':strategy,'fallback_reason':reason,'construction':self.construction(),'lookups':lookups,'work':{'bucket_lookups':len(lookups),'posting_lengths_total':posting_total,'posting_references_visited':posting_total if union is not None else 0,'unique_posting_candidates':len(union)if union is not None else None,'full_hamming_checks':len(checks),'excluded_pool_entries':sum(i in ex for i in pool),'scan_reference_hamming_checks':len(self._signatures)-len(ex)},'checks':checks,'admitted_indices':admitted}

class BucketLSHIndex:
    """Optional exact-same-threshold replacement for LSHIndex's signature scan.

    Cosine/id rescoring unchanged. This does not make signature filtering exact
    cosine nearest-neighbor search. No online insertion/removal is implemented.
    """
    def __init__(self,items,num_bits=64,seed=17,min_shared_bits=None,*,hyperplanes=None,max_blocks=32):
        base=LSHIndex(items,num_bits,seed,min_shared_bits,hyperplanes=hyperplanes)
        self._source=base
        self._items=tuple((x['id'],tuple(x['vector']))for x in base.items)
        self._lookup=HammingBuckets(base.signatures,num_bits,num_bits-base.min_shared_bits,max_blocks=max_blocks)
    def query_audit(self,query,k=10,*,exclude_ids=(),mode='auto'):
        positive_integer(k,'k',zero=True);query=checked_vector(query,self._source.dims,empty=not self._items)
        ids=tuple(exclude_ids)
        if any(type(x)is not str for x in ids)or len(set(ids))!=len(ids):raise ValueError('exclude_ids must be distinct strings')
        excluded=set(ids);indices=[i for i,(id,_)in enumerate(self._items)if id in excluded]
        report=self._lookup.lookup(self._source._signature(query),exclude_indices=indices,mode=mode)
        scored=sorted([(self._items[i][0],cosine_similarity(query,self._items[i][1]))for i in report['admitted_indices']],key=lambda x:(-x[1],x[0]))
        report['cosine_rescores']=len(scored);report['ranking']=[{'id':id,'cosine':score}for id,score in scored[:k]]
        return report
    def query(self,query,k=10,*,exclude_ids=(),mode='auto'):
        r=self.query_audit(query,k,exclude_ids=exclude_ids,mode=mode)
        return [(x['id'],x['cosine'])for x in r['ranking']]
