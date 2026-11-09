"""Complete signature, candidate and threshold witnesses for small cosine indexes."""
import json
import math
from pathlib import Path
from .vectors import checked_items, checked_vector, positive_integer, cosine_similarity, unit
from .exact import ExactIndex
from .lsh import LSHIndex
from .recall import recall_at_k


def admission_probability(bits, threshold, probability):
    """P[Binomial(bits,p) >= threshold], direct positive-term sum for <=128 bits."""
    positive_integer(bits,"bits")
    positive_integer(threshold,"threshold",zero=True)
    if bits > 128 or threshold > bits or isinstance(probability,bool) or not isinstance(probability,(int,float)) or not 0 <= probability <= 1:
        raise ValueError("invalid bounded binomial arguments")
    return min(1.0, math.fsum(math.comb(bits,j)*probability**j*(1-probability)**(bits-j)
                              for j in range(threshold,bits+1)))


def audit(spec):
    allowed = {"items","query","bits","seed","k","threshold","exclude_ids","hyperplanes"}
    if not isinstance(spec,dict) or set(spec)-allowed:
        raise ValueError("spec must be an object with documented keys only")
    raw = spec.get("items")
    if not isinstance(raw,list) or not 1 <= len(raw) <= 64:
        raise ValueError("audit needs1..64 items")
    items = checked_items(raw)
    dims = len(items[0]["vector"])
    if dims > 16 or any(len(x["id"])>64 or not x["id"].isascii() for x in items):
        raise ValueError("audit dimensions <=16 and ASCII id lengths <=64")
    query = checked_vector(spec.get("query"),dims)
    bits = spec.get("bits",16)
    positive_integer(bits,"bits")
    if bits > 128:
        raise ValueError("audit bits <=128")
    k = spec.get("k",3)
    positive_integer(k,"k")
    if k > 64:
        raise ValueError("audit k <=64")
    excluded = spec.get("exclude_ids",[])
    if not isinstance(excluded,list) or any(not isinstance(x,str) for x in excluded) or len(set(excluded))!=len(excluded):
        raise ValueError("exclude_ids must be distinct strings")
    if not set(excluded) <= {x["id"] for x in items}:
        raise ValueError("unknown excluded id")
    index = LSHIndex(items,num_bits=bits,seed=spec.get("seed",17),
                     min_shared_bits=spec.get("threshold"),hyperplanes=spec.get("hyperplanes"))
    exact = ExactIndex(items).query(query,k=k,exclude_ids=excluded)
    truth = {x[0] for x in exact}
    qproj = index.projections(query)
    qsig = index._signature(query)
    rows=[]
    for item,sig in zip(index.items,index.signatures):
        cos=cosine_similarity(query,item["vector"])
        defined=any(query) and any(item["vector"])
        u,v=unit(query),unit(item["vector"])
        p=1-2*math.atan2(math.hypot(*(a-b for a,b in zip(u,v))),
                              math.hypot(*(a+b for a,b in zip(u,v))))/math.pi if defined else None
        rows.append({"id":item["id"],"vector":item["vector"],"unit":unit(item["vector"]),
            "cosine":cos,"projections":index.projections(item["vector"]),
            "signature":format(sig,"0%db"%bits),"shared":bits-(qsig^sig).bit_count(),
            "excluded":item["id"] in excluded,"exact_top_k":item["id"] in truth,
            "bit_agreement_probability":p})
    eligible=[x for x in rows if not x["excluded"]]
    curve=[]
    for t in range(bits+1):
        admitted=sorted([x for x in eligible if x["shared"]>=t],key=lambda x:(-x["cosine"],x["id"]))
        result=[(x["id"],x["cosine"]) for x in admitted[:k]]
        probs=[admission_probability(bits,t,x["bit_agreement_probability"])
               if x["bit_agreement_probability"] is not None else None for x in eligible]
        expected_candidates=math.fsum(probs) if None not in probs else None
        tp=[v for x,v in zip(eligible,probs) if x["exact_top_k"]]
        expected_recall=(math.fsum(tp)/len(truth) if None not in tp else None) if truth else 1.0
        curve.append({"threshold":t,"candidate_ids":[x["id"] for x in admitted],
            "candidate_count":len(admitted),"rescored_fraction":len(admitted)/len(eligible) if eligible else 0.0,
            "results":[list(x) for x in result],"recall":recall_at_k(result,exact),
            "theory_expected_candidates":expected_candidates,"theory_expected_recall":expected_recall})
    return {"schema":"vectorlite.audit.v1","items":items,"query":query,"query_unit":unit(query),
        "bits":bits,"seed":index.seed,"k":k,"threshold":index.min_shared_bits,"exclude_ids":excluded,
        "plane_source":"supplied" if spec.get("hyperplanes") is not None else "python-random-gaussian",
        "hyperplanes":index.hyperplanes,"query_projections":qproj,"query_signature":format(qsig,"0%db"%bits),
        "rows":rows,"exact":[list(x) for x in exact],"curve":curve,
        "selected":curve[index.min_shared_bits],"signature_scan_count":len(items),
        "eligible_count":len(eligible),"theory_contract":"iid isotropic nonzero planes; marginal expectation over rebuilds, not conditional on these fixed planes; zero-vector angular theory undefined"}


def _unique(pairs):
    out={}
    for key,value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key: "+key)
        out[key]=value
    return out


def load_spec(path):
    if Path(path).stat().st_size>131072:
        raise ValueError("spec exceeds128KiB")
    data=Path(path).read_bytes()
    if len(data)>131072:
        raise ValueError("spec exceeds128KiB")
    return json.loads(data,object_pairs_hook=_unique,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def cli_audit(args):
    result=audit(load_spec(args.spec))
    text=json.dumps(result,indent=2,allow_nan=False)+"\n"
    if args.json:
        Path(args.json).write_text(text)
    else:
        print(text,end="")
    return 0
