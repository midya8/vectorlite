/* Independent scalar JavaScript. No Python calculations are used at runtime. */
(function(root){'use strict';
const integer=(x,lo,hi,name)=>{if(!Number.isInteger(x)||x<lo||x>hi)throw Error(name+' out of bounds');return x;};
function vec(v,d){if(!Array.isArray(v)||!v.length||v.length>16||v.some(x=>typeof x!=='number'||!Number.isFinite(x))||(d!==undefined&&v.length!==d))throw Error('finite matching vectors required');return v.slice();}
function unit(v){let s=Math.max(...v.map(Math.abs));if(s===0)return v.map(()=>0);let a=v.map(x=>x/s),n=Math.hypot(...a);return a.map(x=>x/n);}
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const cos=(a,b)=>Math.max(-1,Math.min(1,dot(unit(a),unit(b))));
const agreement=(a,b)=>{const u=unit(a),v=unit(b);return 1-2*Math.atan2(Math.hypot(...u.map((x,i)=>x-v[i])),Math.hypot(...u.map((x,i)=>x+v[i])))/Math.PI;};
function tail(n,t,p){if(p===null)return null;let sum=0,c=1;for(let j=0;j<=n;j++){if(j>=t)sum+=c*Math.pow(p,j)*Math.pow(1-p,n-j);c=c*(n-j)/(j+1);}return Math.min(1,sum);}
const rank=(a,b)=>b.cosine-a.cosine||(a.id<b.id?-1:a.id>b.id?1:0);
function audit(s){
 if(!s||typeof s!=='object'||Array.isArray(s)||Object.keys(s).some(k=>!['items','query','bits','seed','k','threshold','exclude_ids','hyperplanes'].includes(k)))throw Error('unknown spec fields');
 if(!Array.isArray(s.items)||s.items.length<1||s.items.length>64)throw Error('1..64 items required');
 const seen=new Set(),items=s.items.map(x=>{if(!x||typeof x.id!=='string'||!x.id.length||x.id.length>64||/[^\x00-\x7F]/.test(x.id)||seen.has(x.id))throw Error('distinct ASCII ids required');seen.add(x.id);return {id:x.id,vector:vec(x.vector)};});
 const dims=items[0].vector.length;items.forEach(x=>vec(x.vector,dims));const query=vec(s.query,dims),bits=integer(s.bits??16,8,128,'bits'),k=integer(s.k??3,1,64,'k'),threshold=integer(s.threshold??Math.floor(bits*3/4),0,bits,'threshold');
 const excluded=s.exclude_ids??[];if(!Array.isArray(excluded)||new Set(excluded).size!==excluded.length||excluded.some(x=>!seen.has(x)))throw Error('excluded ids must exist');
 if(!Number.isInteger(s.seed??17))throw Error('integer seed required');
 if(!Array.isArray(s.hyperplanes)||s.hyperplanes.length!==bits)throw Error('supply one plane per bit');
 const planes=s.hyperplanes.map(x=>{const v=vec(x,dims);if(!v.some(Boolean))throw Error('nonzero planes required');return v;}),projections=v=>planes.map(p=>dot(unit(p),unit(v))),signature=p=>p.map(x=>x>=0?'1':'0').reverse().join(''),qp=projections(query),qs=signature(qp);
 const rows=items.map(x=>{const c=cos(query,x.vector),pr=projections(x.vector),sig=signature(pr);return {id:x.id,vector:x.vector,unit:unit(x.vector),cosine:c,projections:pr,signature:sig,shared:[...sig].reduce((n,b,i)=>n+(b===qs[i]),0),excluded:excluded.includes(x.id),exact_top_k:false,bit_agreement_probability:query.some(Boolean)&&x.vector.some(Boolean)?agreement(query,x.vector):null};});
 const eligible=rows.filter(x=>!x.excluded),truth=eligible.slice().sort(rank).slice(0,k);truth.forEach(x=>x.exact_top_k=true);const exact=truth.map(x=>[x.id,x.cosine]),curve=[];
 for(let t=0;t<=bits;t++){const admitted=eligible.filter(x=>x.shared>=t).sort(rank),results=admitted.slice(0,k).map(x=>[x.id,x.cosine]),probs=eligible.map(x=>tail(bits,t,x.bit_agreement_probability)),tp=eligible.filter(x=>x.exact_top_k).map(x=>tail(bits,t,x.bit_agreement_probability));curve.push({threshold:t,candidate_ids:admitted.map(x=>x.id),candidate_count:admitted.length,rescored_fraction:eligible.length?admitted.length/eligible.length:0,results,recall:truth.length?admitted.filter(x=>x.exact_top_k).length/truth.length:1,theory_expected_candidates:probs.includes(null)?null:probs.reduce((a,b)=>a+b,0),theory_expected_recall:truth.length?(tp.includes(null)?null:tp.reduce((a,b)=>a+b,0)/truth.length):1});}
 return {items,query,query_unit:unit(query),bits,seed:s.seed??17,k,threshold,exclude_ids:excluded,hyperplanes:planes,query_projections:qp,query_signature:qs,rows,exact,curve,selected:curve[threshold],signature_scan_count:items.length,eligible_count:eligible.length};
}
const api={unit,cos,tail,audit};if(typeof module!=='undefined')module.exports=api;root.VectorAudit=api;
})(typeof window==='undefined'?globalThis:window);
