"""Run from outside the checkout in an environment with the wheel installed."""
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import vectorlite
from vectorlite.audit import audit
fixtures=json.loads(Path(sys.argv[1]).read_text())['cases']
root=Path(sys.argv[2]).resolve()
for name in ['vectorlite','vectorlite.audit','vectorlite.exact','vectorlite.lsh','vectorlite.vectors','vectorlite.cli']:
    module=importlib.import_module(name)
    if root in Path(module.__file__).resolve().parents:
        raise AssertionError('checkout import '+name)
for c in fixtures:assert audit(c['spec'])==c['report']
with tempfile.TemporaryDirectory() as t:
    p=Path(t)/'s.json';p.write_text(json.dumps(fixtures[0]['spec']))
    r=subprocess.run([sys.executable,'-m','vectorlite','audit','--spec',str(p)],capture_output=True,text=True,check=True)
    assert json.loads(r.stdout)==fixtures[0]['report']
    v=Path(t)/'v.jsonl';v.write_text('\n'.join(json.dumps(x) for x in fixtures[0]['spec']['items']))
    subprocess.run([sys.executable,'-m','vectorlite','query','--vectors',str(v),'--query',','.join(map(str,fixtures[0]['spec']['query'])),'-k','3'],check=True)
print(json.dumps({'module_origins':6,'audit_cases':len(fixtures),'self_controls':[x['report']['selected']['recall'] for x in fixtures[-2:]],'audit_cli':True,'original_query_cli':True}))
