#!/usr/bin/env python3
"""Validate optional MiniLM embeddings API acceptance artifacts.

With no artifact directory arguments this runs a dependency-free synthetic self-test.
With one or more directories it validates artifacts produced by
scripts/minilm_embeddings_optional_api_acceptance_smoke.sh.
"""
from __future__ import annotations
import json, math, re, tempfile
from pathlib import Path
from typing import Any

MODEL_ID='sentence-transformers-all-minilm-l6-v2-model-safetensors'
REPO_ID='sentence-transformers/all-MiniLM-L6-v2'
REVISION='c9745ed1d9f207416be6d2e6f8de32d1f16199bf'
EXPECTED_FILES={
 'config.json':(612,'953f9c0d463486b10a6871cc2fd59f223b2c70184f49815e7efbcab5d8908b41'),
 'model.safetensors':(90_868_376,'53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db'),
 'modules.json':(349,'84e40c8e006c9b1d6c122e02cba9b02458120b5fb0c87b746c41e0207cf642cf'),
 'special_tokens_map.json':(112,'303df45a03609e4ead04bc3dc1536d0ab19b5358db685b6f3da123d05ec200e3'),
 'tokenizer.json':(466_247,'be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037'),
 'tokenizer_config.json':(350,'acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b'),
 '1_Pooling/config.json':(190,'4be450dde3b0273bb9787637cfbd28fe04a7ba6ab9d36ac48e92b11e350ffc23'),
}
EXPECTED_TOTAL_BYTES=91_336_236
REQUIRED={'01-v1-health.json','02-install-minilm.json','03-api-embedding-models.json','04-v1-models-exclude-minilm.json','05-v1-embeddings-float.json','06-v1-embeddings-base64-refusal.json','07-chat-embedding-model-refusal.json','summary.json','summary.md'}
EXPECTED_CHECK_ARTIFACTS=REQUIRED-{'summary.json','summary.md'}
REQUIRED_CAVEAT_PHRASES=(
    'Optional local',
    'embedding quality',
    'retrieval quality',
    'latency',
    'throughput',
    'production readiness',
    'legal suitability',
    'arbitrary Hugging Face',
    'ONNX '+'chat',
    'streaming',
    'external proxying',
    'full OpenAI API parity',
    'GGUF tokenizer execution',
    'GGUF '+'runtime',
    'weight loading',
    'generation',
    'dequantization',
    'inference',
)
LOCAL_PATH_PATTERNS=[re.compile('/'+'Users'+'/',re.I),re.compile('/'+'private'+'/'+'tmp',re.I),re.compile('/'+'opt'+'/'+'homebrew',re.I)]
OVERCLAIMS=re.compile(r'(quality\s+proved|production\s+ready|legal\s+(approved|review)|full\s+OpenAI\s+parity|arbitrary\s+(Hugging\s+Face|SafeTensors)|GGUF\s+(runtime|generation|inference|tokenizer\s+execution)|ONNX\s+chat\s+(works|supported)|external\s+provider\s+(proxy|call)\s+(works|succeeded|enabled))',re.I)

def load_json(path:Path)->dict[str,Any]:
    data=json.loads(path.read_text())
    if not isinstance(data,dict): raise AssertionError(f'{path.name} must be object')
    return data

def assert_share_safe(path:Path)->None:
    text=path.read_text()
    for p in LOCAL_PATH_PATTERNS:
        if p.search(text): raise AssertionError(f'{path.name} contains local path-like leak')
    for line in text.splitlines():
        if OVERCLAIMS.search(line) and not re.search(r'\b(no|not|does not|without|doesn\'t)\b',line,re.I):
            raise AssertionError(f'{path.name} contains an overclaim')

def assert_required_caveats(text:str,label:str)->None:
    lowered=text.lower()
    missing=[phrase for phrase in REQUIRED_CAVEAT_PHRASES if phrase.lower() not in lowered]
    if missing: raise AssertionError(f'{label} missing caveat phrase(s): {missing}')

def assert_checks_cover_required_artifacts(checks:Any)->None:
    if not isinstance(checks,list): raise AssertionError('summary.checks must be a list')
    artifacts=[check.get('artifact') for check in checks if isinstance(check,dict)]
    missing=sorted(EXPECTED_CHECK_ARTIFACTS-set(artifacts))
    unexpected=sorted(set(artifacts)-EXPECTED_CHECK_ARTIFACTS)
    duplicates=sorted({artifact for artifact in artifacts if artifact and artifacts.count(artifact)>1})
    if missing or unexpected or duplicates:
        raise AssertionError(f'summary.checks artifact index mismatch: missing={missing} unexpected={unexpected} duplicates={duplicates}')
    for check in checks:
        if not isinstance(check,dict): raise AssertionError('summary.checks entries must be objects')
        if check.get('status')!='passed': raise AssertionError(f'summary check failed: {check}')

def assert_markdown_checks_match_summary(checks:Any,md:str)->None:
    if not isinstance(checks,list): return
    for check in checks:
        if not isinstance(check,dict): continue
        name=check.get('name'); status=check.get('status'); artifact=check.get('artifact')
        if all(isinstance(value,str) and value for value in (name,status,artifact)):
            row=f"- `{name}`: `{status}` ({artifact})"
            if row not in md:
                raise AssertionError(f'summary.md missing check row matching summary.json: {name}')

def assert_loopback_base_url(value:Any)->None:
    if not isinstance(value,str) or not re.fullmatch(r'http://127\.0\.0\.1:[0-9]{2,5}',value):
        raise AssertionError('summary.base_url must be an http://127.0.0.1:<port> loopback URL')
    port=int(value.rsplit(':',1)[1])
    if not 1 <= port <= 65535:
        raise AssertionError('summary.base_url port must be in range 1-65535')

def validate_install(install:dict[str,Any])->None:
    if install.get('id')!=MODEL_ID or install.get('task')!='text_embedding': raise AssertionError('install identity/task mismatch')
    manifest=install.get('download_manifest') or {}
    for k,v in {'repo_id':REPO_ID,'revision':REVISION,'license':'apache-2.0','license_status':'permissive','verification_status':'verified'}.items():
        if manifest.get(k)!=v: raise AssertionError(f'manifest {k} mismatch')
    by={f.get('filename'):f for f in manifest.get('files') or [] if isinstance(f,dict)}
    if set(by)!=set(EXPECTED_FILES): raise AssertionError(f'file set mismatch {sorted(by)}')
    total=0
    for name,(size,sha) in EXPECTED_FILES.items():
        if by[name].get('size_bytes')!=size or by[name].get('sha256')!=sha: raise AssertionError(f'mismatch {name}')
        total+=size
    if total!=EXPECTED_TOTAL_BYTES: raise AssertionError('byte total mismatch')

def validate_embeddings(payload:dict[str,Any])->None:
    if payload.get('object')!='list' or payload.get('model')!=MODEL_ID: raise AssertionError('embedding envelope mismatch')
    data=payload.get('data')
    if not isinstance(data,list) or len(data)!=2: raise AssertionError('expected two embedding rows')
    for idx,item in enumerate(data):
        vec=item.get('embedding')
        if item.get('object')!='embedding' or item.get('index')!=idx or not isinstance(vec,list) or len(vec)!=384 or not all(isinstance(v,float) and math.isfinite(v) for v in vec):
            raise AssertionError('invalid embedding row')
    f=payload.get('fathom') or {}
    if f.get('runtime')!='candle-bert-embeddings' or f.get('embedding_dimension')!=384 or f.get('scope')!='verified local embedding runtime only':
        raise AssertionError('embedding fathom metadata mismatch')

def validate_summary(directory:Path)->None:
    missing=sorted(n for n in REQUIRED if not (directory/n).exists())
    if missing: raise AssertionError(f'missing artifacts {missing}')
    for n in ('summary.json','summary.md'): assert_share_safe(directory/n)
    summary=load_json(directory/'summary.json')
    if summary.get('schema')!='fathom.minilm_embeddings_optional_api_acceptance.summary.v1' or summary.get('passed') is not True: raise AssertionError('summary mismatch')
    assert_loopback_base_url(summary.get('base_url'))
    for k in ('artifact_dir','model_dir','state_dir','log_dir'):
        v=summary.get(k)
        if not isinstance(v,str) or v.startswith('/'): raise AssertionError(f'summary.{k} not share-safe')
    if summary.get('model_id')!=MODEL_ID or summary.get('repo_id')!=REPO_ID or summary.get('revision')!=REVISION: raise AssertionError('summary model identity mismatch')
    assert_checks_cover_required_artifacts(summary.get('checks'))
    caveats='\n'.join(str(x) for x in summary.get('caveats') or [])
    assert_required_caveats(caveats,'summary.json caveats')
    md=(directory/'summary.md').read_text()
    if 'Result: `passed`' not in md or 'What this does not prove' not in md:
        raise AssertionError('summary.md must clearly mark pass and caveats')
    assert_required_caveats(md,'summary.md')
    assert_markdown_checks_match_summary(summary.get('checks'),md)
    validate_install(load_json(directory/'02-install-minilm.json'))
    emb_models=load_json(directory/'03-api-embedding-models.json')
    if MODEL_ID not in [i.get('id') for i in emb_models.get('items',[])]: raise AssertionError('embedding models missing MiniLM')
    v1_models=load_json(directory/'04-v1-models-exclude-minilm.json')
    if MODEL_ID in [i.get('id') for i in v1_models.get('data',[])]: raise AssertionError('MiniLM leaked into /v1/models')
    validate_embeddings(load_json(directory/'05-v1-embeddings-float.json'))
    b64=load_json(directory/'06-v1-embeddings-base64-refusal.json')
    if (b64.get('error') or {}).get('code')!='invalid_request' or 'data' in b64: raise AssertionError('base64 refusal mismatch')
    chat=load_json(directory/'07-chat-embedding-model-refusal.json')
    if 'choices' in chat or (chat.get('error') or {}).get('code') not in {'not_implemented','model_not_found'}: raise AssertionError('chat refusal mismatch')

def sample_embedding():
    return {'object':'list','model':MODEL_ID,'data':[{'object':'embedding','index':0,'embedding':[0.0]*384},{'object':'embedding','index':1,'embedding':[0.1]*384}], 'fathom':{'runtime':'candle-bert-embeddings','embedding_dimension':384,'scope':'verified local embedding runtime only'}}

def write_sample(d:Path)->None:
    d.mkdir(parents=True,exist_ok=True)
    (d/'01-v1-health.json').write_text(json.dumps({'status':'ok'})+'\n')
    (d/'02-install-minilm.json').write_text(json.dumps({'id':MODEL_ID,'task':'text_embedding','download_manifest':{'repo_id':REPO_ID,'revision':REVISION,'license':'apache-2.0','license_status':'permissive','verification_status':'verified','files':[{'filename':n,'size_bytes':s,'sha256':h} for n,(s,h) in EXPECTED_FILES.items()]}},indent=2)+'\n')
    (d/'03-api-embedding-models.json').write_text(json.dumps({'items':[{'id':MODEL_ID}]},indent=2)+'\n')
    (d/'04-v1-models-exclude-minilm.json').write_text(json.dumps({'object':'list','data':[]},indent=2)+'\n')
    (d/'05-v1-embeddings-float.json').write_text(json.dumps(sample_embedding(),indent=2)+'\n')
    (d/'06-v1-embeddings-base64-refusal.json').write_text(json.dumps({'error':{'code':'invalid_request','type':'invalid_request','message':'base64 unsupported','param':None}},indent=2)+'\n')
    (d/'07-chat-embedding-model-refusal.json').write_text(json.dumps({'error':{'code':'not_implemented','type':'not_implemented','message':'not chat','param':None}},indent=2)+'\n')
    checks=[{'name':n,'artifact':a,'description':n,'expected_http_status':200,'http_status':200,'status':'passed'} for n,a in [('health','01-v1-health.json'),('install_minilm','02-install-minilm.json'),('embedding_models_include_minilm','03-api-embedding-models.json'),('v1_models_exclude_minilm','04-v1-models-exclude-minilm.json'),('v1_embeddings_float','05-v1-embeddings-float.json'),('base64_refusal','06-v1-embeddings-base64-refusal.json'),('chat_embedding_refusal','07-chat-embedding-model-refusal.json')]]
    summary={'schema':'fathom.minilm_embeddings_optional_api_acceptance.summary.v1','passed':True,'repo_commit':'sample','started_at':'2026-04-29T00:00:00Z','finished_at':'2026-04-29T00:00:01Z','base_url':'http://127.0.0.1:18187','artifact_dir':'.','model_dir':'models/','state_dir':'state/','log_dir':'logs/','model_id':MODEL_ID,'repo_id':REPO_ID,'revision':REVISION,'checks':checks,'caveats':['Optional local embedding evidence only; not default CI.','Does not prove embedding quality, retrieval quality, latency, throughput, production readiness, legal suitability, arbitrary Hugging Face execution, ONNX chat, streaming, external proxying, or full OpenAI API parity.','Does not claim GGUF tokenizer execution, GGUF runtime, weight loading, generation, dequantization, or inference.']}
    (d/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    (d/'summary.md').write_text('# MiniLM embeddings optional API acceptance artifacts\n\n- Result: `passed`\n- Scope: optional local embedding API evidence only; not default CI.\n- Artifact directory: `.`\n- State directory: `state/`\n- Model directory: `models/`\n- Server log: `logs/server.log`\n\n## Checks\n\n'+'\n'.join(f"- `{c['name']}`: `{c['status']}` ({c['artifact']})" for c in checks)+'\n\n## What this does not prove\n\n- No embedding quality, retrieval quality, latency, throughput, production readiness, legal suitability, arbitrary Hugging Face execution, ONNX chat/general execution, streaming, external proxying, or full OpenAI API parity claim.\n- No public/runtime GGUF tokenizer execution, GGUF runtime, weight loading, generation, dequantization, or inference claim.\n')

def main():
    import sys
    dirs=[Path(a) for a in sys.argv[1:]]
    if not dirs:
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp)/'sample'; write_sample(d); validate_summary(d)
            bad=Path(tmp)/'missing-caveat'; write_sample(bad)
            summary=load_json(bad/'summary.json')
            summary['caveats']=[str(item).replace('retrieval quality, ','') for item in summary['caveats']]
            (bad/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
            try:
                validate_summary(bad)
            except AssertionError as exc:
                if 'retrieval quality' not in str(exc): raise
            else:
                raise AssertionError('missing caveat self-check did not fail')
            bad_index=Path(tmp)/'missing-check-index'; write_sample(bad_index)
            summary=load_json(bad_index/'summary.json')
            summary['checks']=[check for check in summary['checks'] if check.get('artifact')!='07-chat-embedding-model-refusal.json']
            (bad_index/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
            try:
                validate_summary(bad_index)
            except AssertionError as exc:
                if 'summary.checks artifact index mismatch' not in str(exc): raise
            else:
                raise AssertionError('missing check artifact self-check did not fail')
            bad_md=Path(tmp)/'missing-pass-marker'; write_sample(bad_md)
            (bad_md/'summary.md').write_text((bad_md/'summary.md').read_text().replace('- Result: `passed`\n',''))
            try:
                validate_summary(bad_md)
            except AssertionError as exc:
                if 'summary.md must clearly mark pass and caveats' not in str(exc): raise
            else:
                raise AssertionError('missing summary.md pass marker self-check did not fail')
            bad_md_index=Path(tmp)/'missing-markdown-check-index'; write_sample(bad_md_index)
            md=(bad_md_index/'summary.md').read_text()
            (bad_md_index/'summary.md').write_text(md.replace("- `chat_embedding_refusal`: `passed` (07-chat-embedding-model-refusal.json)\n",''))
            try:
                validate_summary(bad_md_index)
            except AssertionError as exc:
                if 'summary.md missing check row matching summary.json' not in str(exc): raise
            else:
                raise AssertionError('missing summary.md check row self-check did not fail')
            bad_base=Path(tmp)/'bad-base-url'; write_sample(bad_base)
            summary=load_json(bad_base/'summary.json')
            summary['base_url']='https://example.invalid'
            (bad_base/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
            try:
                validate_summary(bad_base)
            except AssertionError as exc:
                if 'summary.base_url' not in str(exc): raise
            else:
                raise AssertionError('external summary.base_url self-check did not fail')
        print('MiniLM embeddings optional API acceptance artifact QA self-test passed'); return
    for d in dirs: validate_summary(d)
    print('MiniLM embeddings optional API acceptance artifact QA passed')
if __name__=='__main__': main()
