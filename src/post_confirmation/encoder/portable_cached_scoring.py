"""Versioned path-only companion for frozen v1.1 cached scoring; never encodes."""
import argparse,ast,hashlib,importlib.metadata,json,os,platform,shutil,sys
from pathlib import Path
from datetime import datetime,timezone
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def stamp():return datetime.now(timezone.utc).isoformat()
def inventory():
 return sorted([dict(name=d.metadata['Name'],version=d.version,location=str(d.locate_file(''))) for d in importlib.metadata.distributions()],key=lambda d:(d['name'].lower(),d['location']))
def relocated_tree(source,project,work,out):
 original=ast.parse(source.read_text());tree=ast.parse(source.read_text());changes=[]
 for node in tree.body:
  if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ['ROOT','BASE','OUT']:
   name=node.targets[0].id;value={'ROOT':project,'BASE':work,'OUT':out}[name]
   node.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(value=str(value))],keywords=[]);changes.append(name)
 assert sorted(changes)==['BASE','OUT','ROOT']
 before={n.name:ast.dump(n,include_attributes=False) for n in original.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
 after={n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
 assert before==after,'A scientific or verification function was modified'
 return ast.fix_missing_locations(tree),hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest()
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--project-root',type=Path,required=True);ap.add_argument('--work-root',type=Path,required=True);ap.add_argument('--output-root',type=Path);ap.add_argument('--run',action='store_true',help='Reconstruct scores in a fresh output directory; default verifies inputs only.');a=ap.parse_args()
 project=a.project_root.resolve();work=a.work_root.resolve();source=work/'methods/encoder_check_v1_1.py';cached=work/'encoder_sensitivity'
 original=cached/'pre_execution_manifest.json';amendment=cached/'pre_execution_serialization_amendment.json'
 frozen=json.loads(original.read_text());amended=json.loads(amendment.read_text())
 assert sha(source)==amended['code_sha256'];assert sha(original)==amended['supersedes_freeze_sha256'];assert sha(work/'methods/encoder_check.py')==frozen['code_sha256'];assert sha(work/'methods/encoder_protocol.md')==frozen['protocol_sha256']==amended['protocol_sha256']
 for rel,digest in frozen['inputs_sha256'].items():assert sha(project/rel)==digest,rel
 logical=cached/'requirements_encoder_logical.lock'
 for line in logical.read_text().splitlines():
  if line and not line.startswith('#'):
   name,version=line.split('==');assert importlib.metadata.version(name)==version,(name,importlib.metadata.version(name),version)
 for study in ['acl','biomedical']:
  em=json.loads((cached/study/'encoding_manifest.json').read_text());assert em['status']=='complete' and em['freeze_sha256']==sha(original)
  for name,digest in em['output_sha256'].items():assert sha(cached/study/name)==digest,(study,name)
 original_scoring=json.loads((cached/'scoring_manifest.json').read_text());assert original_scoring['status']=='complete'
 for name,digest in original_scoring['output_sha256'].items():assert sha(cached/name)==digest,name
 if not a.run:
  tree,functions_hash=relocated_tree(source,project,work,cached)
  compile(tree,str(source),'exec')
  print(json.dumps({'status':'inputs_and_path_only_AST_verified; no scoring or encoding run','source_sha256':sha(source),'companion_sha256':sha(Path(__file__)),'unchanged_functions_ast_sha256':functions_hash}));return
 assert a.output_root,'--run requires --output-root';out=a.output_root.resolve();assert not out.exists(),'Refusing existing output root to avoid duplication or replacement'
 assert out!=cached and cached not in out.parents and project not in out.parents,'Use a separate reconstruction output directory'
 # Copy inputs, rather than hard-linking immutable arrays to writable reproduction files.
 out.mkdir(parents=True)
 for name in ['pre_execution_manifest.json','pre_execution_serialization_amendment.json']:
  shutil.copyfile(cached/name,out/name)
 for study in ['acl','biomedical']:
  (out/study).mkdir()
  for name in ['encoding_manifest.json','full.npy','title.npy','token_audit.json']:shutil.copyfile(cached/study/name,out/study/name)
 tree,functions_hash=relocated_tree(source,project,work,out)
 receipt={'created_utc':stamp(),'status':'preflight_complete_before_cached_scoring','scope':'Path-only reconstruction from archived encoded arrays; no fresh acquisition, encoder execution or clean-machine success claimed','project_root':str(project),'work_root':str(work),'output_root':str(out),'companion_sha256':sha(Path(__file__)),'original_source_sha256':sha(work/'methods/encoder_check.py'),'amended_source_sha256':sha(source),'original_freeze_sha256':sha(original),'amendment_sha256':sha(amendment),'logical_requirements_sha256':sha(logical),'path_overrides':['ROOT','BASE','OUT'],'source_hash_scope':'Frozen physical file bytes; companion AST overrides only these three top-level path assignments, all function ASTs unchanged','unchanged_functions_ast_sha256':functions_hash,'python':sys.version,'platform':platform.platform(),'dependency_inventory':inventory(),'expected_original_output_sha256':original_scoring['output_sha256']}
 rp=out/'portable_reproduction_preflight.json';rp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
 os.environ['HF_HUB_OFFLINE']='1';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
 namespace={'__file__':str(source),'__name__':'frozen_encoder_v1_1_companion'}
 try:
  exec(compile(tree,str(source),'exec'),namespace)
  assert namespace['verify']()==sha(original)
  namespace['score_all']()
  actual=json.loads((out/'scoring_manifest.json').read_text());assert actual['status']=='complete'
  matches={name:actual['output_sha256'][name]==digest for name,digest in original_scoring['output_sha256'].items()}
  result={'completed_utc':stamp(),'status':'complete_exact_output_hash_match' if all(matches.values()) else 'complete_hash_differences_require_numerical_review','preflight_sha256':sha(rp),'scoring_manifest_sha256':sha(out/'scoring_manifest.json'),'original_output_hash_matches':matches,'scope':'cached-array scoring only; a fresh encoder or independently installed full environment was not run by this companion'}
 except BaseException as exc:
  result={'completed_utc':stamp(),'status':'failed','preflight_sha256':sha(rp),'error':repr(exc)}
  (out/'portable_reproduction_result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');raise
 (out/'portable_reproduction_result.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
