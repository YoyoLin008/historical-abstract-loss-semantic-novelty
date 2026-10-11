"""Constrain reconstruction to this extraction and the installed Python dependencies.
This is a Python audit-hook guard, not a claim of operating-system sandbox isolation.
"""
import json,os,pathlib,runpy,sys,time
root=pathlib.Path(__file__).resolve().parent
allowed=[root,pathlib.Path(sys.prefix).resolve(),pathlib.Path(sys.base_prefix).resolve()]
blocked=[];opened=set();network_events=[]
def audit(event,args):
 if event.startswith('socket.') and event not in ['socket.__new__']:
  network_events.append(event);raise PermissionError('Network disabled for scalar-only check')
 if event in ['subprocess.Popen','os.system','os.exec','os.posix_spawn']:raise PermissionError('Subprocess disabled for scalar-only check')
 if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
  p=pathlib.Path(os.fsdecode(args[0])).resolve()
  if str(p) in ['/dev/null','/dev/urandom']:return
  if not any(p==a or a in p.parents for a in allowed):
   blocked.append(str(p));raise PermissionError('Read/write outside extracted package/runtime: '+str(p))
  opened.add(str(p))
sys.dont_write_bytecode=True
sys.addaudithook(audit)
mode=sys.argv[1];started=time.monotonic();status='PASS'
try:
 if mode=='reconstruct':
  sys.argv=['reconstruct_endpoints_from_scalar_scores.py','--package',str(root),'--output',str(root/'verification')]
  runpy.run_path(str(root/'reconstruct_endpoints_from_scalar_scores.py'),run_name='__main__')
 elif mode=='tests':
  sys.argv=['test_scalar_reconstruction.py','-v'];runpy.run_path(str(root/'tests/test_scalar_reconstruction.py'),run_name='__main__')
 else:raise ValueError(mode)
except SystemExit as e:
 if e.code not in [0,None]:status='FAILED_CHECK'
 raise
except BaseException:
 status='FAILED_CHECK';raise
finally:
 receipt=dict(status=status,mode=mode,guard='Python audit hooks plus -I isolated interpreter; allowed extraction and installed runtime/dependency directories only',blocked_attempts=blocked,network_events=network_events,files_opened=len(opened),elapsed_seconds=time.monotonic()-started,opened_files=sorted(opened),allowed_roots=[str(a) for a in allowed])
 (root/('isolation_'+mode+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
