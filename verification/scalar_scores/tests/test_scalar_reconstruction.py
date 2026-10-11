"""Synthetic code checks only; not empirical scientific evidence."""
import importlib.util,pathlib,tempfile,unittest
import numpy as np
path=pathlib.Path(__file__).resolve().parents[1]/'scripts/reconstruct_endpoints_from_scalar_scores.py'
if not path.exists():path=pathlib.Path(__file__).resolve().parents[1]/'reconstruct_endpoints_from_scalar_scores.py'
spec=importlib.util.spec_from_file_location('scalar_independent',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class ScalarSyntheticTests(unittest.TestCase):
 def context(self):return m.context(np.array([0.,1.,2.,3.,4.,5.,6.,7.,8.,9.]),list('jihgfedcba'),['A']*7+['B']*3,['A','B'])
 def test_average_rank_exact_ties(self):np.testing.assert_array_equal(m.ranks(np.array([4.,1.,1.,2.])),[3,.5,.5,2])
 def test_lexical_topk_tie(self):self.assertEqual(list(m.ordering([2.,2.,1.],['z','a','m'])),[1,0,2])
 def test_no_rounding_near_tie(self):self.assertEqual(list(m.ordering([2.,np.nextafter(2.,3.)],['a','z'])),[1,0])
 def test_duplicate_ids_rejected(self):
  with self.assertRaises(ValueError):m.context([1.,2.],['x','x'],['A','B'],['A','B'])
 def test_nonfinite_score_rejected(self):
  c=self.context();z=c['b'].copy();z[0]=np.nan
  with self.assertRaises(ValueError):m.endpoint(z,c)
 def test_incomplete_vector_rejected(self):
  with self.assertRaises(ValueError):m.endpoint([1.,2.],self.context())
 def test_source_alignment_rejected(self):
  with self.assertRaises(ValueError):m.context([1.,2.,3.],['x','y','z'],['A','B','C'],['A','B'])
 def test_restoration_zero_endpoints(self):
  c=self.context();v,lost,sel=m.endpoint(c['b'],c);self.assertEqual([v[k] for k in ['G','T','Q','mean_drift_sd']],[0.,0.,0.,0.]);self.assertFalse(lost.any())
 def test_weighted_source_drift(self):
  c=self.context();z=c['b']+np.where(c['A'],2.,0.);v,_,_=m.endpoint(z,c);self.assertAlmostEqual(v['mean_drift_sd'],1.4/c['sd']);self.assertNotAlmostEqual(v['mean_drift_sd'],1/c['sd'])
 def test_absolute_mean_distinction(self):self.assertNotEqual(np.mean(np.abs([-1.,1.])),abs(np.mean([-1.,1.])))
 def test_paired_before_variance(self):
  x=np.arange(200,dtype=float);s=m.summarize(x-x,3.);self.assertEqual(s['sd'],0);self.assertGreater(x.std(),0)
 def test_lossless_nonpickle_reload(self):
  z=np.array([0.12345678901234568,np.nextafter(.5,1.)],dtype=np.float64)
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'a.npz';np.savez_compressed(p,scores=z)
   with np.load(p,allow_pickle=False) as f:np.testing.assert_array_equal(f['scores'],z);self.assertFalse(f['scores'].dtype.hasobject)
 def test_cutoff_tie_expands_band(self):
  b=np.arange(100,dtype=float);b[88:93]=90.;c=m.context(b,[f'{i:03d}' for i in range(100)],['A']*50+['B']*50,['A','B']);self.assertEqual(int(c['band'].sum()),5)
 def test_empty_bins_undefined(self):
  den=np.array([2,0]);num=np.array([1,0]);p=np.divide(num,den,out=np.full(2,np.nan),where=den>0);self.assertTrue(np.isnan(p[1]));self.assertEqual(np.nansum(p*den),1)
 def test_unit_count_rejected(self):
  with self.assertRaises(ValueError):m.summarize([1.,2.],3.)
if __name__=='__main__':unittest.main()
