"""Synthetic fixtures validate diagnostic code; they are never empirical results."""
import pathlib, sys, unittest
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'analysis'))
from revision_mask_intensity import stats,mask_from_arm,validate_pair,ROLES
from revision_neighbors import exact_winners,selected,source_accounting

class SyntheticMechanismTests(unittest.TestCase):
 def test_exact_ties_choose_lexical_historical_id(self):
  f=np.array([[.9,.9,.5]],dtype=np.float32);z,w,t,g=exact_winners(f,f,np.zeros(3,bool),['z','a','b'],True)
  self.assertEqual(w.tolist(),[1]);self.assertEqual(t.tolist(),[True]);self.assertEqual(g.tolist(),[0.])
 def test_same_identity_changed_representation(self):
  f=np.array([[.9,.4]],dtype=np.float32);title=np.array([[.8,.3]],dtype=np.float32)
  z,w,t,g=exact_winners(f,title,np.array([True,False]),['a','b'])
  self.assertEqual(w.tolist(),[0]);self.assertAlmostEqual(float(z[0]),.2,places=6)
 def test_nonwinner_title_can_win_and_lower_novelty(self):
  f=np.array([[.8,.7]],dtype=np.float32);title=np.array([[.6,.9]],dtype=np.float32)
  z,w,t,g=exact_winners(f,title,np.array([False,True]),['a','b'])
  self.assertEqual(w.tolist(),[1]);self.assertLess(float(z[0]),float(np.float32(1)-f.max()))
 def test_unmasked_baseline_winner_upper_bound_all_masks(self):
  rng=np.random.default_rng(31);f=rng.uniform(-1,1,(11,6)).astype('float32');t=rng.uniform(-1,1,(11,6)).astype('float32')
  b,bw,_,_=exact_winners(f,t,np.zeros(6,bool),list('abcdef'))
  for code in range(64):
   mask=np.array([bool(code&(1<<i)) for i in range(6)]);z,_,_,_=exact_winners(f,t,mask,list('abcdef'))
   self.assertTrue(np.all(z[~mask[bw]]<=b[~mask[bw]]))
 def test_focal_selection_ties_follow_lexical_id(self):
  self.assertEqual(selected([1,1,0],['z','a','b']).tolist(),[False,True,False])
 def test_fixed_denominator_recovers_shift_empty_cells_undefined(self):
  d=np.array([1.,2.,-1.,0.]);e=np.array([1,1,0,0],bool);s=np.array([0,1,1,0],bool);t=np.array([1,0,1,0],bool);sel=np.array([1,0,1,0],bool);lost=np.array([1,0,0,0],bool)
  rows=source_accounting(d,e,s,t,sel,lost,np.ones(4,bool))
  self.assertAlmostEqual(sum(r['shift_contribution_sd'] for r in rows),.5)
  self.assertEqual(sum(r['selection_losses_in_cell'] for r in rows),1)
  self.assertTrue(all(r['conditional_shift_sd'] is None for r in rows if r['cell_count']==0))
 def test_source_specific_fixed_denominator(self):
  d=np.array([1.,3.,8.]);false=np.zeros(3,bool)
  rows=source_accounting(d,false,false,false,false,false,np.array([1,1,0],bool))
  self.assertAlmostEqual(sum(r['shift_contribution_sd'] for r in rows),2.)
 def test_mask_rejects_duplicate_and_unknown_ids(self):
  history=[{'id':'a'},{'id':'b'}]
  for ids in [['a','a'],['x']]:
   with self.assertRaises(AssertionError):mask_from_arm(dict(missing_history_ids=ids,missing_count=len(ids)),history)
 def test_matched_year_source_cells(self):
  h=[dict(id='a',year=2018,source='A'),dict(id='b',year=2018,source='A'),dict(id='c',year=2018,source='B')]
  row=dict(mechanism='topic',assignment_id='synthetic',control_matching=['year','source'],control_cells=[dict(cell=dict(year=2018,source='A'),total=2,missing=1),dict(cell=dict(year=2018,source='B'),total=1,missing=0)])
  self.assertEqual(validate_pair(row,h,{ROLES[0]:np.array([1,0,0],bool),ROLES[1]:np.array([0,1,0],bool)}),2)
  with self.assertRaises(AssertionError):validate_pair(row,h,{ROLES[0]:np.array([1,0,0],bool),ROLES[1]:np.array([0,0,1],bool)})
 def test_sd_and_linear_quantile_convention(self):
  s=stats([0.,10.,20.]);self.assertEqual(s['assignment_sd'],10.);self.assertEqual(s['p05'],1.);self.assertEqual(s['p95'],19.)

if __name__=='__main__':unittest.main()
