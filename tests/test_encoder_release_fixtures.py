"""Tiny independent mathematical fixtures; no encoder, real scoring or frozen-file mutation."""
import ast,math,unittest,json,os
from pathlib import Path
import numpy as np
SOURCE=Path(os.environ.get('ENC_REVIEW_SOURCE',str(Path(__file__).resolve().parents[1]/'src/post_confirmation/encoder/encoder_check_v1_1.py')))
tree=ast.parse(SOURCE.read_text())
functions={n.name:n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
# Compile only the three relevant function bodies. Do not import the module's
# filesystem setup or call score_all(), load_model(), encode() or verify().
namespace={'np':np}
for name in ['ranks','selected','outcomes']:
 code=ast.Module(body=[functions[name]],type_ignores=[])
 exec(compile(ast.fix_missing_locations(code),str(SOURCE),'exec'),namespace)

class EndpointFixtures(unittest.TestCase):
 def setUp(self):
  self.ids=['b','z','a','c']
  self.b=np.array([0.,1.,1.,3.])
  self.sd=math.sqrt(19)/4  # exact population SD of [0,1,1,3]
  namespace.update(fi=np.arange(4),ids=self.ids,k=2,old={2,3},A=np.array([True,True,False,False]),Bmask=np.array([False,False,True,True]),contrast=-1.5,sd=self.sd,br=np.array([0.,1.5,1.5,3.]),band=np.array([False,True,True,False]),margin=(self.b-1)/self.sd)
 def test_average_ranks_exact_ties_do_not_use_ids(self):
  np.testing.assert_array_equal(namespace['ranks'](self.b),[0.,1.5,1.5,3.])
  np.testing.assert_array_equal(namespace['ranks'](np.array([7.,7.,7.,7.])),[1.5]*4)
  np.testing.assert_array_equal(namespace['ranks'](np.array([3.,1.,1.,0.])),[3.,1.5,1.5,0.])
 def test_fixed_size_ties_select_lexical_id(self):
  self.assertEqual(namespace['selected'](self.b,self.ids,2),{2,3})
  self.assertEqual(namespace['selected'](np.array([1.,1.,1.,1.]),self.ids,2),{0,2})
  self.assertEqual(namespace['selected'](self.b,['b','a','z','c'],2),{1,3})
 def test_unchanged_scores_zero_all_outcomes(self):
  got=namespace['outcomes'](self.b)
  self.assertEqual(got,dict(G=0.,T=0.,Q=0.,selected_lost_count=0,outside_band_lost_count=0,far_0_1_sd_lost_count=0))
 def test_hand_calculated_source_reversal_tie_and_denominators(self):
  got=namespace['outcomes'](np.array([3.,1.,1.,0.]))
  self.assertAlmostEqual(got['G'],12/math.sqrt(19),14)
  self.assertEqual(got['T'],.5)  # 1 lost out of k=2, not 1/4 focal papers
  self.assertEqual(got['Q'],.5)  # (3+0+0+3)/(4*(4-1))
  self.assertEqual(got['selected_lost_count'],1)
  self.assertEqual(got['outside_band_lost_count'],1)
  self.assertEqual(got['far_0_1_sd_lost_count'],1)
 def test_common_score_translation_preserves_contrast_ranks_selection(self):
  got=namespace['outcomes'](self.b+10)
  self.assertEqual(got['G'],0.);self.assertEqual(got['T'],0.);self.assertEqual(got['Q'],0.)
 def test_far_margin_and_rank_band_are_not_nested(self):
  # Place the lost high-margin selection inside the rank band: the two
  # qualification counts must differ rather than imposing one as a filter.
  namespace['band']=np.array([False,True,True,True])
  got=namespace['outcomes'](np.array([3.,1.,1.,0.]))
  self.assertEqual(got['outside_band_lost_count'],0)
  self.assertEqual(got['far_0_1_sd_lost_count'],1)

 def test_changed_outcomes_are_json_serializable(self):
  got=namespace['outcomes'](np.array([3.,1.,1.,0.]))
  json.dumps(got,allow_nan=False)

 def test_baseline_band_count_is_json_serializable(self):
  nodes=[n.value for n in ast.walk(tree) if isinstance(n,ast.keyword) and n.arg=='baseline_selected_in_band']
  self.assertEqual(len(nodes),1)
  expression=ast.Expression(body=nodes[0])
  count=eval(compile(ast.fix_missing_locations(expression),str(SOURCE),'eval'),namespace)
  self.assertEqual(count,1)
  json.dumps({'baseline_selected_in_band':count},allow_nan=False)

if __name__=='__main__':unittest.main(verbosity=2)
