"""Synthetic fixtures only; no fixture is an empirical result."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

SPEC = importlib.util.spec_from_file_location('revision_descriptive',Path(__file__).parents[1]/'analysis/revision_descriptive_outputs.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)

class RevisionDescriptiveTests(unittest.TestCase):
    def test_weighted_source_shift_algebra(self):
        d=np.array([.3,-.2,.05]);drift=np.array([.1,.2,-.1]);w=.2
        a,b=m.source_shifts(d,drift,w)
        np.testing.assert_allclose(a-b,d,atol=1e-15)
        np.testing.assert_allclose(w*a+(1-w)*b,drift,atol=1e-15)
        self.assertFalse(np.allclose((a+b)/2,drift))

    def test_mean_absolute_not_absolute_mean(self):
        d=np.array([-.5,.5]); self.assertEqual(np.abs(d).mean(),.5)
        self.assertEqual(abs(d.mean()),0.)

    def test_intervention_policy_and_duplicate_mixed_rows(self):
        data=[dict(policy=p,condition='random',rate=.25,metric='G') for p in ['all_title','restored','complete_case','mixed','restored']]
        self.assertEqual(len(m.mixed_only(data)),1)
        with self.assertRaises(ValueError):m.mixed_only(data+[data[3]])

    def test_source_projection_dedup_requires_consistency(self):
        a=[dict(record_id='a',source='X'),dict(record_id='a',source='X'),dict(record_id='b',source='Y')]
        self.assertEqual(m.one_source_map(a),{'a':'X','b':'Y'})
        with self.assertRaises(ValueError):m.one_source_map(a+[dict(record_id='a',source='Y')])

    def test_exact_id_alignment_and_order(self):
        ids,a,b=m.aligned_scores([dict(paper_id='b',score=2),dict(paper_id='a',score=1)], [dict(paper_id='a',score=10),dict(paper_id='b',score=20)])
        self.assertEqual(ids,['a','b']); np.testing.assert_array_equal(a,[1,2]);np.testing.assert_array_equal(b,[10,20])

    def test_duplicate_and_dropped_ids_rejected(self):
        a=[dict(paper_id='a',score=1),dict(paper_id='b',score=2)]
        with self.assertRaises(ValueError):m.aligned_scores(a,a[:1])
        with self.assertRaises(ValueError):m.aligned_scores(a,[a[0],a[0]])

    def test_topk_exact_tie_lexical_id(self):
        ids=['z','a','c','d','e','f','g','h','i','j']
        self.assertEqual(m.selected([2,2,1,1,1,1,1,1,1,1],ids),{'a'})

    def test_scores_never_rounded_before_sort(self):
        ids=['a','z','c','d','e','f','g','h','i','j']
        x=[1,1+1e-12,0,0,0,0,0,0,0,0]
        self.assertEqual(m.selected(x,ids),{'z'})
        self.assertGreater(m.average_ranks(x)[1],m.average_ranks(x)[0])

    def test_average_ranks_and_tie_correct_correlation(self):
        x=[1,1,2,3];y=[1,2,2,3]
        np.testing.assert_array_equal(m.average_ranks(x),[1.5,1.5,3,4])
        rho,q,_,_=m.rank_agreement(x,y)
        self.assertAlmostEqual(rho,float(np.corrcoef([1.5,1.5,3,4],[1,2.5,2.5,4])[0,1]))
        self.assertAlmostEqual(q,.5/3)

    def test_constant_rank_correlation_undefined(self):
        self.assertIsNone(m.rank_agreement([1,1,1],[1,2,3])[0])

    def test_capacity_partition_and_stronger_mean_bound(self):
        base=set('abcde');band=set('abx')
        self.assertEqual(m.capacity(set('acd'),base,band),(3,1,2,1))
        lo=[m.capacity(s,base,band)[3] for s in [set('a'),set('abcde')]]
        self.assertGreater(np.mean(lo),max(0,(1+5)/2-2))
        with self.assertRaises(ValueError):m.capacity({'x'},base,band)

    def test_empirical_style_nesting_and_nonnested_fixture(self):
        n=m.nested_sets(set('abcd'),set('abc'));self.assertTrue(n['margin_subset_outside']);self.assertEqual(n['outside_only_n'],1)
        n=m.nested_sets(set('abc'),set('abcd'));self.assertTrue(n['outside_subset_margin']);self.assertEqual(n['margin_only_n'],1)
        n=m.nested_sets(set('ab'),set('bc'));self.assertFalse(n['outside_subset_margin']);self.assertFalse(n['margin_subset_outside'])

    def test_bin_edges_zero_and_left_closed(self):
        np.testing.assert_array_equal(m.bins_for([-.1,-.05,-.025,-.01,0,.01,.025,.05,.1]),range(1,10))
        self.assertEqual(m.bins_for([-.00001])[0],4)

    def test_empty_bin_undefined(self):
        self.assertIsNone(m.conditional([0,0],0))

    def test_denominator_weighted_recovery(self):
        losses=np.array([[0,2],[1,3],[2,0]])
        den=[2,4]
        p=[m.conditional(losses[:,i],den[i]) for i in range(2)]
        self.assertAlmostEqual(sum(d*v for d,v in zip(den,p)),losses.sum(axis=1).mean())

    def test_cross_encoder_topk_overlap_and_source_totals(self):
        ids=np.array([f'{i:02d}' for i in range(20)])
        x=np.arange(20,dtype=float);y=x.copy();y[0]=21
        a=m.selected(x,ids);b=m.selected(y,ids)
        self.assertEqual(len(a),2);self.assertEqual(len(b),2);self.assertEqual(len(a&b),1)
        src={i:('X' if int(i)%2 else 'Y') for i in ids}
        self.assertEqual(sum(sum(src[i]==s for i in b) for s in ['X','Y']),2)

    def test_nonfinite_and_duplicate_selection_input_rejected(self):
        with self.assertRaises(ValueError):m.selected([1,np.nan],['a','b'])
        with self.assertRaises(ValueError):m.selected([1,2],['a','a'])

if __name__=='__main__': unittest.main()
