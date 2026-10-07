"""Synthetic-only independent numeric checks. Does not read empirical outcomes."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('readout', Path(__file__).with_name('confirmatory_readout.py'))
readout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(readout)


class ReadoutTests(unittest.TestCase):
    def test_hand_calculated_one_extreme_loss(self):
        # Original scores 20..1: dropping score20 to0 moves one rank by19,
        # the other19 by1. Therefore Q=38/(20*19)=.1 and T=1/2.
        b=np.arange(20,0,-1,dtype=float)
        ids=np.array([f'id{i:02d}' for i in range(20)])
        sources=np.array(['ACL','EMNLP']*10)
        ctx=readout.baseline_context(b,ids,sources)
        z=b.copy();z[0]=0
        m, selected, bins=readout.mask_metrics(z,ctx)
        self.assertAlmostEqual(m['G'],2/np.sqrt(33.25))
        self.assertEqual(m['T'],.5)
        self.assertAlmostEqual(m['Q'],.1)
        self.assertAlmostEqual(m['spearman'],1-6*380/(20*(400-1)))
        self.assertEqual(m['far_loss_count_0.1'],1)
        self.assertEqual(m['far_loss_per_k_0.1'],.5)
        self.assertEqual(m['ACL_lost_count'],1)
        self.assertEqual(m['EMNLP_gained_count'],0)
        self.assertEqual(sum(selected),2)
        self.assertEqual(len(bins),20)
        self.assertTrue(any(n==0 and p is None for _,_,n,p in bins))

    def test_ties_never_manufacture_rank_motion(self):
        b=np.arange(100,0,-1,dtype=float);b[:3]=100
        ids=np.array([f'id{i:03d}' for i in range(100)])
        sources=np.array(['ACL','EMNLP']*50)
        ctx=readout.baseline_context(b,ids,sources)
        m,selected,bins=readout.mask_metrics(b.copy(),ctx)
        self.assertEqual((m['G'],m['T'],m['Q']),(0,0,0))
        self.assertEqual(len(readout.average_ranks(b)[:3]),3)
        self.assertTrue(np.all(readout.average_ranks(b)[:3]==98))
        self.assertAlmostEqual(m['spearman'],1)

    def test_cutoff_band_ordinal_centers_and_edge_ties(self):
        b=np.arange(100,0,-1,dtype=float)
        # Ordinal centers .095,.105 are positions10,11 (indices9,10).
        # Tie boundary scores outward, producing indices8..11 after expansion.
        b[8]=b[9];b[11]=b[10]
        ctx=readout.baseline_context(b,np.array([str(i) for i in range(100)]),np.array(['ACL','EMNLP']*50))
        self.assertEqual(np.flatnonzero(ctx['band']).tolist(),[8,9,10,11])
        self.assertEqual(ctx['cutoff']['rank_band_index_low'],9)
        self.assertEqual(ctx['cutoff']['rank_band_index_high'],10)
        # The signed-margin convention assigns exactzero to [0,.01).
        self.assertEqual(np.searchsorted(readout.EDGES,0,side='right')-1,5)

    def test_full_cohort_ordinal_positions(self):
        n=1557;b=np.arange(n,0,-1,dtype=float)
        ctx=readout.baseline_context(b,np.array([f'id{i:04}' for i in range(n)]),np.array(['ACL']*710+['EMNLP']*847))
        self.assertEqual(ctx['k'],156)
        self.assertEqual(ctx['cutoff']['rank_band_index_low'],140)
        self.assertEqual(ctx['cutoff']['rank_band_index_high'],170)
        self.assertEqual(ctx['cutoff']['rank_band_size'],31)

    def test_mask_level_sd_mcse_and_paired_order(self):
        #100zeros+100twos have mean1, samplevariance200/199.
        x=[0]*100+[2]*100
        s=readout.summarize(x,True)
        self.assertEqual(s['mean'],1)
        self.assertAlmostEqual(s['sd_over_masks'],np.sqrt(200/199))
        self.assertAlmostEqual(s['mcse'],1/np.sqrt(199))
        self.assertAlmostEqual(s['ci_low_raw'],1-readout.CRITICAL/np.sqrt(199))
        # Paired constant differences must have zeroSE even withvariablearms.
        paired=np.array(x)- (np.array(x)-.2)
        p=readout.summarize(paired.tolist(),True)
        self.assertAlmostEqual(p['mean'],.2)
        self.assertLess(p['mcse'],1e-16)
        with self.assertRaises(ValueError):readout.summarize(x[:-1],True)
        with self.assertRaises(ValueError):readout.summarize(x+[0],True)
        for invalid in [None,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):readout.summarize(x[:-1]+[invalid],True)
            with self.assertRaises(ValueError):readout.summarize([invalid]*200,True)

    def test_strict_thresholds_and_quantile(self):
        def interval(a,b):return {'ci_low_raw':a,'ci_high_raw':b}
        c=readout.classify(interval(.01,.1),interval(.06,.1),interval(.02,.03))
        self.assertEqual(c['aggregate'],'unresolved');self.assertFalse(c['joint_mean_rule'])
        c=readout.classify(interval(.01,.099),interval(.051,.1),interval(.011,.03))
        self.assertTrue(c['joint_mean_rule']);self.assertTrue(c['joint_with_rank_rule'])
        q=readout.verify_critical();self.assertEqual(q['family'],45)
        self.assertAlmostEqual(readout.wilson(0,200)[0],0)
        self.assertGreater(readout.wilson(0,200)[2],0)

    def test_undefined_and_zero_variance_block(self):
        with self.assertRaises(ValueError):readout.baseline_context([1,1],['a','b'],['ACL','EMNLP'])
        ctx=readout.baseline_context([2,1],['a','b'],['ACL','EMNLP'])
        with self.assertRaises(ValueError):readout.mask_metrics([2,np.nan],ctx)


if __name__=='__main__':unittest.main()
