"""Migration contract tests; legacy originals are recovered from their recorded Git revision."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from . import budgets, physical, ttt, diagnostics
from .schema import rows
from .statistics import bootstrap
from .maintenance import WORKSPACE

REVISION='97c6c67addaa728cb479c09dd2de1ffa105ea5d3'


class MigrationTests(unittest.TestCase):
    def test_original_statistics_exact(self):
        cases=[('analyze_dynamics_only_ttt',budgets,'stats'),('refine_severity_analysis',physical,'stats'),
               ('analyze_ttt_distribution',ttt,'quantiles'),('analyze_ttt_distribution',ttt,'stream_ci')]
        with tempfile.TemporaryDirectory() as tmp:
            for old,new,fn in cases:
                source=subprocess.check_output(['git','show',f'{REVISION}:tdmpc2/experiments/nsgym/scripts/{old}.py'],cwd=WORKSPACE)
                p=Path(tmp)/(old+'.py');p.write_bytes(source)
                spec=importlib.util.spec_from_file_location('original_'+old,p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
                for values in ([1.],[-2.,0.,9.],[1e-9,-1e-5,0.,1.,3.,100.]):
                    actual=getattr(new,fn)(values);expected=getattr(module,fn)(values)
                    if isinstance(actual,dict):
                        self.assertEqual(actual.keys(),expected.keys())
                        actual=list(actual.values());expected=list(expected.values())
                    np.testing.assert_allclose(actual,expected,rtol=0,atol=0,equal_nan=True) if all(x is not None for x in actual) else self.assertEqual(actual,expected)
    def test_bootstrap_rejects_nonfinite(self):
        for values in ([],[float('nan')],[1,float('inf')]):
            with self.assertRaises(ValueError):bootstrap(values)
    def test_required_operations_and_registration(self):
        from .__main__ import OPERATIONS
        self.assertEqual(len(OPERATIONS),11)
        from .sources import SOURCES
        self.assertEqual(set(SOURCES),{'halfcheetah','reacher'})
        self.assertTrue(all(x['adapter'] in SOURCES for x in SOURCES.values()))
    def test_diagnostic_trace_from_saved_updates(self):
        base=WORKSPACE/'results/nsgym/dynamics-q-750k-v1/followup-analysis-v1'
        trace=list(rows(base/'all_update_diagnostics.csv'))
        grouped={}
        for r in trace:
            key=diagnostics.key(r);converted={}
            for k,v in r.items():
                try:converted[k]=float(v)
                except ValueError:converted[k]=v
            grouped.setdefault(key,[]).append(converted)
        self.assertEqual(len(grouped),36)
        for values in grouped.values():
            self.assertEqual(len(values),118)
            self.assertEqual(diagnostics.trace_summary(values)['updates'],118)

if __name__=='__main__':unittest.main()
