"""Authoritative read-only NS-Gym analysis interface; no model/environment execution."""
import argparse
import importlib
from pathlib import Path
import sys
from .schema import read_required
from . import sources

OPERATIONS=('inventory','validate','nominal','physical','frozen','adaptation','q_distribution',
            'planning','cross_task','full','cleanup_preview')
FORMATS={'ttt':'ttt','budgets':'budgets','severity':'physical','reacher':'reacher','diagnostics':'diagnostics'}


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    # Accept task=x / method=x notation without introducing a second config system.
    argv=[('--'+x.replace('method=','methods=',1).replace('=', ' ',1)).split() if x.startswith(('task=','method=')) else [x] for x in argv]
    argv=[v for part in argv for v in part]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=OPERATIONS)
    parser.add_argument('--format',choices=tuple(FORMATS),help='Historical raw-source adapter; arguments after -- go to the migrated adapter')
    parser.add_argument('--task',default='all')
    parser.add_argument('--methods','--method',default='frozen,dynamics_only,dynamics_q,joint')
    parser.add_argument('--results-root','--source-root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    parser.add_argument('--no-plots',action='store_true')
    parser.add_argument('--unified-output',type=Path,help='Also regress an existing unified comparisons.csv against original condition summaries')
    if '--' in argv:
        split=argv.index('--');adapter_args=argv[split+1:];argv=argv[:split]
    else:adapter_args=[]
    a, unknown=parser.parse_known_args(argv)
    if unknown and not a.format:parser.error('Unrecognized arguments: '+str(unknown))
    adapter_args=unknown+adapter_args
    config=read_required(a.config)
    sources.SOURCES.clear();sources.SOURCES.update(config['sources'])
    if a.task!='all' and a.task not in sources.SOURCES:parser.error('Task not registered in config.sources')
    if not set(a.methods.split(',')) <= set(config['methods']):parser.error('Unregistered method')
    if a.format:
        module=importlib.import_module('.'+FORMATS[a.format],__package__)
        contracts=config.get('historical_adapters', {})
        if a.format=='severity' and 'physical' in contracts:
            c=contracts['physical'];module.PARAMS=tuple(c['parameters']);module.LABELS=c['labels']
            module.SEVERITIES=tuple(c['severities']);module.PRIMARY_SHA256=c['checkpoint_sha256']
            module.COLORS=dict(zip(module.SEVERITIES,module.plt.get_cmap('viridis')(module.np.linspace(.08,.9,len(module.SEVERITIES)))))
        if a.format=='reacher' and 'reacher' in contracts:
            module.FAMILIES=contracts['reacher']['families'];module.GRID=contracts['reacher']['severities']
        previous=sys.argv
        try:
            sys.argv=[f'analysis {a.operation} --format {a.format}',*adapter_args,'--output',str(a.output)]
            module.main()
        finally:sys.argv=previous
        return
    if adapter_args:parser.error('Arguments after -- require --format')
    from . import engine
    results=(a.results_root or engine.DEFAULT_RESULTS).resolve()
    if a.operation=='validate':
        from .regression import run
        if not run(results,a.output.resolve(),a.unified_output):raise SystemExit(1)
        return
    if a.operation=='cleanup_preview':
        from .maintenance import cleanup_preview
        a.output.mkdir(parents=True,exist_ok=False);cleanup_preview(results,a.output);return
    arguments=['--results-root',str(results),'--output',str(a.output),'--task',a.task,
               '--methods',a.methods,'--config',str(a.config)]
    if a.no_plots:arguments+=['--no-plots']
    if a.operation=='inventory':arguments+=['--command','inventory']
    else:arguments+=['--analysis',a.operation]
    engine.main(arguments)


if __name__=='__main__':main()
