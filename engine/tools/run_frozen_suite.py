"""Serial9B repair, feature comparisons, then independent frozen scoring."""
from pathlib import Path
import sys,subprocess,json,time,os
ROOT=Path(__file__).resolve().parents[1];WORKSPACE=ROOT.parent
HELPER='/root/fsas/home/anonymous/experiments/dataforge_v8_9b_v11Learning20261012/remote_probe.py'
BASELINES=ROOT/'baselines'


def main():
    status=ROOT/'research/frozen_suite_status.json'
    if status.exists():raise FileExistsError('Existing suite must be inspected; do not rerun')
    def update(**values):status.write_text(json.dumps({'pid':os.getpid(),'unix':time.time(),**values},indent=2)+'\n')
    def invoke(cmd,log,**state):
        update(phase='running',**state)
        with log.open('x') as f:subprocess.run([sys.executable,'-B',*map(str,cmd)],stdout=f,stderr=subprocess.STDOUT,check=True)
    update(phase='starting')
    try:
        for name in ['ionosphere','german']:
            run=ROOT/'work'/(name+'_fresh_20261013');skill=ROOT/'skills'/(name+'-learned')
            invoke([ROOT/'tools/benchmark_transfer.py','--base-skill',skill/'base_skill','--skill',skill,
                '--input',ROOT/'benchmarks'/name/'public_inputs/repair_evaluation.csv','--out',run,
                '--remote-helper',HELPER,'--tag','Fresh'+name.title()+'V1820261013'],
                ROOT/'research'/(name+'_online.log'),dataset=name,component='repair')
            if name=='ionosphere':continue # No class labels are model-visible in the original protocol.
            for condition,file in [('dirty','input.csv'),('pure','A_output.csv'),('system','B_output.csv')]:
                feature_input=run/file
                if condition!='dirty':
                    view=ROOT/'work'/(name+'_'+condition+'_classifier_view_20261013')
                    invoke([ROOT/'tools/build_classifier_view.py','--skill',skill,'--input',run/file,'--out',view],
                        ROOT/'research'/(name+'_'+condition+'_view.log'),dataset=name,component='training_CV_view',condition=condition)
                    feature_input=view/'classifier_training_view.csv'
                out=ROOT/'work'/(name+'_'+condition+'_features_20261013')
                invoke([BASELINES/('run_caafe_adapted_v4.py' if condition=='dirty' else 'run_features_grouped_f1.py'),'--input',feature_input,'--shared-summary',run/'shared_model_view.json',
                    '--out',out,'--remote-helper',HELPER,'--tag',name.title()+condition.title()+'FeaturesV1820261013','--iterations','5'],
                    ROOT/'research'/(name+'_'+condition+'_features.log'),dataset=name,component='feature_learning',condition=condition)
        # No formal reference/holdout is opened until EVERY trained output closes.
        for name in ['ionosphere','german']:
            run=ROOT/'work'/(name+'_fresh_20261013');private=ROOT/'benchmarks'/name/'local_private'
            invoke([ROOT/'tools/score_transfer.py','--run',run,'--reference',private/'repair_evaluation_reference.csv',
                '--holdout',private/'downstream_holdout_reference.csv','--report',ROOT/'reports'/(name+'_fresh_20261013')],
                ROOT/'research'/(name+'_score.log'),dataset=name,component='independent_repair_scoring')
            if name=='ionosphere':continue
            invoke([ROOT/'tools/score_classifier_views.py','--dataset',name],
                ROOT/'research'/(name+'_classifier_views_score.log'),dataset=name,component='independent_classifier_view_scoring')
            for condition in ['dirty','pure','system']:
                out=ROOT/'work'/(name+'_'+condition+'_features_20261013')
                invoke([BASELINES/('score_caafe_v4.py' if condition=='dirty' else 'score_features_grouped_f1.py'),'--run',out,'--holdout',private/'downstream_holdout_reference.csv',
                    '--report',ROOT/'reports'/(name+'_'+condition+'_features_20261013')],
                    ROOT/'research'/(name+'_'+condition+'_features_score.log'),dataset=name,component='independent_feature_scoring',condition=condition)
        update(phase='complete',datasets=['ionosphere','german'])
        print('ALL_FROZEN_FRESH_COMPARISONS_COMPLETE',flush=True)
    except BaseException as exc:
        update(phase='failed',error=str(exc));raise


if __name__=='__main__':main()
