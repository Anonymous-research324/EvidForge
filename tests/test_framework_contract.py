"""Behavioral/scientific contracts of the paper system; no model calls or reference reads."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,sys,tempfile,unittest

R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'engine/source_runtime'))
from unseen_table.common import validate_declaration,equal,normalize_bundle,fingerprint
from unseen_table.features import value_features,detection_features,candidate_features
from unseen_table.candidate_pool import candidates,MAX_CANDIDATES

spec=importlib.util.spec_from_file_location('paper_arithmetic',R/'engine/learning/arithmetic.py')
arithmetic=importlib.util.module_from_spec(spec);spec.loader.exec_module(arithmetic)

def fixture():
    profile={'min':'0','max':'100','precision':{},'nonblank':5,'labeled':5,'value_groups':{},
             'numeric_sd':1.,'numeric_mean':10.,'numeric':True,'allow_blank':False,'learning_error_fraction':.2}
    fitted={'declaration':{'targets':['a','b']},'profiles':{'a':profile,'b':profile},'rules':[],
            'value_modes':{},'row_state':{},'relations':[],'source_consensus':{},'edit_patterns':[],
            'input_conditions':{},'training_cells':10}
    return profile,fitted

class FrameworkContractTests(unittest.TestCase):
    def test_fixed_core_matches_contract(self):
        contract=json.loads((R/'provenance/framework_contract.json').read_text())
        frozen=json.loads((R/'provenance/core_sha256.json').read_text())['files']
        self.assertEqual(len(frozen),124)
        for path,digest in frozen.items():self.assertEqual(hashlib.sha256((R/'engine'/path).read_bytes()).hexdigest(),digest,path)
        self.assertEqual(contract['dimensions'],{'single_value':22,'candidate':'d+78','error_detector':24,'modification_check':'d+85'})

    def test_evidence_dimensions_and_keep_candidate(self):
        profile,fitted=fixture();row={'a':'10','b':'20'}
        value=value_features('10',profile,{})
        self.assertEqual(len(value),22)
        pool=candidates(row,'a',fitted)
        self.assertTrue(any(c['kind']=='keep' and equal(c['value'],'10') for c in pool))
        self.assertLessEqual(len(pool),MAX_CANDIDATES)
        keep=next(c for c in pool if c['kind']=='keep')
        self.assertEqual(len(candidate_features(row,'a',keep,fitted,{})),len(fitted['declaration']['targets'])+78)
        self.assertEqual(len(detection_features(row,'a',fitted,{})),24)

    def test_expression_feature_boundary_and_unsafe_syntax(self):
        self.assertEqual(len(arithmetic.FIELDS-arithmetic.CONTEXT_FIELDS),18)
        self.assertEqual(len(arithmetic.CONTEXT_FIELDS),5)
        features={name:0. for name in arithmetic.FIELDS};features['rank_score']=.9
        self.assertEqual(arithmetic.Expression('rank_score + 0.1')(features),1.)
        for source in ["__import__('os')","open('reference.csv')","rank_score.real","[x for x in [1]]","row['a']","truth",'10**100']:
            with self.assertRaises((ValueError,SyntaxError),msg=source):arithmetic.Expression(source)
        with self.assertRaises(ValueError):arithmetic.Expression('1 / 0')(features)
        with self.assertRaises(ValueError):arithmetic.Expression('rank_score')({**features,'private_reference':1})

    def test_id_label_not_targets_or_context(self):
        schema={'fields':{'entity_id':{'identity':True},'a':{},'class':{'role':'label'}},
                'targets':['a'],'entity_fields':['entity_id'],'protected_fields':['class'],
                'visible_context_fields':[],'provenance':'synthetic contract fixture'}
        validate_declaration(schema,['entity_id','a','class'])
        for changes in [{'targets':['entity_id']},{'targets':['class']},{'visible_context_fields':['class']}]:
            with self.assertRaises(ValueError):validate_declaration({**schema,**changes},['entity_id','a','class'])

    def test_deployment_quality_boundary(self):
        # Execute the actual standalone native quality function without loading numeric dependencies.
        tree=ast.parse((R/'engine/application/catalog.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='quality')
        ns={};exec(compile(ast.Module(body=[function],type_ignores=[]),'<original-quality-contract>','exec'),ns)
        quality=ns['quality'];passing={'changes':5,'clean':10,'P':.98,'damage':.005}
        self.assertTrue(quality(passing))
        for changes in [{'changes':4},{'clean':9},{'P':.9799},{'damage':.00501},{'damage':None}]:
            self.assertFalse(quality({**passing,**changes}),changes)

    def test_score_gap_not_allowed_to_define_score(self):
        tree=ast.parse((R/'engine/learning/candidate_verifier.py').read_text())
        rank=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='rank')
        # Invalid scoring expressions are rejected before touching candidate frames.
        ns={};exec(compile(ast.Module(body=[rank],type_ignores=[]),'<original-rank-contract>','exec'),ns)
        with self.assertRaisesRegex(ValueError,'score_gap'):ns['rank']([],{'score_expression':'score_gap','accept_expression':'True'})

    def test_outer_validation_never_borrowed_for_learning(self):
        tree=ast.parse((R/'engine/workflow/onboarding.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='split_learning')
        ns={'validate_declaration':validate_declaration,'normalize_bundle':normalize_bundle,
            'fingerprint':fingerprint,'deepcopy':copy.deepcopy}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original-split-contract>','exec'),ns)
        schema={'fields':{'entity_id':{'identity':True},'a':{}},'targets':['a'],
                'entity_fields':['entity_id'],'visible_context_fields':[],'provenance':'synthetic contract fixture'}
        def group(i):
            return {'group':f'g{i}','provenance':'synthetic contract fixture',
                    'records':[{'row_key':f'row{i}','dirty':{'entity_id':f'id{i}','a':str(100+i)},'truth':{'a':str(i)}}]}
        bundle={'learning':[group(i) for i in range(20)],'validation':[group(i) for i in range(20,30)]}
        before=copy.deepcopy(bundle);internal,_=ns['split_learning'](bundle,schema)
        got={g['group'] for zone in internal.values() for g in zone}
        self.assertEqual(got,{g['group'] for g in bundle['learning']})
        self.assertFalse(got & {g['group'] for g in bundle['validation']})
        self.assertEqual(bundle,before)
        insufficient={'learning':bundle['learning'][:19],'validation':bundle['validation']+[group(31)]}
        with self.assertRaisesRegex(ValueError,'LEARNING'):ns['split_learning'](insufficient,schema)

    def test_no_generated_context_rule_is_exact_prior_path(self):
        tree=ast.parse((R/'engine/supplementary/execute.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);prior=root/'prior';prior.mkdir()
            records=[{'row':0,'field':'a','before':'12','after':'10'}]
            tables={'configuration.json':{'prior_skill_path':str(prior)},'manifest.json':{'files':{}},
                    'source_freeze.json':{'prior_manifest_sha256':'fixed','context_model_sha256':'fixed'}}
            def cannot_predict(*args):raise AssertionError('context prediction must not run without a compiled candidate')
            ns={'Path':Path,'load':lambda p:tables[Path(p).name],'sha256':lambda p:'fixed',
                'prior_run':lambda *a,**k:(copy.deepcopy(records),[],[]),'predict':cannot_predict}
            exec(compile(ast.Module(body=[fn],type_ignores=[]),'<original-context-contract>','exec'),ns)
            output=ns['run']([{'a':'12'}],root)
            self.assertEqual(output,[{**records[0],'prior_after':'10','context_rule_used':False}])

    def test_paper_sampling_settings_are_not_globally_overwritten(self):
        main=json.loads((R/'experiments/protocol.json').read_text())
        self.assertEqual(main['split'],[.1,.1,.64,.16]);self.assertEqual(main['temperature'],0)
        tab=json.loads((R/'experiments/paper_runs/tabclean_same9b/config.json').read_text())
        self.assertEqual(tab['seeds'],[2021,2022,2023]);self.assertEqual(tab['output_token_ceiling'],8192)
        self.assertEqual({k:tab['sampling'][k] for k in ['temperature','top_p','top_k','presence_penalty','thinking']},
                         {'temperature':.7,'top_p':.8,'top_k':20,'presence_penalty':1.5,'thinking':False})
        digit=json.loads((R/'experiments/paper_runs/llm_digit70/config.json').read_text())
        self.assertEqual(digit['seeds'],list(range(2013,2018)));self.assertEqual(digit['pollution_rate'],.7)
        new=json.loads((R/'experiments/paper_runs/new_errors/config.json').read_text())
        self.assertEqual(new['seeds'],[7013,7014,7015]);self.assertEqual(new['pollution']['rate'],.3)

if __name__=='__main__':unittest.main()
