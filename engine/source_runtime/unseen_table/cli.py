"""Explicit independent v6 entry. Existing legacy CLI is delegated unchanged."""
import argparse
import json
from pathlib import Path
from .common import load, dump, fresh, guard_output

def main(argv=None):
    p = argparse.ArgumentParser(description='从示范学习陌生表，再由程序批量执行冻结技能')
    sub = p.add_subparsers(dest='command', required=True)
    skills = sub.add_parser('skills')
    skills.add_argument('--registry', default=str(Path(__file__).resolve().parents[1] / 'skills'))
    inspect = sub.add_parser('inspect')
    inspect.add_argument('csv')
    inspect.add_argument('--out', required=True)
    inspect.add_argument('--registry')
    inspect.add_argument('--context')
    prep = sub.add_parser('prepare-learning')
    prep.add_argument('csv')
    prep.add_argument('--out', required=True)
    prep.add_argument('--targets', nargs='+', required=True)
    prep.add_argument('--identity', nargs='+', required=True)
    prep.add_argument('--context', required=True)
    prep.add_argument('--skill-id', required=True)
    prep.add_argument('--groups-per-part', type=int, default=10)
    prep.add_argument('--seed', type=int, default=20260927)
    prep.add_argument('--visible-context', nargs='*', default=[])
    prep.add_argument('--source-field')
    learn = sub.add_parser('learn')
    learn.add_argument('--labels', required=True)
    learn.add_argument('--declaration', required=True)
    learn.add_argument('--context', required=True)
    learn.add_argument('--out', required=True)
    learn.add_argument('--offline', action='store_true')
    learn.add_argument('--require-quality', action='store_true')
    run = sub.add_parser('run')
    run.add_argument('csv')
    run.add_argument('--skill', required=True)
    run.add_argument('--context', required=True)
    run.add_argument('--out', required=True)
    run.add_argument('--groups', type=int, default=80)
    run.add_argument('--analyze-conflicts', action='store_true')
    run.add_argument('--allow-partial', action='store_true', help='未达开发目标时仅用于明确标注的诊断运行')
    for command in (learn, run):
        command.add_argument('--model-url')
        command.add_argument('--model')
        command.add_argument('--max-calls', type=int, default=12)
        command.add_argument('--wall-seconds', type=float, default=900)
        command.add_argument('--seed', type=int, default=20260927 if command is learn else None)
    score = sub.add_parser('score')
    score.add_argument('--run', required=True)
    score.add_argument('--reference', required=True)
    score.add_argument('--out', required=True)
    score.add_argument('--holdout')
    agg = sub.add_parser('aggregate')
    agg.add_argument('reports', nargs='+')
    agg.add_argument('--out', required=True)
    args = p.parse_args(argv)
    if args.command == 'skills':
        from .skill_library import inventory
        result = inventory(args.registry)
    elif args.command == 'inspect':
        from .inspect_table import inspect, match
        from .save_skill import list_skills
        result = inspect(args.csv)
        root = fresh(args.out)
        dump(root / 'profile.json', result)
        if args.registry:
            if not args.context:
                p.error('技能匹配需要来源范围context')
            dump(root / 'matching.json', match(result, list_skills(args.registry), load(args.context)))
    elif args.command == 'prepare-learning':
        from .inspect_table import prepare_learning
        result = prepare_learning(args.csv, args.out, args.targets, args.identity, load(args.context), args.skill_id, count=args.groups_per_part, seed=args.seed, visible=args.visible_context, source=args.source_field)
    elif args.command == 'learn':
        from .learn_skill import learn
        from .save_skill import save
        from .model_client import ModelClient
        if not 1 <= args.max_calls <= 12 or not 0 < args.wall_seconds <= 900:
            p.error('模型最多12次调用；学习与执行的时间上限900秒')
        guard_output(args.out)
        if Path(args.out).exists():
            p.error('学习输出目录已经存在')
        if args.offline and (args.model_url or args.model):
            p.error('离线学习不能配置模型')
        if not args.offline and (not args.model_url or not args.model):
            p.error('请配置实际模型，或明确使用--offline')
        client = None if args.offline else ModelClient(args.model_url, args.model, Path(args.out).with_name(Path(args.out).name + '_model_trace'), seed=args.seed, max_calls=args.max_calls, wall_seconds=args.wall_seconds)
        skill = learn(load(args.labels), load(args.declaration), load(args.context), client=client, seed=args.seed, wall_seconds=args.wall_seconds, work=Path(args.out).with_name(Path(args.out).name + '_learning_work'))
        skill['learning_summary']['model'] = getattr(client, 'model', None)
        saved = save(args.out, skill)
        result = {'skill': str(Path(args.out).absolute()), 'learning': saved['learning_summary'], 'validation': saved['validation_summary'], 'quality_gate': saved['quality_gate']}
        if args.require_quality and (not saved['quality_gate']['passed']):
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(3)
    elif args.command == 'run':
        from .save_skill import read
        from .execute_skill import initialize, run
        from .model_client import ModelClient
        skill = read(args.skill)
        if args.seed is not None and args.seed != skill['seed']:
            p.error('执行种子须与学习技能一致；不同种子的实验先建立对应技能')
        if (args.model_url or args.model) and (not args.analyze_conflicts):
            p.error('模型只用于代表性冲突分析；请明确使用--analyze-conflicts')
        if args.analyze_conflicts and (not args.model_url or not args.model):
            p.error('冲突分析需要模型地址和名称')
        remaining = max(0, 12 - (skill['learning_summary']['model_calls'] or 0))
        if not 1 <= args.max_calls <= 12 or not 0 < args.wall_seconds <= 900:
            p.error('模型最多12次调用；学习与执行的时间上限900秒')
        wall_remaining = max(0, min(args.wall_seconds, skill['learning_summary'].get('workflow_wall_limit', 900)) - skill['learning_summary'].get('seconds', 0))
        root = initialize(args.csv, skill, load(args.context), args.out, groups=args.groups, allow_partial=args.allow_partial)
        client = ModelClient(args.model_url, args.model, root / 'model_trace', seed=skill['seed'], max_calls=min(remaining, args.max_calls), wall_seconds=wall_remaining) if args.analyze_conflicts and remaining and wall_remaining and skill['quality_gate']['passed'] else None
        result = run(root, client=client, wall_seconds=wall_remaining)
    elif args.command == 'score':
        from .report_results import score
        result = score(args.run, args.reference, args.out, holdout_path=args.holdout)
    else:
        from .report_results import aggregate
        result = aggregate(args.reports, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
