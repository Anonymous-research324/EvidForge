"""Public entry: profile, learn, native rule creation, repair, old reproduction."""
from pathlib import Path
import argparse,sys,shutil,subprocess
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT))
from learning.environment import RUNTIME
from unseen_table.common import load,dump,read_table,write_table,fresh,sha256


def main():
    p=argparse.ArgumentParser();sp=p.add_subparsers(dest='action',required=True)
    i=sp.add_parser('inspect');i.add_argument('--input',required=True);i.add_argument('--context')
    l=sp.add_parser('prepare');l.add_argument('--labels',required=True);l.add_argument('--declaration',required=True)
    l.add_argument('--context',required=True);l.add_argument('--out',required=True)
    n=sp.add_parser('learn');n.add_argument('--skill',required=True);n.add_argument('--out',required=True)
    n.add_argument('--codex',required=True);n.add_argument('--transport',required=True)
    n.add_argument('--tag',required=True)
    r=sp.add_parser('run');r.add_argument('--input',required=True);r.add_argument('--skill',required=True);r.add_argument('--out',required=True)
    x=sp.add_parser('legacy');x.add_argument('args',nargs=argparse.REMAINDER)
    a=p.parse_args()
    if a.action=='inspect':
        fields,rows=read_table(a.input)
        from application.catalog import match
        from unseen_table.common import number
        profile={c:{'nonblank':sum(bool(row[c].strip()) for row in rows),
            'numeric':sum(number(row[c]) is not None for row in rows)} for c in fields}
        print({'rows':len(rows),'columns':fields,'profile':profile,'skill_match':match(a.input,load(a.context) if a.context else None),
               'next_step':'Match confirmed source/schema to a saved skill; otherwise provide corrected20% learning/validation and protected field roles.'})
    elif a.action=='prepare':
        from learning.prepare import prepare
        prepare(load(a.labels),load(a.declaration),load(a.context),a.out)
    elif a.action=='learn':
        if not a.codex:raise ValueError('Existing CodexCLI path is required; no new agent framework is installed')
        subprocess.run([sys.executable,'-B',str(ROOT/'application/native_learning.py'),'--work',a.skill,'--out',a.out,
            '--codex',a.codex,'--remote-helper',a.transport,'--tag',a.tag],check=True)
    elif a.action=='run':
        from learning.execute import run
        from skill_runtime.package import open_skill
        skill=Path(a.skill);config=load(skill/'configuration.json')
        _,base,*_=open_skill(config['base_skill_path']);fields,rows=read_table(a.input)
        if set(fields)!=set(base['declaration']['fields']):raise ValueError('Exact field roles/schema must match the learned package')
        output=[dict(row) for row in rows];ds=run(rows,skill);root=fresh(a.out)
        for d in ds:output[d['row']][d['field']]=d['after']
        write_table(root/'input.csv',fields,rows);write_table(root/'output.csv',fields,output);dump(root/'decisions.json',ds)
        dump(root/'output_freeze.json',{'files':{n:sha256(root/n) for n in ['input.csv','output.csv','decisions.json']},
             'formal_reference_read':False,'automatic_deployment_qualification':'pending independent validation'})
    else:
        raise SystemExit(subprocess.run([sys.executable,'-B',str(ROOT/'legacy_entry.py'),*a.args]).returncode)


if __name__=='__main__':main()
