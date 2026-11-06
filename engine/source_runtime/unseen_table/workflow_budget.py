"""One monotonic workflow budget and killable CPU workers; no client can reset it."""
import time, sys, subprocess, os, tempfile
from pathlib import Path
from .common import PACKAGE, fresh, dump, load

class WorkflowBudget:

    def __init__(self, seconds=900, work=None):
        if not 0 < seconds <= 900:
            raise ValueError('工作流时间上限须在900秒以内')
        self.started = time.monotonic()
        self.limit = seconds
        self.calls = 0
        self.work = fresh(work) if work else Path(tempfile.mkdtemp(prefix='dataforge-v6-learning-'))

    @property
    def elapsed(self):
        return time.monotonic() - self.started

    @property
    def remaining(self):
        return max(0.0, self.limit - self.elapsed)

    def check(self):
        if self.remaining <= 0:
            raise TimeoutError('全流程时间用完；保留工作记录，不能发布已完成技能')

    def save(self):
        dump(self.work / 'budget.json', {'calls': self.calls, 'max_calls': 12, 'elapsed': self.elapsed, 'limit': self.limit})

    def fit(self, records, declaration, seed, extra_rules, name):
        self.check()
        request = self.work / (name + '_request.json')
        result = self.work / (name + '_result.json')
        dump(request, {'action': 'fit', 'records': records, 'declaration': declaration, 'seed': seed, 'extra_rules': extra_rules})
        run_worker(request, result, self.remaining, self.work / (name + '.log'))
        self.save()
        return load(result)

    def observe(self, records, fitted, name):
        self.check()
        request = self.work / (name + '_request.json')
        result = self.work / (name + '_result.json')
        dump(request, {'action': 'check', 'rows': [r['dirty'] for r in records], 'fitted': fitted})
        run_worker(request, result, self.remaining, self.work / (name + '.log'))
        self.save()
        return load(result)

def run_worker(request, result, seconds, log):
    if seconds <= 0:
        raise TimeoutError('没有剩余执行时间')
    env = dict(os.environ)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONPATH'] = str(PACKAGE)
    with Path(log).open('xb') as stream:
        proc = subprocess.Popen([sys.executable, '-B', '-m', 'unseen_table.worker', str(request), str(result)], cwd=PACKAGE, env=env, stdout=stream, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise TimeoutError('本地训练或修复达到硬截止；子进程已终止，原材料保留')
    if code:
        raise RuntimeError(f'本地工作进程失败，请查看{log}')
    if not Path(result).is_file():
        raise RuntimeError('工作进程没有返回完整结果')

class ScopedClient:

    def __init__(self, client, budget, limit, scope):
        self.client = client
        self.budget = budget
        self.limit = limit
        self.scope = scope
        self.calls = 0

    def complete(self, system, payload):
        self.budget.check()
        if self.calls >= self.limit or self.budget.calls >= 12:
            raise RuntimeError('该折或全流程模型次数用完')
        before = getattr(self.client, 'calls', 0)
        if hasattr(self.client, 'deadline'):
            self.client.deadline = min(self.client.deadline, time.monotonic() + self.budget.remaining)
        try:
            return self.client.complete(system, {**payload, 'learning_scope': self.scope})
        finally:
            used = max(0, getattr(self.client, 'calls', before) - before)
            self.calls += used
            self.budget.calls += used
            self.budget.save()
