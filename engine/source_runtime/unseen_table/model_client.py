"""Audited bounded requests; conservative input limit, wall time, explicit missing usage."""
import json
import time
import urllib.request
from .common import fresh, dump

def input_size(system, payload):
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}]
    return len(json.dumps(messages, ensure_ascii=False).encode()) + 128

class ModelClient:

    def __init__(self, url, model, trace, *, seed=20260927, max_calls=12, max_input_bytes=6000, max_output_tokens=1500, wall_seconds=900, timeout=75):
        if not 1 <= max_calls <= 12 or not 1 <= max_output_tokens <= 1500 or (not 1 <= max_input_bytes <= 6000):
            raise ValueError('模型预算超过固定上限')
        if not 0 < wall_seconds <= 900:
            raise ValueError('模型时间预算必须在900秒以内')
        self.url = url.rstrip('/') + '/chat/completions' if url.rstrip('/').endswith('/v1') else url
        if not self.url.endswith('/chat/completions'):
            raise ValueError('需要兼容模型接口地址')
        self.model = model
        self.trace = fresh(trace)
        self.seed = seed
        self.max_calls = max_calls
        self.max_input_bytes = max_input_bytes
        self.max_output_tokens = max_output_tokens
        self.deadline = time.monotonic() + wall_seconds
        self.timeout = timeout
        self.calls = 0
        self.tokens = 0
        self.unknown_usage_calls = []

    def complete(self, system, payload):
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}]
        size = input_size(system, payload)
        if size > self.max_input_bytes:
            raise ValueError('代表性请求仍过长；需要减少示范，不发送整批格子')
        if self.calls >= self.max_calls or time.monotonic() >= self.deadline:
            raise RuntimeError('固定模型调用或时间预算已用完')
        self.calls += 1
        stem = f'{self.calls:04d}'
        body = {'model': self.model, 'messages': messages, 'seed': self.seed, 'temperature': 0, 'max_tokens': self.max_output_tokens, 'response_format': {'type': 'json_object'}, 'chat_template_kwargs': {'enable_thinking': False}}
        dump(self.trace / f'{stem}_request.json', body)
        started = time.monotonic()
        measured = False
        try:
            request = urllib.request.Request(self.url, json.dumps(body).encode(), {'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=max(0.1, min(self.timeout, self.deadline - time.monotonic()))) as response:
                result = json.load(response)
            dump(self.trace / f'{stem}_response.json', result)
            usage = result.get('usage', {}).get('total_tokens')
            if type(usage) is not int or usage < 0:
                raise ValueError('接口未提供有效用量')
            self.tokens += usage
            measured = True
            if result['choices'][0]['finish_reason'] != 'stop':
                raise ValueError('回答被截断，不能视为成功学习')
            answer = json.loads(result['choices'][0]['message']['content'])
            if not isinstance(answer, dict):
                raise ValueError('回答必须为JSON对象')
            return answer
        except Exception as error:
            if not measured:
                self.unknown_usage_calls.append(self.calls)
            dump(self.trace / f'{stem}_error.json', {'error': str(error)})
            raise
        finally:
            dump(self.trace / f'{stem}_timing.json', {'seconds': time.monotonic() - started, 'input_upper_bound_bytes': size})
            dump(self.trace / 'usage.json', {'calls': self.calls, 'tokens': self.tokens, 'unknown_usage_calls': self.unknown_usage_calls})
