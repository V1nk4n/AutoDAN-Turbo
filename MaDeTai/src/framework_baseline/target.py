import re

_CODE_START = re.compile(
    r'(\n\s*(?:import|from|def|class)\s|\n\s*print\s*\(|"""\s*\n\s*(?:import|from|print))',
    re.MULTILINE,
)


class Target():
    '''
    Temporary implementation of Target class
    '''
    def __init__(self, model):
        self.model = model

    def _trim_at_stops(self, text: str, stop_sequences) -> str:
        if not text or not stop_sequences:
            return text
        earliest = len(text)
        for stop in stop_sequences:
            if not stop:
                continue
            idx = text.find(stop)
            if idx != -1 and idx < earliest:
                earliest = idx
        if earliest < len(text):
            return text[:earliest].rstrip()
        return text

    def _strip_code_drift(self, text: str) -> str:
        if not text:
            return text
        match = _CODE_START.search(text)
        if match:
            return text[:match.start()].rstrip()
        stripped = text.lstrip()
        if stripped.startswith('"""') and ('import ' in stripped or 'print(' in stripped):
            return ""
        return text

    def _generation_kwargs(self):
        cfg = getattr(self.model, 'config', {}) or {}
        temperature = float(cfg.get('temperature', 0.6))
        kwargs = {
            'do_sample': temperature > 0,
            'temperature': max(temperature, 1e-5),
            'top_p': float(cfg.get('top_p', 0.9)),
        }
        if not kwargs['do_sample']:
            kwargs.pop('temperature', None)
            kwargs.pop('top_p', None)
        if 'repetition_penalty' in cfg:
            kwargs['repetition_penalty'] = float(cfg['repetition_penalty'])
        if 'no_repeat_ngram_size' in cfg:
            kwargs['no_repeat_ngram_size'] = int(cfg['no_repeat_ngram_size'])
        max_new_tokens = int(cfg.get('max_new_tokens', 256))
        stops = list(cfg.get('stop_sequences', []) or [])
        prefix = cfg.get('instruct_prefix', '') or ''
        system_prompt = cfg.get('system_prompt', 'You are a helpful assistant.')
        if system_prompt is None:
            system_prompt = ''
        strip_code = bool(cfg.get('strip_code_drift', False) or prefix)
        return max_new_tokens, kwargs, stops, prefix, system_prompt, strip_code

    def respond(self, prompt):
        max_new_tokens, gen_kwargs, stops, prefix, system_prompt, strip_code = self._generation_kwargs()
        wrapped_prompt = f"{prefix}{prompt}" if prefix else prompt
        response = self.model.generate(system_prompt, wrapped_prompt, max_length=max_new_tokens, **gen_kwargs)
        response = self._trim_at_stops(response, stops)
        if strip_code:
            response = self._strip_code_drift(response)
        return response
