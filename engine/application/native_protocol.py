"""Close on the measured optimum among eligible programs, without a loop.

The previous backend already required the highest inner-source F1. This
adapter narrows the JSON tool choices to that same permitted set. It changes
no generated rules, source parameters, quality gates or scoring equations.
"""
from application.config_protocol import Protocol as PreviousProtocol,state,classify


class Protocol(PreviousProtocol):
    def payload(self):
        request=super().payload();current=state(self.work)
        valid=[q for q in current['programs'] if q.get('eligible_for_research_compile')]
        if valid:
            maximum=max(q['metrics']['F1'] for q in valid)
            choices=[q['program_id'] for q in valid if q['metrics']['F1']>=maximum-1e-12]
            schema=request['structured_outputs']['json']
            for branch in schema.get('anyOf',[schema]):
                if branch['properties']['action']['enum']==['compile']:
                    branch['properties']['program_id']['enum']=choices
        return request
