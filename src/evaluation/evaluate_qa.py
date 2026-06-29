import os
import re
import sys
import json
import argparse
from tqdm import tqdm
import backoff
import openai
from openai import OpenAI
import numpy as np


model_zoo = {
    'llama-3.1-70b-instruct': ('meta-llama/Meta-Llama-3.1-70B-Instruct', 'local'),
    'gpt-4o-mini': ('gpt-4o-mini-2024-07-18', 'openai'),
    'gpt-4o': ('gpt-4o-2024-08-06', 'openai'),
    'gpt-5': ('gpt-5', 'openai'),
}


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError))
def chat_completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


_SEMANTIC = (
    " Judge by meaning, not exact wording: a paraphrase or different vocabulary"
    " conveying the same fact is correct, and a response that states the answer"
    " more specifically or more precisely is correct. A response that gives the"
    " correct answer plus extra correct detail is correct unless the extra detail"
    " is wrong. A response that omits the required fact, gives only a subset of"
    " it, or contradicts it, is incorrect."
)


def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if abstention:
        template = ("I will give you an unanswerable question, an explanation, and a response"
                    " from a model. Answer yes if the model identifies the question as"
                    " unanswerable — saying the information is incomplete or that the asked"
                    " information is not available counts."
                    "\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
                    "Does the model correctly identify the question as unanswerable? Answer yes or no only.")
        return template.format(question, answer, response)

    if task in ('single-session-user', 'single-session-assistant', 'multi-session'):
        template = ("I will give you a question, a correct answer, and a response from a model."
                    " Answer yes if the response conveys the correct answer." + _SEMANTIC +
                    "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                    "Is the model response correct? Answer yes or no only.")
    elif task == 'temporal-reasoning':
        template = ("I will give you a question, a correct answer, and a response from a model."
                    " Answer yes if the response conveys the correct answer. Do not penalize"
                    " off-by-one errors in a count of days/weeks/months." + _SEMANTIC +
                    "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                    "Is the model response correct? Answer yes or no only.")
    elif task == 'knowledge-update':
        template = ("I will give you a question, a correct answer, and a response from a model."
                    " Answer yes if the response gives the correct, updated answer; mentioning"
                    " the earlier or outdated value alongside it is fine as long as the updated"
                    " answer is present." + _SEMANTIC +
                    "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                    "Is the model response correct? Answer yes or no only.")
    elif task == 'single-session-preference':
        template = ("I will give you a question, a rubric for the desired personalized response,"
                    " and a response from a model. Answer yes if the response recalls and uses"
                    " the user's personal information correctly; it need not cover every point in"
                    " the rubric."
                    "\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
                    "Is the model response correct? Answer yes or no only.")
    else:
        raise NotImplementedError
    return template.format(question, answer, response)


def judge(client, model, prompt):
    kwargs = dict(client=client, model=model, n=1,
                  messages=[{"role": "user", "content": prompt}])
    # Reasoning models (gpt-5, o-series) reject temperature/max_tokens and need
    # headroom for reasoning tokens; non-reasoning models stay byte-identical.
    if not re.match(r"^(gpt-5|o[1-9])", model):
        kwargs.update(temperature=0, max_tokens=10)
    completion = chat_completions_with_backoff(**kwargs)
    return 'yes' in (completion.choices[0].message.content or '').strip().lower()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='LongMemEval QA judge.')
    ap.add_argument('metric_model', help='judge model: ' + ', '.join(model_zoo))
    ap.add_argument('hyp_file', help='hypotheses JSONL ({question_id, hypothesis} per line)')
    ap.add_argument('ref_file', help='reference dataset JSON (question_id, question, answer, question_type)')
    args = ap.parse_args()

    if args.metric_model not in model_zoo:
        print('Requested metric model is not supported:', args.metric_model)
        sys.exit(1)
    metric_model, source = model_zoo[args.metric_model]
    if source == 'openai':
        # Pass organization into the v1 client constructor; a module-level
        # openai.organization is not consulted by an explicit OpenAI(...), so
        # org-scoped keys would otherwise be ignored.
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'),
                        organization=os.getenv('OPENAI_ORGANIZATION'))
    else:
        client = OpenAI(api_key='EMPTY', base_url='http://localhost:8001/v1')

    def _load_records(path):
        """A LongMemEval file is either a JSON array or one JSON object per
        line. Read it fully (closing the handle), then parse whichever shape it
        is; blank lines are skipped."""
        with open(path) as f:
            text = f.read()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return [json.loads(ln) for ln in text.splitlines() if ln.strip()]

    hypotheses = _load_records(args.hyp_file)
    references = _load_records(args.ref_file)
    qid2qdata = {e['question_id']: e for e in references}
    qid2qtype = {e['question_id']: e['question_type'] for e in references}

    qtype2acc = {t: [] for t in set(qid2qtype.values())}
    overall = []
    result_file = '{}.eval-results-{}'.format(args.hyp_file, args.metric_model)
    with open(result_file, 'w') as out_f:
        for entry in tqdm(hypotheses):
            qid = entry['question_id']
            if qid not in qid2qtype:
                continue
            qtype = qid2qtype[qid]
            prompt = get_anscheck_prompt(
                qtype, qid2qdata[qid]['question'], qid2qdata[qid]['answer'],
                entry['hypothesis'], abstention='_abs' in qid,
            )
            label = judge(client, metric_model, prompt)
            entry = {**entry, 'autoeval_label': {'model': metric_model, 'label': label}}
            print(json.dumps(entry), file=out_f)
            qtype2acc[qtype].append(1 if label else 0)
            overall.append(1 if label else 0)

    if not overall:
        sys.exit('No hypotheses were evaluated: every entry was skipped (check '
                 'that the hypothesis question_ids match the reference file). '
                 'Refusing to report nan.')
    print('Accuracy:', round(float(np.mean(overall)), 4))
    for k, v in sorted(qtype2acc.items()):
        if v:
            print('\t{}: {} ({})'.format(k, round(float(np.mean(v)), 4), len(v)))
    print('Saved to', result_file)
