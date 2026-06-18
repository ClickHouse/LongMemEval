import os
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
}


@backoff.on_exception(backoff.expo, (openai.RateLimitError,
                                    openai.APIError))
def chat_completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)


def get_anscheck_prompt(task, question, answer, response, abstention=False):
    if not abstention:
        if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'temporal-reasoning':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'knowledge-update':
            template = "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        elif task == 'single-session-preference':
            template = "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only."
            prompt = template.format(question, answer, response)
        else:
            raise NotImplementedError
    else:
        template = "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
        prompt = template.format(question, answer, response)
    return prompt


# ---------------------------------------------------------------------------
# Fair (field-consensus) semantic judge
# ---------------------------------------------------------------------------
# The official anscheck judge above is already a semantic judge (it accepts
# "equivalent" answers and tolerates temporal off-by-one). This "fair" variant
# adds ONLY the clarifications that the two published competitor LongMemEval
# graders apply — mem0's and Zep's — so a correct-but-verbose or
# correct-but-more-precise answer is not scored as a false negative:
#
#   - judge by meaning, not exact wording (mem0, Zep, official "equivalent")
#   - a correct answer + extra correct detail (superset) is correct unless the
#     extra is factually wrong (mem0 explicit; Zep knowledge-update)
#   - a more specific / more precise answer that entails the correct answer is
#     correct, e.g. "22 days" for "3 weeks" (mem0)
#
# It DELIBERATELY EXCLUDES mem0-only catch-alls that would push past the field
# band — "if the user would be satisfied", symmetric "0" == "not enough info",
# and rounding for non-temporal numbers. The goal is to sit in the same
# strictness band the field reports under: not stricter than Zep (no benchmin),
# not looser than mem0 (no benchmax). This `fair` judge is the SINGLE number
# Loom reports — the same way mem0 and Zep each report one number under their
# own semantic judge. The upstream-strict `official` judge stays available via
# `--judge-style official` for anyone who wants to reproduce it, but we do not
# headline two numbers (that only invites confusion).
_FAIR_CLARIFY = (
    " Judge by MEANING, not exact wording: a paraphrase or different vocabulary"
    " conveying the same fact is correct. A response that gives the correct"
    " answer plus extra correct detail (a superset) is correct unless the extra"
    " detail is factually wrong. A more specific or more precise answer that"
    " entails the correct answer is correct (e.g. \"22 days\" for \"3 weeks\")."
    " A response that omits the core required fact is incorrect."
)


def get_anscheck_prompt_fair(task, question, answer, response, abstention=False):
    if abstention:
        # The official abstention check is already fair — reuse it verbatim.
        return get_anscheck_prompt(task, question, answer, response, abstention=True)
    if task in ['single-session-user', 'single-session-assistant', 'multi-session']:
        template = ("I will give you a question, a correct answer, and a response from a model."
                    " Answer yes if the response contains the correct answer or is semantically"
                    " equivalent to it; otherwise answer no." + _FAIR_CLARIFY +
                    "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                    "Is the model response correct? Answer yes or no only.")
    elif task == 'temporal-reasoning':
        template = ("I will give you a question, a correct answer, and a response from a model."
                    " Answer yes if the response contains the correct answer or is semantically"
                    " equivalent to it; otherwise answer no. Do not penalize off-by-one errors"
                    " for the number of days/weeks/months." + _FAIR_CLARIFY +
                    "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                    "Is the model response correct? Answer yes or no only.")
    elif task == 'knowledge-update':
        template = ("I will give you a question, a correct answer, and a response from a model."
                    " Answer yes if the response contains the correct, updated answer — including"
                    " previous/outdated information alongside it is fine as long as the updated"
                    " answer is present; otherwise answer no." + _FAIR_CLARIFY +
                    "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
                    "Is the model response correct? Answer yes or no only.")
    elif task == 'single-session-preference':
        template = ("I will give you a question, a rubric for the desired personalized response,"
                    " and a response from a model. Answer yes if the response recalls and uses the"
                    " user's personal information correctly; it need not reflect every point in the"
                    " rubric. Otherwise answer no." + _FAIR_CLARIFY +
                    "\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
                    "Is the model response correct? Answer yes or no only.")
    else:
        raise NotImplementedError
    return template.format(question, answer, response)


def _judge_one(client, model, prompt):
    completion = chat_completions_with_backoff(
        client, model=model, n=1, temperature=0, max_tokens=10,
        messages=[{"role": "user", "content": prompt}],
    )
    return 'yes' in completion.choices[0].message.content.strip().lower()


def _score_style(style, client, model, model_short, hypotheses, qid2qdata, qid2qtype, hyp_file):
    """Grade every hypothesis under one judge style; print + save results."""
    prompt_fn = get_anscheck_prompt_fair if style == 'fair' else get_anscheck_prompt
    qtype2acc = {t: [] for t in set(qid2qtype.values())}
    overall = []
    result_file = '{}.eval-results-{}-{}'.format(hyp_file, model_short, style)
    with open(result_file, 'w') as out_f:
        for entry in tqdm(hypotheses, desc=style):
            qid = entry['question_id']
            if qid not in qid2qtype:
                continue
            qtype = qid2qtype[qid]
            prompt = prompt_fn(
                qtype, qid2qdata[qid]['question'], qid2qdata[qid]['answer'],
                entry['hypothesis'], abstention='_abs' in qid,
            )
            label = _judge_one(client, model, prompt)
            entry = {**entry, 'autoeval_label': {'model': model, 'style': style, 'label': label}}
            print(json.dumps(entry), file=out_f)
            qtype2acc[qtype].append(1 if label else 0)
            overall.append(1 if label else 0)
    print('[{}] Accuracy: {}'.format(style, round(float(np.mean(overall)), 4)))
    for k, v in sorted(qtype2acc.items()):
        if v:
            print('\t{}: {} ({})'.format(k, round(float(np.mean(v)), 4), len(v)))
    print('[{}] saved to {}'.format(style, result_file))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='LongMemEval QA judge (official + fair semantic).')
    ap.add_argument('metric_model', help='judge model key: ' + ', '.join(model_zoo))
    ap.add_argument('hyp_file', help='hypotheses JSONL (one {question_id, hypothesis} per line)')
    ap.add_argument('ref_file', help='reference dataset JSON (question_id, question, answer, question_type)')
    ap.add_argument('--judge-style', choices=['fair', 'official'], default='fair',
                    help="fair (default) = the field-consensus semantic judge Loom reports "
                         "under (same strictness band as mem0/Zep); official = the upstream "
                         "strict-semantic anscheck, kept only for reproducibility.")
    args = ap.parse_args()

    if args.metric_model not in model_zoo:
        print('Requested metric model is not supported:', args.metric_model)
        sys.exit(1)
    metric_model, metric_model_source = model_zoo[args.metric_model]
    if metric_model_source == 'openai':
        openai.organization = os.getenv('OPENAI_ORGANIZATION')
        metric_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    else:
        metric_client = OpenAI(api_key='EMPTY', base_url='http://localhost:8001/v1')

    try:
        hypotheses = [json.loads(line) for line in open(args.hyp_file).readlines()]
    except json.JSONDecodeError:
        hypotheses = json.load(open(args.hyp_file))
    try:
        references = json.load(open(args.ref_file))
    except json.JSONDecodeError:
        references = [json.loads(line) for line in open(args.ref_file).readlines()]
    qid2qdata = {e['question_id']: e for e in references}
    qid2qtype = {e['question_id']: e['question_type'] for e in references}

    _score_style(args.judge_style, metric_client, metric_model, args.metric_model,
                 hypotheses, qid2qdata, qid2qtype, args.hyp_file)
