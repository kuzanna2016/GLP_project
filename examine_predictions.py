import json
import random
import os
import colorama
import jsonlines
import matplotlib.pyplot as plt

from coref_metrics import CorefEvaluator, PrCorefEvaluator, update_coref_evaluator
from visdial_metrics import SparseGTMetrics

def preproc_ontonotes(prediction_fp):
    with jsonlines.open(prediction_fp) as f:
        ontonotes = [obj for obj in f]
    for i, dialog in enumerate(ontonotes):
        dialog['sentences'] = [word for sentence in dialog['sentences'] for word in sentence]
    return ontonotes


def examine_coreference(prediction_fp, n_examples, isontonotes):
    predictions = []
    if os.path.isdir(prediction_fp):
        for f in os.listdir(prediction_fp):
            pred = json.load(open(os.path.join(prediction_fp, f)))
            pred['image_id'] = f[:-5]
            predictions.append(pred)
    elif isontonotes:
        predictions = preproc_ontonotes(prediction_fp)
    else:
        predictions = json.load(open(prediction_fp))
    coref_evaluator = CorefEvaluator()
    pr_evaluator = PrCorefEvaluator()
    for pred in predictions:
        update_coref_evaluator(pred, coref_evaluator, pr_evaluator)
    coref_precision, coref_recall, coref_f1 = coref_evaluator.get_prf()
    prp_precision, prp_recall, prp_f1 = pr_evaluator.get_prf()
    print('Total scores:')
    print(
        f'Coref precision:\t{coref_precision:.4f}\n'
        f'Coref recall:\t{coref_recall:.4f}\n'
        f'Coref f1:\t{coref_f1:.4f}\n'
    )
    print(
        f'Pronoun_Coref_average_precision:\t{prp_precision:.4f}\n'
        f'Pronoun_Coref_average_recall:\t{prp_recall:.4f}\n'
        f'Pronoun_Coref_average_f1:\t{prp_f1:.4f}\n'
    )
    print('-'*60)
    ixs = random.sample(range(len(predictions)), k=n_examples)
    for i, ix in enumerate(ixs):
        coref_evaluator = CorefEvaluator()
        pr_evaluator = PrCorefEvaluator()
        pred = predictions[ix]
        update_coref_evaluator(pred, coref_evaluator, pr_evaluator)
        print(i+1, pred.get('image_id', ix))
        coref_precision, coref_recall, coref_f1 = coref_evaluator.get_prf()
        prp_precision, prp_recall, prp_f1 = pr_evaluator.get_prf()
        print(
            f'Coref precision:\t{coref_precision:.4f}\n'
            f'Coref recall:\t{coref_recall:.4f}\n'
            f'Coref f1:\t{coref_f1:.4f}\n'
        )
        print(
            f'Pronoun_Coref_average_precision:\t{prp_precision:.4f}\n'
            f'Pronoun_Coref_average_recall:\t{prp_recall:.4f}\n'
            f'Pronoun_Coref_average_f1:\t{prp_f1:.4f}\n'
        )
        tokens = pred['sentences']
        for j, p_cluster in enumerate(pred['predicted_clusters']):
            for np in p_cluster:
                tokens[np[0]] = fr'[{tokens[np[0]]}'
                tokens[np[1]] = f'{tokens[np[1]]}]{j}'
        pretty_print_corefs(tokens, pred['gold_clusters'])
        print('='*60)
    return predictions


def pretty_print_corefs(tokens, gold_clusters):
    colorama.init()
    colored_tokens = pretty_print_coref_sentence(tokens, gold_clusters)
    sentence = []
    for token in colored_tokens:
        sentence.append(token)
        if token == '[SEP]':
            print(' '.join(sentence))
            sentence = []


def pretty_print_coref_sentence(tokens, clusters):
    fore_colors = ['RED', 'GREEN', 'YELLOW', 'CYAN', 'MAGENTA', 'BLUE']
    back_colors = ['BLACK', 'BLUE', 'WHITE']
    for i, cluster in enumerate(clusters):
        forecolor_id = i % len(fore_colors)
        # backcolor_id = i // len(fore_colors)

        forecolor = fore_colors[forecolor_id]
        # backcolor = back_colors[backcolor_id]
        for start, end in cluster:
            # b = getattr(colorama.Back, backcolor)
            f = getattr(colorama.Fore, forecolor)
            r = getattr(colorama.Style, 'RESET_ALL')
            tokens[start] = f'{f}{tokens[start]}'
            tokens[end] = f'{tokens[end]}{r}'
    return tokens


def examine_visdial(predictions, dialogs, n_examples):
    evaluator = SparseGTMetrics()
    target_ranks = [d['dialog'][-1]['gt_index'] for d in dialogs['data']['dialogs']]
    predicted_ranks = [p['ranks'] for p in predictions]
    evaluator.observe(predicted_ranks, target_ranks, rounds=[p['round_id'] for p in predictions])
    metrics = evaluator.retrieve(reset=False)
    ixs = random.sample(range(len(predictions)), k=n_examples)
    for i, ix in enumerate(ixs):
        dialog = dialogs['data']['dialogs'][ix]
        print(i + 1, dialog['image_id'])
        print('C:', dialog['caption'])
        for r in dialog['dialog']:
            q = dialogs['data']['questions'][r['question']]
            a = dialogs['data']['answers'][r['answer']]
            print('Q:', q)
            print('A:',a)
        pred_ranks = [predicted_ranks[ix].index(i) for i in range(1,6)]
        pred_answers = [dialogs['data']['answers'][p] for p in pred_ranks]
        print('Top-5 predicted:', '/'.join([p if p != a else f'__{p}__' for p in pred_answers]))
        print('=' * 60)



    return metrics


def main(prediction_fp, target_fp=None, vd=False, n_examples=10, save_fp=None, isontonotes=True):
    if not vd:
        predictions = examine_coreference(prediction_fp, n_examples, isontonotes)
        if save_fp is not None:
            json.dump(predictions, open(save_fp, 'w'))
    else:
        predictions = json.load(open(prediction_fp))
        dialogs = json.load(open(target_fp))
        metrics = examine_visdial(predictions, dialogs, n_examples)
        for r in [1,5,10]:
            rounds = {int(k[-1]):v.item() for k, v in metrics.items() if 'round' in k and f'r_{r}' in k}
            rounds = sorted(rounds.items())
            plt.plot(*zip(*rounds))
            plt.title(f'R@{r} for rounds')
            plt.show()
        print(metrics)


if __name__ == '__main__':
    # main(
    #     prediction_fp='../VD_PCR_predictions/clevr/conly_MB-JC_eval_coref_output_best',
    #     vd=False,
    #     n_examples=50,
    #     save_fp='../VD_PCR_predictions/clevr/conly_MB-JC_eval_coref_output_best.json'
    # )
    main(
        prediction_fp='../VD_PCR_predictions/clevr/vonly_MB-JC-HP-crf_cap-test_predict.json',
        target_fp='clevr/CLEVR_VD_VAL_VISDIAL_1000_pictures_mix_dialogs.json',
        n_examples=50,
        vd=True
    )
    # main(
    #     prediction_fp='clevr/CLEVR_VD_VAL_VISDIAL_1000_pictures_full_dialogs_spanbert_prediction.jsonlines',
    #     n_examples=50,
    #     isontonotes=True,
    # )
