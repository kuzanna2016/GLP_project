import json
import random
import os
import colorama

from coref_metrics import CorefEvaluator, PrCorefEvaluator, update_coref_evaluator
from visdial_metrics import SparseGTMetrics

def examine_coreference(prediction_fp, n_examples):
    predictions = []
    for f in os.listdir(prediction_fp):
        pred = json.load(open(os.path.join(prediction_fp, f)))
        pred['image_id'] = f[:-5]
        predictions.append(pred)
    ixs = random.sample(range(len(predictions)), k=n_examples)
    for i, ix in enumerate(ixs):
        coref_evaluator = CorefEvaluator()
        pr_evaluator = PrCorefEvaluator()
        pred = predictions[ix]
        update_coref_evaluator(pred, coref_evaluator, pr_evaluator)
        print(i+1, pred['image_id'])
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


def examine_visdial(predictions, dialogs):
    evaluator = SparseGTMetrics()
    target_ranks = [d['dialog'][-1]['gt_index'] for d in dialogs['data']['dialogs']]
    predicted_ranks = [p['ranks'] for p in predictions]
    evaluator.observe(predicted_ranks, target_ranks)
    metrics = evaluator.retrieve(reset=False)
    return metrics


def main(prediction_fp, target_fp=None, vd=False, n_examples=10, save_fp=None):
    if not vd:
        predictions = examine_coreference(prediction_fp, n_examples)
        if save_fp is not None:
            json.dump(predictions, open(save_fp, 'w'))
    else:
        predictions = json.load(open(prediction_fp))
        dialogs = json.load(open(target_fp))
        metrics = examine_visdial(predictions, dialogs)
        print(metrics)


if __name__ == '__main__':
    # main(
    #     prediction_fp='../VD_PCR_predictions/clevr/conly_MB-JC_eval_coref_output_best',
    #     vd=False,
    #     n_examples=50,
    #     save_fp='../VD_PCR_predictions/clevr/conly_MB-JC_eval_coref_output_best.json'
    # )
    main(
        prediction_fp='../VD_PCR_predictions/clevr/vonly_MB-JC_predict_visdial_prediction_0402.json',
        target_fp='clevr/CLEVR_VD_VAL_VISDIAL_1000_pictures_mix_dialogs.json',
        vd=True,
        n_examples=50,
        save_fp='../VD_PCR_predictions/clevr/conly_MB-JC_eval_coref_output_best.json'
    )
