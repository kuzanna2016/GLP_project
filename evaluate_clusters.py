import json
import spacy
import difflib
import transformers

from coreference import extract_clusters

nlp = spacy.load("en_core_web_sm")
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")


def move_ix(cluster, step=1):
    return [[i + step for i in c] for c in cluster]


def get_pairs(clusters):
    pairs = []
    overlaps = []
    for i, cluster0 in enumerate(clusters):
        i_overlap = []
        for j, cluster1 in enumerate(clusters[i + 1:]):
            i_overlap.append(len(set(cluster1) & set(cluster0)))
        overlaps.append(i_overlap)
    taken = []
    for i, i_overlaps in enumerate(overlaps):
        overlap_len = sorted(enumerate(i_overlaps), key=lambda x: x[1], reverse=True)
        for j, l in overlap_len:
            if l != 0 and i not in taken and i + 1 + j not in taken:
                pairs.append([clusters[i], clusters[i + 1 + j]])
                taken.append(i)
                taken.append(i + 1 + j)
    leftovers = [[c] for i, c in enumerate(clusters) if i not in taken]
    pairs.extend(leftovers)
    return pairs


def precise_comparison(clusters_gold, clusters_pred, i, j, d):
    tp, fp, fn = 0, 0, 0
    clusters_pred = [tuple(tuple(i) for i in c) for c in clusters_pred]
    clusters_pred = [tuple(sorted(c)) for c in clusters_pred]
    clusters_pred = sorted(clusters_pred, key=lambda d: d[0][0])
    clusters_gold = [tuple(tuple(i) for i in c) for c in clusters_gold]
    clusters_gold = [tuple(sorted(c)) for c in clusters_gold]
    clusters_gold = sorted(clusters_gold, key=lambda d: d[0][0])
    difference = set(clusters_pred) ^ set(clusters_gold)
    difference = list(difference)
    recovered = True
    if len(difference) == 0:
        tp += sum(len(c) for c in clusters_pred)
        return 1, tp, fp, fn
    else:
        print(i, j, 'Clusters with errors')
        one_cluster_difference = [d for d in difference if len(d) == 1]
        for c in one_cluster_difference:
            recovered = False
            if c in clusters_gold:
                fn += 1
                print('Missing in prediction', c)
            else:
                fp += 1
                print('Extra in prediction', c)
        difference = [d for d in difference if len(d) > 1]
        pairs = get_pairs(difference)
        for pair in pairs:
            if len(pair) == 1:
                recovered = False
                if pair[0] in clusters_gold:
                    fn += len(pair[0])
                    print('Missing in prediction', pair[0])
                else:
                    fp += len(pair[0])
                    print('Extra in prediction', pair[0])
                continue
            if set(pair[0]) == set(pair[1]):
                continue
            recovered = False
            if pair[0] in clusters_gold:
                diff = list(d.compare(pair[0], pair[1]))
            else:
                diff = list(d.compare(pair[1], pair[0]))
            for s in diff:
                if s.startswith('-'):
                    fn += 1
                elif s.startswith('+'):
                    fp += 1
                else:
                    tp += 1
            print('\n'.join(diff))
            print('-' * 30)
        if recovered:
            print('Recovered')
            return 1, tp, fp, fn
        print('=' * 40)
    return 0, tp, fp, fn


def main(clevr_path, clusters_gold_path, save_pred=None, count_solo=False, gold_aligned=False):
    if save_pred:
        pred = []
    clevr = json.load(open(clevr_path))
    clevr_annotated_clusters = json.load(open(clusters_gold_path))
    # clevr_annotated_clusters = [{'i':i+100,'j':j,'clusters':[c for c in dialog if len(c) > 1]}
    #     for i, dialogs in enumerate(clevr_annotated_clusters)
    #     for j, dialog in enumerate(dialogs)
    #
    # ]
    # json.dump(clevr_annotated_clusters, open(clusters_gold_path, 'w'))
    # for dialog in clevr_annotated_clusters:
    #     dialog['clusters'] = [[[i + 1 for i in c] for c in cluster] for cluster in dialog['clusters']]

    d = difflib.Differ()
    score = 0
    tp_all, fp_all, fn_all = 0, 0, 0
    for n_case, case in enumerate(clevr_annotated_clusters):
        i, j = case['i'], case['j']
        if gold_aligned:
            dialog = clevr[n_case]
        else:
            dialog = clevr[i]['dialogs'][j]
        clusters_pred = extract_clusters(dialog)
        clusters_pred = [move_ix(c, -1) for c in clusters_pred]
        clusters_gold = case['clusters']
        if not count_solo:
            clusters_gold = [c for c in clusters_gold if len(c) > 1]
        if save_pred:
            pred.append({'i': i, 'j': j, 'clusters': clusters_pred})
        s, tp, fp, fn = precise_comparison(clusters_gold, clusters_pred, i, j, d)
        tp_all += tp
        fp_all += fp
        fn_all += fn
        score += s
    if save_pred:
        json.dump(pred, open(save_pred, 'w'))
    print(f'{score}/{len(clevr_annotated_clusters)} matched completely, {score / len(clevr_annotated_clusters)}')
    print(f'Precision {tp_all / (tp_all + fp_all)}, recall {tp_all / (tp_all + fn_all)}')


if __name__ == '__main__':
    main(clevr_path='clevr/clevr_val_random_75_dialogs.json',
         clusters_gold_path='clevr/clevr_val_random_75_dialogs_tokens_annotated_clusters_revised.json',
         gold_aligned=True
         # save_pred='clevr/clevr_val_random_75_dialogs_tokens_annotated_clusters_predicted.json'
         )
    # main(clevr_path='clevr/clevr_val_with_coref_0_4.json',
    #      clusters_gold_path='clevr/clevr_val_with_coref_0_4_annotated.json')
