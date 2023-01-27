import json

import spacy
import transformers
import random

nlp = spacy.load("en_core_web_sm")
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")


def print_clevr_dialog_w_stats(dialog):
    template_info = dialog['template_info'][0]
    graph = dialog['graph']['history'][0]
    caption = dialog['caption']
    inputs = template_info.get('inputs', '')
    dep = graph.get('dependence', '')
    f_id = graph.get('focus_id', '')
    print('index\tinputs\tdependence\tfocus_id')
    print(f"-1\t{inputs}\t{dep}\t\t{f_id}\t{caption}")
    for i, d in enumerate(dialog['dialog']):
        template_info = dialog['template_info'][i + 1]
        inputs = template_info.get('inputs', '')
        graph_history = dialog['graph']['history'][i + 1]
        dep = graph_history.get('dependence', '')
        f_id = graph_history.get('focus_id', '')
        print(f"{i}\t{inputs}\t{dep}\t\t{f_id}\t{d['question']} - {d['answer']}")


def enumerate_tokens(full_dialog_text):
    full_dialog_tokens = [t for s in full_dialog_text for t in tokenizer.tokenize(s) + ['<br>']]
    enumerated_dialog = []
    line_tokens = []
    for i, t in enumerate(full_dialog_tokens):
        if t == '<br>':
            enumerated_dialog.append(' '.join(f'{i}.{t}' for i, t in line_tokens))
            line_tokens = []
            continue
        line_tokens.append((i, t))
    return enumerated_dialog


def _print_dialog_w_tokens(dialog, i, j, fp):
    full_dialog_text = [dialog['caption'],
                        *[str(c) for r in dialog['dialog'] for c in [r['question'], r['answer']]]]
    enumerated_dialog = enumerate_tokens(full_dialog_text)
    if fp is not None:
        fp.write(f'Dialog {i},{j}, objects:\n')
        for o in dialog['graph']['objects'].values():
            fp.write(
                str(o['id']) + '\t' +
                o.get('shape', '') + ' ' +
                o.get('size', '') + ' ' +
                o.get('material', '') + ' ' +
                o.get('color', '') + '\n'
            )
        fp.write('C: ' + enumerated_dialog[0] + '\n')
        for i_r, r in enumerate(enumerated_dialog[1:]):
            if i_r % 2 == 0:
                fp.write(f'Q{i_r // 2 + 1}: ' + r + '\n')
            else:
                fp.write(f'A: ' + r + '\n')
        fp.write('\n' + '=' * 50 + '\n\n')
    else:
        print(f'Dialog {i},{j}, objects:')
        for o in dialog['graph']['objects'].values():
            print(str(o['id']) + '\t' + o.get('shape', ''), o.get('size', ''), o.get('material', ''),
                  o.get('color', ''))
        print('C:', enumerated_dialog[0])
        for i_r, r in enumerate(enumerated_dialog[1:]):
            if i_r % 2 == 0:
                print(f'Q{i_r // 2 + 1}:', r)
            else:
                print(f'A:', r)
        print('\n' + '=' * 50 + '\n')


def print_clevr_with_indexed_tokens(clevr, fp=None, n_random=None, prepare_json=False):
    if fp is not None:
        json_fp = fp[:-4]+'_annotated_clusters.json'
        fp = open(fp, 'w')
    prepared_json = []
    if n_random is not None:
        i_s = random.sample(range(len(clevr)), k=n_random)
        j_s = random.choices(range(5), k=n_random)
        for i, j in zip(i_s, j_s):
            dialog = clevr[i]['dialogs'][j]
            _print_dialog_w_tokens(dialog, i, j, fp)
            prepared_json.append({'i':i,'j':j,'clusters':[]})
    else:
        for i, dialogs in enumerate(clevr):
            i = dialogs['image_index']
            for j, dialog in enumerate(dialogs['dialogs']):
                _print_dialog_w_tokens(dialog, i, j, fp)
                prepared_json.append({'i': i, 'j': j, 'clusters': []})
    if fp is not None:
        fp.close()
        if prepare_json is not None:
            json.dump(prepared_json, open(json_fp, 'w'))


def reconstruct_clusters(clusters, tokens):
    for i, cluster in enumerate(clusters):
        print(f'{i} corefrence cluster has {len(cluster)} NPs:')
        for start, end in cluster:
            print('\t' + ' '.join(tokens[start - 1:end]))


def print_visdial_dialog__with_clusters(dialog_data, questions, answers,
                                        with_indexes=False):
    clusters = dialog_data['clusters']
    caption = dialog_data['caption']
    tokens = caption.split()
    tokens.append('<br>')
    if with_indexes:
        print('C: ' + ' '.join(f'{i + 1}.{t}' for i, t in enumerate(tokens)))
    else:
        print('C: ' + caption)
    pad = len(tokens)
    for dialog_round in dialog_data['dialog']:
        q = questions[dialog_round['question']]
        tokens.extend(q.split())
        tokens.append('<br>')
        if with_indexes:
            print('Q: ' + ' '.join(f'{i + 1 + pad}.{t}' for i, t in enumerate(q.split())))
        else:
            print('Q: ' + q)
        pad = len(tokens)
        if 'answer_options' in dialog_round:
            continue
        a = answers[dialog_round['answer']]
        tokens.extend(a.split())
        tokens.append('<br>')
        if with_indexes:
            print('A: ' + ' '.join(f'{i + 1 + pad}.{t}' for i, t in enumerate(a.split())))
        else:
            print('A: ' + a)
        pad = len(tokens)
    reconstruct_clusters(clusters, tokens)
