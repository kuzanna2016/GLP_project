import json
import tqdm
import transformers
import tokenizations
import jsonlines
import argparse
from coreference import _get_difference, count_start_per_round

tokenizer_un = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-cased")

parser = argparse.ArgumentParser()
parser.add_argument('--clevr_path', type=str, default='clevr/CLEVR_VD_VAL_VISDIAL_1000_pictures_full_dialogs.json')
parser.add_argument('--save_path', type=str,
                    default='clevr/CLEVR_VD_VAL_VISDIAL_1000_pictures_full_dialogs_ontonotes.jsonlines')


def prepare_sentence(sentence, data, subtoken_num, sent_num):
    sentence_text = tokenizer_un.convert_tokens_to_string(sentence.split()).capitalize()
    raw_tokens = sentence_text.split()
    tokens = tokenizer.tokenize(sentence_text)
    cased2uncased, uncased2cased = tokenizations.get_alignments(tokens, sentence.split())

    ctoken = raw_tokens[0]
    cpos = 0
    for token in tokens:
        data['sentences'][-1].append(token)
        data['speakers'][-1].append("-")
        data['sentence_map'].append(sent_num)
        data['subtoken_map'].append(subtoken_num)

        if token.startswith("##"):
            token = token[2:]
        if len(ctoken) == len(token):
            subtoken_num += 1
            cpos += 1
            if cpos < len(raw_tokens):
                ctoken = raw_tokens[cpos]
        else:
            ctoken = ctoken[len(token):]
    data['sentences'][-1].append("[SEP]")
    data['sentences'].append(["[CLS]"])
    data['speakers'][-1].append("[SPL]")
    data['speakers'].append(["[SPL]"])
    data['sentence_map'].append(sent_num)
    data['subtoken_map'].append(subtoken_num)
    data['sentence_map'].append(sent_num + 1)
    data['subtoken_map'].append(subtoken_num + 1)
    return data, subtoken_num, uncased2cased


def update_mapping(uncased2cased, cased_pointer, sent_uncased2cased):
    cased_pointer += 1
    uncased2cased.extend([[i + cased_pointer for i in m] for m in sent_uncased2cased])
    cased_pointer += max([i for m in sent_uncased2cased for i in m]) + 1
    uncased2cased.append([cased_pointer])
    cased_pointer += 1
    return uncased2cased, cased_pointer


def main(args):
    clevr = json.load(open(args.clevr_path))
    ontonotes = []
    for dialog in tqdm.tqdm(clevr['data']['dialogs']):
        data = {
            'doc_key': "bc",
            'sentences': [["[CLS]"]],
            'speakers': [["[SPL]"]],
            'clusters': [],
            'sentence_map': [0],
            'subtoken_map': [0],
        }
        cased_pointer = 0
        uncased2cased = [[0], ]
        text_uncased = '[CLS] ' + dialog['caption'] + ' [SEP]'
        sent_num = 0
        subtoken_num = 0
        data, subtoken_num, sent_uncased2cased = prepare_sentence(dialog['caption'], data, subtoken_num, sent_num)
        uncased2cased, cased_pointer = update_mapping(uncased2cased, cased_pointer, sent_uncased2cased)
        sent_num += 1
        for r in dialog['dialog']:
            q = clevr['data']['questions'][r['question']]
            text_uncased += ' ' + q + ' [SEP]'
            data, subtoken_num, sent_uncased2cased = prepare_sentence(q, data, subtoken_num, sent_num)
            uncased2cased, cased_pointer = update_mapping(uncased2cased, cased_pointer, sent_uncased2cased)
            sent_num += 1
            a = clevr['data']['answers'][r['answer']]
            text_uncased += ' ' + a + ' [SEP]'
            data, subtoken_num, sent_uncased2cased = prepare_sentence(a, data, subtoken_num, sent_num)
            uncased2cased, cased_pointer = update_mapping(uncased2cased, cased_pointer, sent_uncased2cased)
            sent_num += 1
        data['gold_clusters'] = [
            [
                [min(uncased2cased[c[0]]), max(uncased2cased[c[1]])]
                for c in cluster
            ]
            for cluster in dialog['clusters']
        ]
        ontonotes.append(data)
    with jsonlines.open(args.save_path, mode='w') as writer:
        for data in ontonotes:
            writer.write(data)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
