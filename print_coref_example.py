import json
from ContextVD.data_construction.clevr_utils import pretty_print_corefs
import transformers
from collections import defaultdict

tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")


def get_tokens_char_span(char_ix, tokens, text):
    tokens_to_chars = []
    for t in tokens:
        if t.startswith('##'):
            t = t[2:]
        start = text.find(t)
        end = start + len(t)
        tokens_to_chars.append((char_ix + start, char_ix + end))
        char_ix += end + 1
        text = text[end + 1:]
    return tokens_to_chars, char_ix


def from_token_to_char(dialog):
    round_mapping = []
    round_token_count = []
    caption = dialog['caption']
    caption_tokens = tokenizer.tokenize(caption)
    caption_map, char_ix = get_tokens_char_span(0, caption_tokens, caption.lower())
    round_mapping.append(caption_map)
    round_token_count.append(0)
    token_count = len(caption_tokens) + 1
    for r in dialog['dialog']:
        char_ix = 0
        round_token_count.append(token_count)
        q = r['question']
        a = str(r['answer'])
        q_t = tokenizer.tokenize(q)
        a_t = tokenizer.tokenize(a)
        q_map, char_ix = get_tokens_char_span(char_ix, q_t, q.lower())
        a_map, char_ix = get_tokens_char_span(char_ix, a_t, a.lower())
        round_mapping.append(q_map + [(-1,-1)] + a_map)
        token_count += len(q_t) + 1 + len(a_t) + 1
    return round_mapping, round_token_count


if __name__ == '__main__':
    clevr = json.load(open('clevr/clevr_val_raw_0_999.json'))
    clusters = json.load(open('clevr/clevr_val_100_110_predicted_clusters.json'))
    for case in clusters:
        dialog = clevr[case['i']]['dialogs'][case['j']]
        rounds_mapping, rounds_token_count = from_token_to_char(dialog)
        groups = defaultdict(list)
        for i, cluster in enumerate(case['clusters']):
            for c in cluster:
                round, token_count = [(r, count) for r, count in enumerate(rounds_token_count) if count <= c[0]][-1]
                groups[round].append({
                    "group_id": i,
                    "span": [rounds_mapping[round][c[0] - token_count][0], rounds_mapping[round][c[1] - token_count][1]]
                })
        pretty_print_corefs(dialog, groups)
