import json
import spacy
import tqdm
import transformers
import random
import argparse

from coreference import group_nps_pipline, prepare_for_indexing, index_clusters, index_nps_and_pronouns
from make_dense_annotations import main as make_dense

nlp = spacy.load("en_core_web_sm")
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

parser = argparse.ArgumentParser(description='Main script for visdial-coref')
parser.add_argument('--clevr_path', type=str, default='clevr/CLEVR_VD_VAL.json')
parser.add_argument('--save_path', type=str, default='clevr/CLEVR_VD_VAL_VISDIAL_1000_pictures_full_dialogs.json')
parser.add_argument('--dense_save_path', type=str, default='clevr/CLEVR_VD_VAL_VISDIAL_1000_pictures_full_dialogs_dense.json')
parser.add_argument('--mix_dialog_length', action='store_true')
parser.add_argument('--n_answers', type=int, default=100)
parser.add_argument('--n_dialogs_per_image', type=int, default=1)

def connect_pronoun_info(nps, pronouns, clusters):
    pronoun_info = []
    for pronoun in pronouns:
        cluster = [cluster for cluster in clusters if pronoun in cluster]
        if not cluster:
            continue
        cluster = cluster[0]
        correct = [c for c in cluster if c != pronoun and c[0] < pronoun[0]]
        if not correct:
            continue
        candidates = [np for np in nps if np[0] < pronoun[0]]
        for c in correct:
            if c not in candidates:
                candidates.append(c)
        candidates = sorted(candidates)
        pronoun_info.append({
            "current_pronoun": pronoun,
            "reference_type": 0,
            "correct_NPs": correct,
            "candidate_NPs": candidates
        })
    return pronoun_info


def main(args):
    clevr = json.load(open(args.clevr_path))
    images = json.load(open('clevr/images.json'))
    images = [i[2:-1] for i in images]
    clevr_questions = []
    clevr_answers = []
    clevr_dialogs = []
    for dialog_set in tqdm.tqdm(clevr):
        image_ix = str(dialog_set['image_index'])
        image_ix = 'CLEVR_val_' + '0' * (6 - len(image_ix)) + image_ix
        if image_ix not in images:
            continue
        dialogs = random.sample(dialog_set['dialogs'], k=args.n_dialogs_per_image) if args.n_dialogs_per_image < len(
            dialog_set['dialogs']) else dialog_set['dialogs']
        for dialog in dialogs:
            new_rounds = []
            dialog_tokenized = []
            caption = tokenizer.tokenize(dialog['caption'])
            dialog_tokenized.append(caption)
            caption = ' '.join(caption)
            for dialog_round in dialog['dialog']:
                question = tokenizer.tokenize(dialog_round['question'])
                dialog_tokenized.append(question)
                question = ' '.join(question)
                answer = tokenizer.tokenize(str(dialog_round['answer']))
                dialog_tokenized.append(answer)
                answer = ' '.join(answer)
                if question not in clevr_questions:
                    clevr_questions.append(question)
                if answer not in clevr_answers:
                    clevr_answers.append(answer)
                question_ix = clevr_questions.index(question)
                answer_ix = clevr_answers.index(answer)
                new_rounds.append({
                    'answer': answer_ix,
                    'question': question_ix,
                    'answer_options': list(range(100)),
                    "gt_index": answer_ix,
                })
            groupped_nps, round_nps, round_pronouns = group_nps_pipline(dialog)
            start_token_per_round, end_ix, aligned_rounds = prepare_for_indexing(dialog)
            if args.mix_dialog_length:
                i = random.choice(range(1, len(new_rounds)))
                nps, pronouns = index_nps_and_pronouns(round_nps, round_pronouns, start_token_per_round, end_ix,
                                                       aligned_rounds, last_round=i)
                clusters = index_clusters(groupped_nps, start_token_per_round, end_ix, aligned_rounds, last_round=i)
                pronoun_info = connect_pronoun_info(nps, pronouns, clusters)
                clevr_dialogs.append({
                    'image_id': image_ix,
                    'dialog': new_rounds[:i],
                    'caption': caption,
                    'clusters': clusters,
                    'round_id': i,
                    'pronoun_info': pronoun_info
                })
            else:
                nps, pronouns = index_nps_and_pronouns(round_nps, round_pronouns, start_token_per_round, end_ix,
                                                       aligned_rounds)
                clusters = index_clusters(groupped_nps, start_token_per_round, end_ix, aligned_rounds)
                pronoun_info = connect_pronoun_info(nps, pronouns, clusters)
                new_dialog = {
                    'image_id': image_ix,
                    'dialog': new_rounds,
                    'caption': caption,
                    'clusters': clusters,
                    'round_id': len(new_rounds),
                    'pronoun_info': pronoun_info,
                }
                clevr_dialogs.append(new_dialog)
    while len(clevr_answers) < args.n_answers:
        clevr_answers += clevr_answers[:args.n_answers - len(clevr_answers)]
    clevr_visdial = {
        'data': {
            'dialogs': clevr_dialogs,
            'answers': clevr_answers,
            'questions': clevr_questions
        },
        'split': 'test',
        'version': '1.2'
    }
    json.dump(clevr_visdial, open(args.save_path, 'w'))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    make_dense(args.save_path, args.dense_save_path)

