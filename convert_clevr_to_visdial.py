import json
import spacy
import tqdm
import transformers
import random

from coreference import group_nps_pipline, prepare_for_indexing, index_clusters, index_nps_and_pronouns

nlp = spacy.load("en_core_web_sm")
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")

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


def main(clevr_path, save_path, mix_dialogs=3):
    clevr = json.load(open(clevr_path))
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
        for dialog in dialog_set['dialogs']:
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
            if mix_dialogs > 0:
                ixs = random.sample(range(1,len(new_rounds) - 1), k=mix_dialogs)
                ixs += [len(new_rounds)]
                new_dialogs = []
                for i in ixs:
                    nps, pronouns = index_nps_and_pronouns(round_nps, round_pronouns, start_token_per_round, end_ix,
                                                           aligned_rounds, last_round=i)
                    clusters = index_clusters(groupped_nps, start_token_per_round, end_ix, aligned_rounds, last_round=i)
                    pronoun_info = connect_pronoun_info(nps, pronouns, clusters)
                    new_dialogs.append({
                        'image_id': image_ix,
                        'dialog': new_rounds[:i],
                        'caption': caption,
                        'clusters': clusters,
                        'round_id': i,
                        'pronoun_info': pronoun_info
                    })
                clevr_dialogs.extend(new_dialogs)
            else:
                nps, pronouns = index_nps_and_pronouns(round_nps, round_pronouns, start_token_per_round, end_ix, aligned_rounds)
                clusters =  index_clusters(groupped_nps, start_token_per_round, end_ix, aligned_rounds)
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
    clevr_visdial = {
        'data': {
            'dialogs': clevr_dialogs,
            'answers': clevr_answers,
            'questions': clevr_questions
        },
        'split': 'test',
        'version': '1.2'
    }
    json.dump(clevr_visdial, open(save_path, 'w'))


if __name__ == '__main__':
    main(clevr_path='clevr/clevr_val_raw_0_999.json',
         save_path='clevr/CLEVR_VD_VAL_VISDIAL_1000_pictures_full_dialogs_1.json',
         mix_dialogs=-1)
