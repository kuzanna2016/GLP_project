import json
import spacy
import tqdm
import transformers
import random

from coreference import extract_clusters

nlp = spacy.load("en_core_web_sm")
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")


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
            clusters = extract_clusters(dialog)
            if mix_dialogs > 0:
                ixs = random.choices(range(len(new_rounds)), k=mix_dialogs)
                ixs += [len(new_rounds)]
                new_dialogs = [
                    {
                        'image_id': image_ix,
                        'dialog': new_rounds[:i],
                        'caption': caption,
                        'clusters': clusters,
                        'round_id': i,
                    }
                    for i in ixs
                ]
                clevr_dialogs.extend(new_dialogs)
            else:
                new_dialog = {
                    'image_id': image_ix,
                    'dialog': new_rounds,
                    'caption': caption,
                    'clusters': clusters,
                    'round_id': len(new_rounds),
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
    main(clevr_path='clevr/CLEVR_VD_VAL.json',
         save_path='clevr/CLEVR_VD_VAL_VISDIAL_1000_pictures_full_dialogs.json',
         mix_dialogs=0)
