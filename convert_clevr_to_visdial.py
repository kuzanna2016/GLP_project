import json
import spacy
import tqdm
import transformers

from coreference import extract_clusters

nlp = spacy.load("en_core_web_sm")
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")


def main(clevr_path, save_path):
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
        for dialog in dialog_set['dialogs'][:1]:
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
                    'question': question_ix
                })
            clusters = extract_clusters(dialog)
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
        'version': '1.1'
    }
    json.dump(clevr_visdial, open(save_path, 'w'))


if __name__ == '__main__':
    main(clevr_path='clevr/CLEVR_VD_VAL.json',
         save_path='clevr/CLEVR_VD_VAL_VISDIAL_1_dialog_per_1000_pictures.json')
