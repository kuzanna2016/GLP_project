import json
import spacy
import difflib
import os
import transformers


nlp = spacy.load("en_core_web_sm")
tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")


def main(prediction_folder):
    predictions = []
    for f in os.listdir(prediction_folder):
        pred = json.load(open(os.path.join(prediction_folder, f)))
        pred['image_id'] = f
        predictions.append(pred)



if __name__ == '__main__':
    # main(clevr_path='clevr/clevr_val_random_75_dialogs.json',
    #      clusters_gold_path='clevr/clevr_val_random_75_dialogs_annotated_clusters_revised.json',
    #      gold_aligned=True
    #      # save_pred='clevr/clevr_val_random_75_dialogs_tokens_annotated_clusters_predicted.json'
    #      )
    main(prediction_folder='../VD_PCR_predictions/clevr/conly_MB-JC_eval_coref_output/test')
