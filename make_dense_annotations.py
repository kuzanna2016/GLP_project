import json


def main(clevr_path, save_path):
    clevr = json.load(open(clevr_path))
    dense = [
        {"image_id": d['image_id'], "round_id": d['round_id'],
         "gt_relevance": [1.0 if i == d['dialog'][-1]['answer'] else 0.0 for i in range(100)]}
        for d in clevr['data']['dialogs']
    ]
    json.dump(dense, open(save_path, 'w'))


if __name__ == '__main__':
    main(clevr_path='clevr/CLEVR_VD_VAL_VISDIAL_1000_pictures_full_dialogs.json',
         save_path='clevr/CLEVR_VD_VAL_VISDIAL_1000_pictures_full_dialogs_dense.json')
