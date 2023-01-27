import json
from utils import print_clevr_with_indexed_tokens

clevr = json.load(open('clevr/CLEVR_VD_VAL.json'))
images = json.load(open('clevr/images.json'))
images = [i[2:-1] for i in images]
filtered_clevr = []
for dialog_set in clevr:
    image_ix = str(dialog_set['image_index'])
    image_ix = 'CLEVR_val_' + '0' * (6 - len(image_ix)) + image_ix
    if image_ix not in images:
        continue
    filtered_clevr.append(dialog_set)
  
print_clevr_with_indexed_tokens(
    filtered_clevr,
    fp='clevr/clevr_val_visdial_1000_tokens.txt'
)
