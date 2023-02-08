# GLP2022 Coreference project

This repository contains scripts for automatic coreference annotation
of [CLEVR-VD dataset](https://github.com/SuperJohnZhang/ContextVD),
helper scripts for data exploration and [VD-PCR model](https://github.com/HKUST-KnowComp/VD-PCR) prediction evaluation.

Generated dataset, model configs, image features and predictions can be found in
project's [Google Drive](https://drive.google.com/drive/folders/1_WYM50o-AOQpz_xDp6QW7UMVVeYo2GOz).

## Annotation

Annotation guidelines can be found in [annotation_and_dataset_notes.md](annotation_and_dataset_notes.md),
the 150 manually annotated dialogs are in folder [clevr](clevr).

To print dialogs with enumerated tokens and objects
use [prepare_dialogs_for_annotation.py](prepare_dialogs_for_annotation.py).

## Coreference annotation and conversion to VisDial format

To convert CLEVR-VD to VisDial format with annotated coreference
use [convert_clevr_to_visdial.py](convert_clevr_to_visdial.py).

To evaluate clustering performance on golden clusters use [evaluate_clusters.py](evaluate_clusters.py),
it will also print differences in the prediction and target clusters.

## VD-PCR model

[run_on_vm_instructions.md](run_on_vm_instructions.md) contains instruction on how to prepare the environment,
extract image features from ViLBERT and run VD-PCR on our dataset.
Updated configs are in [config](config).

## Predictions

Performance evaluation on coreference resolution and visual dialog can be done
with [examine_predictions.py](examine_predictions.py).
[visdial_metrics.py](visdial_metrics.py) and [coref_metrics.py](coref_metrics.py) are
taken from the [VD-PCR repo](https://github.com/HKUST-KnowComp/VD-PCR).

## SpanBERT

To convert CLEVR-VDCR to ontonotes format use [convert_to_ontonotes.py](convert_to_ontonotes.py).
We used [this implementation](https://github.com/mandarjoshi90/coref) for SpanBERT evaluation.
