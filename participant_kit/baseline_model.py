import torch
import json
import os
import argparse
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from datasets import load_dataset

LABEL_LIST = [0, 1]
MODEL_NAME = 'FacebookAI/xlm-roberta-base'


def tokenize_and_map_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples['model_output_text'], return_offsets_mapping=True, padding=True, truncation=True
    )
    offset_mappings = tokenized_inputs['offset_mapping']
    all_labels = examples['hard_labels']
    tok_labels_batch = []

    for batch_idx in range(len(offset_mappings)):
        offset_mapping = offset_mappings[batch_idx]
        hard_labels = all_labels[batch_idx]
        tok_labels = [0] * len(offset_mapping)
        for idx, start_end in enumerate(offset_mapping):
            start = start_end[0]
            end = start_end[1]
            for (label_start, label_end) in hard_labels:
                if start >= label_start and end <= label_end:
                    tok_labels[idx] = 1
        tok_labels_batch.append(tok_labels)
    tokenized_inputs['labels'] = tok_labels_batch
    return tokenized_inputs


def compute_iou(predictions, labels):
    """
    Compute the Intersection over Union (IoU) for each sample.
    """
    import numpy as np
    iou_scores = []
    for pred, label in zip(predictions, labels):
        max_len = max(max(pred) if pred else 0, max(label) if label else 0) + 1
        pred_binary = np.zeros(max_len, dtype=int)
        label_binary = np.zeros(max_len, dtype=int)

        for start, end in pred:
            pred_binary[start:end] = 1
        for start, end in label:
            label_binary[start:end] = 1

        intersection = np.sum(pred_binary & label_binary)
        union = np.sum(pred_binary | label_binary)
        iou = intersection / union if union > 0 else 0.0
        iou_scores.append(iou)

    return np.mean(iou_scores)


def compute_metrics(p):
    """
    Compute metrics for the Trainer.
    """
    predictions, labels = p
    predictions = torch.argmax(torch.tensor(predictions), dim=2)

    true_labels = []
    true_predictions = []
    for prediction, label in zip(predictions, labels):
        spans_pred = []
        spans_label = []

        for idx, val in enumerate(prediction):
            if val == 1:
                spans_pred.append(idx)
        for idx, val in enumerate(label):
            if val == 1:
                spans_label.append(idx)

        true_labels.append([(sp, sp + 1) for sp in spans_label])
        true_predictions.append([(sp, sp + 1) for sp in spans_pred])

    iou = compute_iou(true_predictions, true_labels)

    return {"iou": iou}


def train_model(train_file, val_file, output_dir):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=2)

    data_files = {
        'train': train_file,
        'validation': val_file
    }
    dataset = load_dataset('json', data_files=data_files)

    tokenized_datasets = dataset.map(lambda x: tokenize_and_map_labels(x, tokenizer), batched=True)
    train_dataset = tokenized_datasets['train']
    eval_dataset = tokenized_datasets['validation']

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()
    print(f"Model trained and evaluated successfully. Model checkpoint saved in {output_dir}")


def test_model(test_file, model_path):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(model_path)

    test_dataset = load_dataset('json', data_files={'test': test_file})['test']

    inputs = tokenizer(test_dataset['model_output_text'], padding=True, truncation=True, return_offsets_mapping=True, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        outputs = model(inputs.input_ids)
    preds = torch.argmax(outputs.logits, dim=2)
    probs = F.softmax(outputs.logits, dim=2)

    hard_labels_all = {}
    soft_labels_all = {}
    predictions_all = []
    for i, pred in enumerate(preds):
        hard_labels_sample = []
        soft_labels_sample = []
        positive_indices = torch.nonzero(pred == 1, as_tuple=False)
        offset_mapping = inputs['offset_mapping'][i]
        for j, offset in enumerate(offset_mapping):
            soft_labels_sample.append({'start': offset[0].item(), 'end': offset[1].item(), 'prob': probs[i][j][1].item()})
            if j in positive_indices:
                hard_labels_sample.append((offset[0].item(), offset[1].item()))
        soft_labels_all[test_dataset['id'][i]] = soft_labels_sample
        hard_labels_all[test_dataset['id'][i]] = hard_labels_sample
        predictions_all.append({'id': test_dataset['id'][i], 'hard_labels': hard_labels_sample, 'soft_labels': soft_labels_sample})

    with open(f"hard_labels.json", 'w') as f:
        json.dump(hard_labels_all, f)
    with open(f"soft_labels.json", 'w') as f:
        json.dump(soft_labels_all, f)
    with open(f"predictions.jsonl", 'w') as f:
        for pred_dict in predictions_all:
            print(json.dumps(pred_dict), file=f)

    print("Predictions saved to predictions.jsonl")


def main(args):
    if args.mode == 'train':
        train_model(train_file=args.train_file, val_file=args.val_file, output_dir=args.output_dir)
    else:
        test_model(test_file=args.test_file, model_path=args.model_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test the model")
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help="Train or test mode")
    parser.add_argument('--train_file', type=str, help="Path to the training file")
    parser.add_argument('--val_file', type=str, help="Path to the validation file")
    parser.add_argument('--test_file', type=str, help="Path to the test file")
    parser.add_argument('--model_checkpoint', type=str, default="./results", help="Path to the trained checkpoint")
    parser.add_argument('--output_dir', type=str, default="./results", help="Output directory for model checkpoints")
    args = parser.parse_args()
    main(args)
