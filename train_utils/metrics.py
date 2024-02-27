import numpy as np
import evaluate

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    f1_score = evaluate.load("f1")
    accuracy = evaluate.load("accuracy")
    metrics = {}
    metrics["f1"] = f1_score.compute(predictions=predictions, references=labels, average='macro')['f1']
    metrics["accuracy"] = accuracy.compute(predictions=predictions, references=labels)['accuracy']

    return metrics