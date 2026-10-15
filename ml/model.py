"""Trainable, local sentiment baseline. No external service or model download."""
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

LABELS = ['negative', 'neutral', 'positive']
ALIASES = {'olumlu':'positive', 'olumsuz':'negative', 'nötr':'neutral', 'notr':'neutral', **{x:x for x in LABELS}}


def normalize(text):
    return re.sub(r'\s+', ' ', text.replace('İ','i').replace('I','ı').lower()).strip()


def read_labeled(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not {'text','label'}.issubset(reader.fieldnames or []):
            raise ValueError('Eğitim CSV dosyası text ve label sütunları içermeli.')
        unique = {}
        for row in reader:
            text, label = row['text'].strip(), ALIASES.get(normalize(row['label']))
            if not text or not label:
                raise ValueError('Boş metin veya geçersiz etiket: positive/negative/neutral kullanın.')
            key = normalize(text)
            if key in unique and unique[key][1] != label:
                raise ValueError('Aynı metne çelişen etiketler verilmiş.')
            unique[key] = (text, label)
    return list(unique.values())


def make_pipeline(kind='logistic_regression'):
    features = FeatureUnion([
        ('word', TfidfVectorizer(preprocessor=normalize, ngram_range=(1,2), sublinear_tf=True)),
        ('char', TfidfVectorizer(preprocessor=normalize, analyzer='char_wb', ngram_range=(3,5), sublinear_tf=True))
    ])
    classifier = LogisticRegression(max_iter=1500, class_weight='balanced', random_state=42, C=4) if kind == 'logistic_regression' else MultinomialNB(alpha=0.5)
    return Pipeline([('features',features),('classifier',classifier)])


def evaluate(model, rows):
    texts, labels = zip(*rows)
    predictions = model.predict(list(texts))
    return {'macro_f1':float(f1_score(labels, predictions, labels=LABELS, average='macro', zero_division=0)),
            'accuracy':float(accuracy_score(labels, predictions)),
            'classification_report':classification_report(labels, predictions, labels=LABELS, output_dict=True, zero_division=0),
            'confusion_matrix':confusion_matrix(labels,predictions,labels=LABELS).tolist(),
            'labels':LABELS,
            'errors':[{'text':t,'expected':y,'predicted':str(p)} for t,y,p in zip(texts,labels,predictions) if y != p]}


def train(train_path, test_path, output):
    rows = read_labeled(train_path)
    counts = Counter(label for _,label in rows)
    if set(counts) != set(LABELS) or min(counts.values()) < 5:
        raise ValueError('Üç sınıfın her birinde en az 5 farklı eğitim örneği gerekli.')
    if test_path:
        test_rows = read_labeled(test_path)
        overlap = {normalize(t) for t,_ in rows} & {normalize(t) for t,_ in test_rows}
        if overlap:
            raise ValueError('Eğitim ve test metinleri çakışıyor. Veri sızıntısını giderin.')
        if {label for _,label in test_rows} != set(LABELS):
            raise ValueError('Test kümesi üç sınıfı da içermeli.')
        split = 'explicit_holdout'
    else:
        rows,test_rows = train_test_split(rows,test_size=0.25,random_state=42,stratify=[y for _,y in rows])
        split = 'stratified_random_75_25'
    x,y = zip(*rows)
    reports, selected = {}, None
    for kind in ['logistic_regression','multinomial_nb']:
        model = make_pipeline(kind)
        model.fit(list(x),list(y))
        reports[kind] = evaluate(model,test_rows)
        if kind == 'logistic_regression':
            selected = model
    report = {'schema_version':1,'trained_at':datetime.now(timezone.utc).isoformat(),
              'model':'logistic_regression','train_rows':len(rows),'test_rows':len(test_rows),
              'split':split,'train_class_counts':dict(Counter(y)),
              'train_sha256':hashlib.sha256(Path(train_path).read_bytes()).hexdigest(),
              'dataset_note':'Bundled examples are synthetic. Custom CSV quality must be assessed separately.',
              'metrics':reports,
              'score_note':'Softmax-style classifier probability is not a calibrated correctness guarantee.'}
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    # Metadata + model are one bundle, replaced atomically for local readers.
    temporary = output/'model.tmp'
    joblib.dump({'model':selected,'report':report},temporary)
    temporary.replace(output/'model.joblib')
    (output/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    return report


class SentimentModel:
    def __init__(self, artifact_dir, dataset_dir):
        path = Path(artifact_dir)/'model.joblib'
        if not path.exists():
            train(Path(dataset_dir)/'train.csv',Path(dataset_dir)/'holdout.csv',artifact_dir)
        bundle = joblib.load(path)
        self.pipeline, self.report = bundle['model'], bundle['report']

    def predict(self, texts, explain=False):
        features = self.pipeline.named_steps['features'].transform(texts)
        classifier = self.pipeline.named_steps['classifier']
        probabilities = classifier.predict_proba(features)
        classes = list(classifier.classes_)
        results = []
        names = self.pipeline.named_steps['features'].get_feature_names_out() if explain else None
        for i, probs in enumerate(probabilities):
            index = int(np.argmax(probs))
            label = classes[index]
            evidence = []
            if explain:
                row = features.getrow(i)
                contributions = [(names[j].split('__',1)[-1],float(value*classifier.coef_[index,j]))
                                 for j,value in zip(row.indices,row.data) if names[j].startswith('word__')]
                evidence = [{'term':t,'weight':round(w,4)} for t,w in sorted(contributions,key=lambda x:x[1],reverse=True)[:5] if w>0]
            results.append({'sentiment':str(label),'confidence':round(float(probs[index]),4),
                            'probabilities':{str(c):round(float(p),4) for c,p in zip(classes,probs)},
                            'evidence':evidence})
        return results
