"""Optional BERTurk fine-tuning experiment; does not replace the dashboard baseline.

Install requirements-transformer.txt before use. Downloads a public foundation
model on first use. No paid API is involved. Training is much faster on a GPU.
"""
import argparse
import json
from pathlib import Path

from .model import LABELS,normalize,read_labeled,train


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--csv',default='datasets/train.csv')
    parser.add_argument('--test-csv',default='datasets/holdout.csv')
    parser.add_argument('--model',default='dbmdz/bert-base-turkish-cased')
    parser.add_argument('--output',default='artifacts/transformer')
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--max-length',type=int,default=128)
    args=parser.parse_args()
    if args.epochs<1 or args.batch_size<1: parser.error('Epoch and batch size must be positive.')
    rows,test_rows=read_labeled(args.csv),read_labeled(args.test_csv)
    if {normalize(t) for t,_ in rows}&{normalize(t) for t,_ in test_rows}:
        raise ValueError('Training and test texts overlap.')
    if {y for _,y in rows}!=set(LABELS) or {y for _,y in test_rows}!=set(LABELS):
        raise ValueError('Both datasets must contain all three labels.')
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer,AutoModelForSequenceClassification,set_seed
    from sklearn.metrics import classification_report,confusion_matrix,f1_score,accuracy_score

    set_seed(42)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    # Same datasets for baseline comparison. Never use test loss for training or checkpoint selection.
    baseline=train(args.csv,args.test_csv,output/'baseline')
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    network=AutoModelForSequenceClassification.from_pretrained(args.model,num_labels=3,
            id2label=dict(enumerate(LABELS)),label2id={label:i for i,label in enumerate(LABELS)})
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.to(device)
    def collate(batch):
        texts,labels=zip(*batch)
        encoded=tokenizer(list(texts),padding=True,truncation=True,max_length=args.max_length,return_tensors='pt')
        encoded['labels']=torch.tensor([LABELS.index(y) for y in labels])
        return {key:value.to(device) for key,value in encoded.items()}
    loader=DataLoader(rows,batch_size=args.batch_size,shuffle=True,collate_fn=collate)
    optimizer=torch.optim.AdamW(network.parameters(),lr=2e-5,weight_decay=0.01)
    for epoch in range(args.epochs):
        network.train();loss_total=0
        for batch in loader:
            optimizer.zero_grad();result=network(**batch);result.loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(),1.0)
            optimizer.step();loss_total+=result.loss.item()
        print(f'Epoch {epoch+1}: training loss={loss_total/len(loader):.4f}',flush=True)
    network.eval();predictions=[]
    with torch.no_grad():
        for batch in DataLoader(test_rows,batch_size=args.batch_size,collate_fn=collate):
            predictions.extend(LABELS[i] for i in network(**batch).logits.argmax(dim=-1).cpu().tolist())
    labels=[y for _,y in test_rows]
    metrics={'macro_f1':f1_score(labels,predictions,labels=LABELS,average='macro',zero_division=0),
             'accuracy':accuracy_score(labels,predictions),
             'classification_report':classification_report(labels,predictions,labels=LABELS,output_dict=True,zero_division=0),
             'confusion_matrix':confusion_matrix(labels,predictions,labels=LABELS).tolist(),
             'errors':[{'text':t,'expected':y,'predicted':p} for (t,y),p in zip(test_rows,predictions) if y!=p]}
    report={'foundation_model':args.model,'epochs':args.epochs,'train_rows':len(rows),'test_rows':len(test_rows),
            'device':str(device),'transformer':metrics,'baselines':baseline['metrics'],
            'note':'Bundled data is synthetic. No real-world performance claim. Dashboard still uses its local Logistic Regression model.'}
    network.save_pretrained(output/'model',safe_serialization=True)
    tokenizer.save_pretrained(output/'model')
    (output/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'transformer_macro_f1':metrics['macro_f1'],'report':str(output/'comparison.json')},indent=2))


if __name__=='__main__': main()
