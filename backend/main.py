import csv
import io
import json
import os
import re
import sqlite3
import time
import uuid
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ml.model import SentimentModel, normalize
from .topics import TOPICS, detect_topics

ROOT=Path(os.path.abspath(__file__)).parents[1]
DATA=Path(os.getenv('YORUM_DATA_DIR',str(ROOT/'data')))
ARTIFACTS=Path(os.getenv('YORUM_ARTIFACT_DIR',str(ROOT/'artifacts')))
DATA.mkdir(parents=True,exist_ok=True)
DB=DATA/'radar.sqlite3'
model=SentimentModel(ARTIFACTS,ROOT/'datasets')


@contextmanager
def database():
    db=sqlite3.connect(DB,timeout=20); db.row_factory=sqlite3.Row
    try:
        with db: yield db
    finally: db.close()


with database() as db:
    db.executescript('''CREATE TABLE IF NOT EXISTS datasets(
        id TEXT PRIMARY KEY,name TEXT,created_at TEXT,row_count INTEGER);
        CREATE TABLE IF NOT EXISTS reviews(
        id TEXT PRIMARY KEY,dataset_id TEXT,text TEXT,date TEXT,rating REAL,
        sentiment TEXT,confidence REAL,topics TEXT,model_version TEXT);
        CREATE INDEX IF NOT EXISTS review_dataset ON reviews(dataset_id);
        CREATE INDEX IF NOT EXISTS review_sentiment ON reviews(sentiment);
    ''')

app=FastAPI(title='Yorum Radar API',version='1.0.0')


def parse_csv(raw):
    if len(raw)>5*1024*1024:
        raise HTTPException(413,'CSV en fazla 5 MB olabilir.')
    if b'\x00' in raw:
        raise HTTPException(422,'CSV metin dosyası olmalı; Excel dosyasını CSV olarak kaydedin.')
    text=None
    for encoding in ['utf-8-sig','cp1254']:
        try:
            text=raw.decode(encoding); break
        except UnicodeDecodeError: pass
    if text is None:
        raise HTTPException(422,'Dosyayı UTF-8 CSV olarak kaydedin.')
    try:
        dialect=csv.Sniffer().sniff(text[:8192],delimiters=',;\t')
    except csv.Error:
        dialect=csv.excel
    try:
        reader=csv.DictReader(io.StringIO(text),dialect=dialect)
        columns=[c.strip() for c in reader.fieldnames or []]
        if not columns or any(not c for c in columns) or len(set(columns))!=len(columns):
            raise HTTPException(422,'CSV tekil, boş olmayan sütun başlıkları içermeli.')
        reader.fieldnames=columns
        rows=[]
        for row in reader:
            if None in row: raise HTTPException(422,'Bazı satırlarda sütun sayısı başlıkla uyuşmuyor.')
            rows.append({k:(v or '').strip() for k,v in row.items()})
            if len(rows)>5000: raise HTTPException(413,'Bir seferde en fazla 5000 yorum yüklenebilir.')
        if not rows: raise HTTPException(422,'CSV içinde veri satırı bulunamadı.')
        return columns,rows
    except csv.Error:
        raise HTTPException(422,'CSV okunamadı; ayırıcı ve tırnak işaretlerini kontrol edin.')


def import_rows(name,columns,rows,text_column,date_column=None,rating_column=None):
    for column in [text_column,date_column,rating_column]:
        if column and column not in columns: raise HTTPException(422,f'Sütun bulunamadı: {column}')
    if not text_column: raise HTTPException(422,'Yorum sütunu seçmelisiniz.')
    usable=[]; skipped=0; bad_dates=0; bad_ratings=0
    for row in rows:
        text=row[text_column]
        if not text: skipped+=1; continue
        if len(text)>3000: raise HTTPException(422,'Yorum başına en fazla 3000 karakter destekleniyor.')
        day=None; rating=None
        if date_column and row[date_column]:
            try: day=date.fromisoformat(row[date_column]).isoformat()
            except ValueError: bad_dates+=1
        if rating_column and row[rating_column]:
            try:
                rating=float(row[rating_column].replace(',','.'))
                if not 1<=rating<=5: raise ValueError()
            except ValueError: rating=None; bad_ratings+=1
        usable.append((text,day,rating))
    if not usable: raise HTTPException(422,'Seçili sütunda boş olmayan yorum bulunamadı.')
    dataset_id=uuid.uuid4().hex
    predictions=model.predict([text for text,_,_ in usable])
    version=model.report['trained_at']
    records=[(uuid.uuid4().hex,dataset_id,text,day,rating,p['sentiment'],p['confidence'],json.dumps(detect_topics(text),ensure_ascii=False),version)
             for (text,day,rating),p in zip(usable,predictions)]
    with database() as db:
        existing=db.execute('SELECT COUNT(*) FROM reviews').fetchone()[0]
        if existing+len(records)>50000: raise HTTPException(413,'Yerel kütüphane limiti 50.000 yorum; eski veri setlerini silin.')
        db.execute('INSERT INTO datasets VALUES(?,?,?,?)',(dataset_id,name[:180],datetime.now(timezone.utc).isoformat(),len(records)))
        db.executemany('INSERT INTO reviews VALUES(?,?,?,?,?,?,?,?,?)',records)
    return {'id':dataset_id,'rows':len(records),'skipped_blank':skipped,'invalid_dates':bad_dates,'invalid_ratings':bad_ratings}


@app.get('/api/health')
def health():
    return {'status':'ok','model':'TF-IDF + Logistic Regression','api_key_required':False}


@app.get('/api/model')
def model_report():
    return model.report


@app.get('/api/datasets')
def datasets():
    with database() as db:
        return [dict(r) for r in db.execute('SELECT * FROM datasets ORDER BY created_at DESC')]


@app.post('/api/preview')
async def preview(file: UploadFile=File(...)):
    columns,rows=parse_csv(await file.read(5*1024*1024+1))
    return {'columns':columns,'preview':rows[:5],'row_count':len(rows)}


@app.post('/api/import',status_code=201)
async def upload(file: UploadFile=File(...),text_column: str=Form(...),date_column: Optional[str]=Form(None),rating_column: Optional[str]=Form(None)):
    columns,rows=parse_csv(await file.read(5*1024*1024+1))
    return import_rows(file.filename or 'Yorumlar',columns,rows,text_column,date_column or None,rating_column or None)


@app.post('/api/demo',status_code=201)
def demo():
    with database() as db:
        existing=db.execute("SELECT id,row_count FROM datasets WHERE name='Örnek mağaza · Sentetik veri'").fetchone()
    if existing: return {'id':existing['id'],'rows':existing['row_count'],'existing':True}
    columns,rows=parse_csv((ROOT/'datasets/demo.csv').read_bytes())
    return import_rows('Örnek mağaza · Sentetik veri',columns,rows,'yorum','tarih','puan')


@app.get('/api/sample')
def sample():
    return FileResponse(ROOT/'datasets/demo.csv',media_type='text/csv',filename='ornek-yorumlar.csv')


@app.delete('/api/datasets/{dataset_id}',status_code=204)
def remove_dataset(dataset_id: str):
    with database() as db:
        if not db.execute('SELECT 1 FROM datasets WHERE id=?',(dataset_id,)).fetchone(): raise HTTPException(404,'Veri seti bulunamadı.')
        db.execute('DELETE FROM reviews WHERE dataset_id=?',(dataset_id,))
        db.execute('DELETE FROM datasets WHERE id=?',(dataset_id,))


class ReviewInput(BaseModel):
    text: str=Field(min_length=3,max_length=3000)


@app.post('/api/predict')
def predict(body: ReviewInput):
    if len(body.text.strip())<3: raise HTTPException(422,'En az 3 karakter yazın.')
    started=time.perf_counter()
    result=model.predict([body.text],explain=True)[0]
    return {**result,'topics':detect_topics(body.text),'latency_ms':round((time.perf_counter()-started)*1000,2)}


def filtered_rows(dataset_id=None,sentiment=None,topic=None,q=None,date_from=None,date_to=None):
    clauses=[]; params=[]
    if sentiment and sentiment not in ['positive','negative','neutral']: raise HTTPException(422,'Geçersiz duygu filtresi.')
    if date_from and date_to and date_from>date_to: raise HTTPException(422,'Başlangıç tarihi bitişten sonra olamaz.')
    for column,value in [('dataset_id',dataset_id),('sentiment',sentiment)]:
        if value: clauses.append(column+'=?'); params.append(value)
    if q:
        clauses.append('text LIKE ? ESCAPE \'\\\'')
        params.append('%'+q.replace('\\','\\\\').replace('%','\\%').replace('_','\\_')+'%')
    if date_from: clauses.append('date>=?'); params.append(date_from.isoformat())
    if date_to: clauses.append('date<=?'); params.append(date_to.isoformat())
    sql='SELECT r.*,d.name AS dataset_name FROM reviews r JOIN datasets d ON d.id=r.dataset_id'
    if clauses: sql+=' WHERE '+' AND '.join(clauses)
    sql+=' ORDER BY date DESC,r.rowid DESC'
    with database() as db:
        rows=[dict(r) for r in db.execute(sql,params)]
    for row in rows: row['topics']=json.loads(row['topics'])
    return [row for row in rows if not topic or topic in row['topics']]


STOPWORDS=set('ve bir bu şu çok için ile ama daha en de da ürün ürünün geldi kadar olarak gibi mi ben hiç var yok olan her göre sonra önce şimdi böyle gayet beni bana oldu hem ne işte'.split())


@app.get('/api/dashboard')
def dashboard(dataset_id: Optional[str]=None,sentiment: Optional[str]=None,topic: Optional[str]=None,q: Optional[str]=Query(None,max_length=200),date_from: Optional[date]=None,date_to: Optional[date]=None,page: int=Query(1,ge=1),page_size: int=Query(10,ge=1,le=100)):
    rows=filtered_rows(dataset_id,sentiment,topic,q,date_from,date_to)
    counts=Counter(row['sentiment'] for row in rows)
    topic_stats=defaultdict(Counter); trend=defaultdict(Counter); words=Counter()
    for row in rows:
        for name in row['topics']: topic_stats[name][row['sentiment']]+=1
        if row['date']: trend[row['date']][row['sentiment']]+=1
        if row['sentiment']=='negative':
            words.update(set(word for word in re.findall(r'[a-zçğıöşü]+',normalize(row['text'])) if len(word)>3 and word not in STOPWORDS))
    ratings=[row['rating'] for row in rows if row['rating'] is not None]
    themes=sorted([{'name':name,'total':sum(values.values()),'negative':values['negative'],'positive':values['positive'],'neutral':values['neutral']} for name,values in topic_stats.items()],key=lambda x:(-x['negative'],-x['total'],x['name']))
    insights=[]
    if themes and themes[0]['negative']:
        insights.append(f"En fazla olumsuz yorum {themes[0]['name']} konusunda: {themes[0]['negative']} yorum.")
    low=sum(row['confidence']<0.55 for row in rows)
    if low: insights.append(f'{low} yorumda model skoru %55 altında; insan incelemesi önerilir.')
    if rows and not trend: insights.append('Tarih bilgisi yok; zaman grafiği için tarih sütunu eşleştirin.')
    start=(page-1)*page_size
    return {'total':len(rows),'counts':{label:counts[label] for label in ['positive','neutral','negative']},
            'negative_rate':round(counts['negative']/len(rows)*100,1) if rows else 0,
            'average_rating':round(sum(ratings)/len(ratings),2) if ratings else None,'rating_count':len(ratings),
            'low_confidence':low,'topics':themes,'available_topics':list(TOPICS)+['Genel'],
            'trend':[{'date':day,**{k:values[k] for k in ['positive','neutral','negative']}} for day,values in sorted(trend.items())],
            'undated':sum(row['date'] is None for row in rows),'keywords':[{'word':word,'count':n} for word,n in words.most_common(8)],
            'insights':insights,'reviews':rows[start:start+page_size],'page':page,'page_size':page_size,
            'pages':max(1,(len(rows)+page_size-1)//page_size)}


def csv_safe(value):
    text='' if value is None else str(value)
    return "'"+text if text.lstrip().startswith(('=','+','-','@','\t','\r')) else text


@app.get('/api/export')
def export(dataset_id: Optional[str]=None,sentiment: Optional[str]=None,topic: Optional[str]=None,q: Optional[str]=Query(None,max_length=200),date_from: Optional[date]=None,date_to: Optional[date]=None):
    rows=filtered_rows(dataset_id,sentiment,topic,q,date_from,date_to)
    stream=io.StringIO(); writer=csv.writer(stream)
    writer.writerow(['yorum','tarih','puan','duygu','model_skoru','konular','veri_seti','model_surumu'])
    for row in rows:
        writer.writerow([csv_safe(row[k]) for k in ['text','date','rating','sentiment','confidence']]+[csv_safe(' | '.join(row['topics'])),csv_safe(row['dataset_name']),csv_safe(row['model_version'])])
    return Response(stream.getvalue().encode('utf-8-sig'),media_type='text/csv',headers={'Content-Disposition':'attachment; filename="yorum-radar-analiz.csv"'})


if (ROOT/'frontend/dist').exists():
    app.mount('/',StaticFiles(directory=ROOT/'frontend/dist',html=True),name='frontend')
else:
    @app.get('/')
    def build_required():
        return {'message':'Frontend build gerekli: frontend klasöründe npm ci ve npm run build. Ya da BASLAT.cmd kullanın.'}
