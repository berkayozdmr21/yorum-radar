import csv
import io
import os
import tempfile
import unittest
from pathlib import Path

Path('data').mkdir(exist_ok=True)
scratch=tempfile.TemporaryDirectory(dir='data')
os.environ['YORUM_DATA_DIR']=str(Path(scratch.name)/'data')
os.environ['YORUM_ARTIFACT_DIR']=str(Path(scratch.name)/'artifacts')
from fastapi.testclient import TestClient
from backend.main import app,csv_safe
from backend.topics import detect_topics
from ml.model import normalize,read_labeled,train


class APITests(unittest.TestCase):
    def setUp(self):
        self.client=TestClient(app)
        for row in self.client.get('/api/datasets').json():
            self.client.delete('/api/datasets/'+row['id'])

    def upload(self,text,**mapping):
        return self.client.post('/api/import',files={'file':('reviews.csv',text.encode('utf-8-sig'),'text/csv')},data={'text_column':'yorum',**mapping})

    def test_health_and_model_report(self):
        self.assertFalse(self.client.get('/api/health').json()['api_key_required'])
        report=self.client.get('/api/model').json()
        self.assertEqual(report['train_rows'],120)
        self.assertEqual(report['test_rows'],36)
        self.assertIn('multinomial_nb',report['metrics'])

    def test_local_predict_three_classes(self):
        for text,label in [('Çok memnun kaldım teşekkürler','positive'),('Ürün bozuk geldi çok kötü','negative'),('Ürünün ölçüleri nedir?','neutral')]:
            result=self.client.post('/api/predict',json={'text':text}).json()
            self.assertEqual(result['sentiment'],label)
            self.assertAlmostEqual(sum(result['probabilities'].values()),1,places=3)
            self.assertIsInstance(result['evidence'],list)

    def test_predict_validation(self):
        self.assertEqual(self.client.post('/api/predict',json={'text':'  '}).status_code,422)
        self.assertEqual(self.client.post('/api/predict',json={'text':'x'*3001}).status_code,422)

    def test_preview_semicolon_and_cp1254(self):
        text='yorum;tarih;puan\nÜrün güzel;2026-10-15;5\n'
        response=self.client.post('/api/preview',files={'file':('x.csv',text.encode('cp1254'),'text/csv')})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()['columns'],['yorum','tarih','puan'])
        self.assertEqual(response.json()['preview'][0]['yorum'],'Ürün güzel')

    def test_import_aggregation_and_filters(self):
        result=self.upload('yorum,tarih,puan\nÜrün harika kalitesi çok iyi,2026-10-10,5\nKargo çok geç geldi,2026-10-11,1\nÜrünün ölçüleri nedir?,2026-10-12,3\n',date_column='tarih',rating_column='puan')
        self.assertEqual(result.status_code,201,result.text)
        ds=result.json()['id']
        dashboard=self.client.get('/api/dashboard').json()
        self.assertEqual(dashboard['total'],3)
        self.assertEqual(sum(dashboard['counts'].values()),3)
        self.assertEqual(dashboard['average_rating'],3)
        self.assertEqual(len(dashboard['trend']),3)
        negative=self.client.get('/api/dashboard',params={'sentiment':'negative'}).json()
        self.assertEqual(negative['total'],1)
        self.assertEqual(self.client.get('/api/dashboard',params={'topic':'Kargo & teslimat'}).json()['total'],1)
        self.assertEqual(self.client.get('/api/dashboard',params={'date_from':'2026-10-11','date_to':'2026-10-11'}).json()['total'],1)
        self.assertEqual(self.client.get('/api/dashboard',params={'q':'ölçüleri'}).json()['total'],1)
        self.assertEqual(self.client.get('/api/dashboard',params={'dataset_id':'other'}).json()['total'],0)
        self.assertEqual(self.client.delete('/api/datasets/'+ds).status_code,204)
        self.assertEqual(self.client.get('/api/dashboard').json()['total'],0)

    def test_import_blank_and_bad_metadata(self):
        result=self.upload('yorum,tarih,puan\nİyi bir ürün,not-a-date,100\n,,\n',date_column='tarih',rating_column='puan')
        self.assertEqual(result.status_code,201,result.text)
        self.assertEqual(result.json()['rows'],1)
        self.assertEqual(result.json()['skipped_blank'],1)
        self.assertEqual(result.json()['invalid_dates'],1)
        self.assertEqual(result.json()['invalid_ratings'],1)
        self.assertIsNone(self.client.get('/api/dashboard').json()['average_rating'])

    def test_csv_schema_and_limits(self):
        for text in ['yorum,yorum\na,b\n','yorum\n','yorum\na,b\n']:
            response=self.client.post('/api/preview',files={'file':('x.csv',text.encode(),'text/csv')})
            self.assertEqual(response.status_code,422,response.text)
        self.assertEqual(self.upload('yorum\n'+('x'*3001)+'\n').status_code,422)
        self.assertEqual(self.upload('yorum\nörnek\n',date_column='missing').status_code,422)
        self.assertEqual(self.upload('yorum\n'+'örnek\n'*5001).status_code,413)
        self.assertEqual(self.client.post('/api/preview',files={'file':('x.csv',b'x'* (5*1024*1024+1),'text/csv')}).status_code,413)

    def test_demo_idempotency_and_pagination(self):
        first=self.client.post('/api/demo').json()
        second=self.client.post('/api/demo').json()
        self.assertEqual(first['id'],second['id'])
        self.assertEqual(first['rows'],36)
        result=self.client.get('/api/dashboard',params={'page':2,'page_size':10}).json()
        self.assertEqual(len(result['reviews']),10)
        self.assertEqual(result['pages'],4)
        self.assertEqual(self.client.get('/api/dashboard',params={'page':0}).status_code,422)

    def test_export_filters_and_formula_safety(self):
        self.upload('yorum\n"=HYPERLINK(\"\"evil\"\")"\nGüzel ürün\n')
        response=self.client.get('/api/export',params={'q':'HYPERLINK'})
        self.assertEqual(response.status_code,200)
        parsed=list(csv.DictReader(io.StringIO(response.content.decode('utf-8-sig'))))
        self.assertEqual(len(parsed),1)
        self.assertTrue(parsed[0]['yorum'].startswith("'="))
        self.assertTrue(csv_safe('  =1+1').startswith("'"))

    def test_invalid_filters_and_missing_delete(self):
        self.assertEqual(self.client.get('/api/dashboard',params={'sentiment':'bad'}).status_code,422)
        self.assertEqual(self.client.get('/api/dashboard',params={'date_from':'2026-10-15','date_to':'2026-10-01'}).status_code,422)
        self.assertEqual(self.client.delete('/api/datasets/missing').status_code,404)

    def test_literal_search_and_sql_parameterization(self):
        self.upload('yorum\n%100 iyi\nNormal yorum\n')
        self.assertEqual(self.client.get('/api/dashboard',params={'q':'%'}).json()['total'],1)
        self.assertEqual(self.client.get('/api/dashboard',params={'q':"' OR 1=1 --"}).json()['total'],0)

    def test_topics_multi_label(self):
        topics=detect_topics('Kargo hasarlı geldi ve iade ücretim yatırılmadı')
        self.assertIn('Kargo & teslimat',topics)
        self.assertIn('İade & değişim',topics)
        self.assertIn('Fiyat & ödeme',topics)
        self.assertEqual(detect_topics('Sıradan bir yorum'),['Genel'])


class TrainingTests(unittest.TestCase):
    def test_normalization_and_no_leakage(self):
        self.assertEqual(normalize(' İYİ  IŞIK '),'iyi ışık')
        rows=read_labeled('datasets/train.csv')
        holdout=read_labeled('datasets/holdout.csv')
        self.assertFalse({normalize(t) for t,_ in rows}&{normalize(t) for t,_ in holdout})

    def test_train_refuses_test_overlap(self):
        with tempfile.TemporaryDirectory(dir='data') as tmp:
            with self.assertRaisesRegex(ValueError,'çakışıyor'):
                train('datasets/train.csv','datasets/train.csv',tmp)

    def test_train_rejects_conflicting_labels(self):
        path=Path(scratch.name)/'conflict.csv'
        path.write_text('text,label\naynı yorum,positive\nAYNI YORUM,negative\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'çelişen'):
            read_labeled(path)

    def test_single_class_training_rejected(self):
        path=Path(scratch.name)/'oneclass.csv'
        path.write_text('text,label\na,positive\nb,positive\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'Üç sınıf'):
            train(path,None,Path(scratch.name)/'bad-artifact')


if __name__=='__main__': unittest.main()
