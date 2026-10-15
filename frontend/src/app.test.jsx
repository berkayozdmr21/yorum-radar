import React from 'react';
import {render,screen,cleanup,waitFor,fireEvent} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {vi,beforeEach,afterEach,test,expect} from 'vitest';
import {App} from './main.jsx';

const dashboard={total:0,counts:{positive:0,negative:0,neutral:0},topics:[],available_topics:['Kargo & teslimat'],trend:[],keywords:[],insights:[],reviews:[],page:1,pages:1,page_size:10,negative_rate:0,low_confidence:0,average_rating:null,undated:0};
const model={model:'logistic_regression',train_rows:120,test_rows:36,trained_at:'2026-10-15T12:00:00Z',metrics:{}};
let fetchMock;
beforeEach(()=>{
  fetchMock=vi.fn(async (url,options={})=>{
    let body={};
    if(url.startsWith('/api/dashboard')) body=dashboard;
    else if(url==='/api/datasets') body=[];
    else if(url==='/api/model') body=model;
    else if(url==='/api/preview') body={columns:['yorum','tarih','puan'],preview:[{yorum:'Örnek yorum',tarih:'2026-10-15',puan:'5'}],row_count:1};
    else if(url==='/api/import') body={id:'uploaded',rows:1};
    else if(url==='/api/predict') body={sentiment:'negative',confidence:.8,probabilities:{negative:.8,neutral:.1,positive:.1},topics:['Kargo & teslimat'],evidence:[{term:'geç',weight:.3}],latency_ms:4};
    return {ok:true,status:200,json:async()=>body};
  });
  vi.stubGlobal('fetch',fetchMock);
});
afterEach(()=>{cleanup();vi.unstubAllGlobals();});

test('empty dashboard offers a real import path',async()=>{
  render(<App/>);
  await screen.findByText('Yorumların burada anlam kazanacak.');
  expect(screen.getByRole('button',{name:'CSV yükle'})).toBeTruthy();
  expect(screen.getByRole('link',{name:'Dışa aktar'}).getAttribute('href')).toBe('/api/export?');
});

test('CSV mapping submits selected columns and refreshes dataset',async()=>{
  const user=userEvent.setup();render(<App/>);
  await user.click(screen.getByRole('button',{name:'CSV yükle'}));
  const dialog=screen.getByRole('dialog');
  const file=new File(['yorum,tarih,puan\nÖrnek yorum,2026-10-15,5'], 'reviews.csv',{type:'text/csv'});
  await user.upload(dialog.querySelector('input[type=file]'),file);
  await screen.findByText('1 satır · İlk 3 yorum');
  await user.click(screen.getByRole('button',{name:'Yükle ve analiz et'}));
  await waitFor(()=>expect(screen.queryByRole('dialog')).toBeNull());
  const call=fetchMock.mock.calls.find(([url])=>url==='/api/import');
  expect(call[1].body.get('text_column')).toBe('yorum');
  expect(call[1].body.get('date_column')).toBe('tarih');
  expect(call[1].body.get('rating_column')).toBe('puan');
  await waitFor(()=>expect(fetchMock.mock.calls.some(([url])=>url.includes('dataset_id=uploaded'))).toBe(true));
});

test('lab shows sentiment prediction and explanatory terms',async()=>{
  const user=userEvent.setup();render(<App/>);
  await user.click(screen.getByRole('button',{name:'Analiz laboratuvarı'}));
  fireEvent.change(screen.getByRole('textbox',{name:'Analiz edilecek yorum'}),{target:{value:'Kargo çok geç geldi'}});
  await user.click(screen.getByRole('button',{name:'Analiz et'}));
  await screen.findByText('Tahmini destekleyen kelimeler');
  expect(screen.getByText('geç')).toBeTruthy();
  const call=fetchMock.mock.calls.find(([url])=>url==='/api/predict');
  expect(JSON.parse(call[1].body).text).toBe('Kargo çok geç geldi');
});

test('sentiment filter is applied to dashboard and export',async()=>{
  render(<App/>);
  fireEvent.change(screen.getByRole('combobox',{name:'Duygu filtresi'}),{target:{value:'negative'}});
  await waitFor(()=>expect(fetchMock.mock.calls.some(([url])=>url.includes('sentiment=negative'))).toBe(true));
  expect(screen.getByRole('link',{name:'Dışa aktar'}).getAttribute('href')).toContain('sentiment=negative');
});

test('upload dialog closes with Escape',async()=>{
  const user=userEvent.setup();render(<App/>);
  await user.click(screen.getByRole('button',{name:'CSV yükle'}));
  expect(screen.getByRole('dialog')).toBeTruthy();
  await user.keyboard('{Escape}');
  expect(screen.queryByRole('dialog')).toBeNull();
});

test('API errors are displayed to the user',async()=>{
  fetchMock.mockImplementation(async()=>({ok:false,status:503,json:async()=>({detail:'Sunucu kullanılamıyor'})}));
  render(<App/>);
  await screen.findByText('Sunucu kullanılamıyor');
});
