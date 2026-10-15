"""Transparent, multi-label topic rules; separate from the trained sentiment model."""
from ml.model import normalize

TOPICS = {
    'Kargo & teslimat':['kargo','teslim','paket','dağıtım','kurye','geç geldi','gecik'],
    'Ürün kalitesi':['kalite','kaliteli','kalitesiz','bozuk','arız','kırıl','kırıldı','hasar','malzeme','kumaş','çizik','performans','eksik','pil'],
    'Fiyat & ödeme':['fiyat','pahalı','ucuz','indirim','ödeme','para','ücret','fatura'],
    'Müşteri hizmetleri':['müşteri','temsilci','destek','hizmet','satıcı','servis','telefon','cevap','ilgisiz'],
    'İade & değişim':['iade','değişim','değiştir','yanlış beden'],
    'Uygulama deneyimi':['uygulama','arayüz','hesabıma','sepet','çök','bildirim','giremiyorum']
}


def detect_topics(text):
    normalized=normalize(text)
    found=[name for name,keywords in TOPICS.items() if any(word in normalized for word in keywords)]
    return found or ['Genel']
