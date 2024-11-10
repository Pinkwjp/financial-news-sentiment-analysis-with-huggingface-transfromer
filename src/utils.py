from transformers import pipeline 
from GoogleNews import GoogleNews


def fetch_company_news(company: str, lang='en', reigon='US') -> list:
    """to ensure better search results, use a specific company name"""
    googlenews = GoogleNews()
    googlenews.enableException(True)
    googlenews = GoogleNews(lang=lang, region=reigon)
    googlenews.clear()
    googlenews.get_news(company)
    return googlenews.results()




# link for the model:
# https://huggingface.co/mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
MODEL_ID = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

def created_classifier_pipeline():
    return pipeline("text-classification", model=MODEL_ID)