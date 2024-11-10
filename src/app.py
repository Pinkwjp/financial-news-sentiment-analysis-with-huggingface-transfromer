import pandas as pd

from src.utils import fetch_company_news, create_classifier_pipeline


def main(company: str, lang='en', reigon='US'):
    news_df = fetch_company_news(company=company, lang=lang, reigon=reigon)
    classifier = create_classifier_pipeline()
    sentiments = classifier(news_df['title'].to_list())
    news_and_sentiments_df = pd.concat([news_df, pd.DataFrame(sentiments)], axis=1)
    return news_and_sentiments_df


