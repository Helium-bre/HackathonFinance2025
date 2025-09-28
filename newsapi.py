from key import NEWS_API_KEY as NEWS_KEY
from newsapi import NewsApiClient

# Init
newsapi = NewsApiClient(api_key=NEWS_KEY)

# /v2/top-headlines
top_headlines = newsapi.get_top_headlines(country="cn")

# /v2/everything
# all_articles = newsapi.get_everything(q='Trump',
#                                       sources='bbc-news,the-verge',
#                                       domains='bbc.co.uk,techcrunch.com',
#                                       from_param='2025-09-01',
#                                       to='2025-09-26',
#                                       language='en',
#                                       sort_by='relevancy',
#                                       page=2)

# /v2/top-headlines/sources
#print(all_articles.get("articles"))
articles = top_headlines.get("articles")
print(articles)
print(top_headlines)
#sources = newsapi.get_sources()