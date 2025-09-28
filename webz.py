import requests
from key import WEBZ_KEY as NEWS_KEY
from urllib.parse import quote_plus
from typing import List
OR = " OR "
KEYWORDS = ["war","sanction","crisis","peace","agreement"]   # MAXIMUM OF 5 KEYWORDS
COUNTRY = "France"



def get_info(keywords : List[str], country : str) -> List[str]:
    """
    Get news about info in a specific country, given specific keywords
    """
    keyword_request = OR.join(keywords)
    query = quote_plus(f"language:english topic:economy (title:{country} AND ({keyword_request}))")
    news = requests.get(f"https://api.webz.io/newsApiLite?token={NEWS_KEY}&q={query}")
    data = news.json()
    posts = data.get("posts")
    print(data.get("requestsLeft"))
    return [post.get("title") for post in posts]
    

if __name__ == "__main__":
    print(get_info(KEYWORDS,COUNTRY))