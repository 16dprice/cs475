import nltk
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup as soup


url = "https://www.coolstuffinc.com/a/jimdavis-09162019-sending-the-cats-to-the-astrolabe"
url2 = "https://www.coolstuffinc.com/a/jimdavis-09132019-force-of-negation-is-the-most-important-card-in-modern"

req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
webpage = urlopen(req).read()

page_soup = soup(webpage, "html.parser")

content = page_soup.find_all("section", class_="gm-article-content")

if len(content) == 1:
    print(content[0].encode_contents())

clean_text = ' '.join(soup(content[0].encode_contents(), "html.parser").stripped_strings)
print(clean_text)

# ----------------------------------------------------------------------------------------------------------------------

req = Request(url2, headers={'User-Agent': 'Mozilla/5.0'})
webpage = urlopen(req).read()

page_soup = soup(webpage, "html.parser")

content = page_soup.find_all("section", class_="gm-article-content")

if len(content) == 1:
    print(content[0].encode_contents())

clean_text2 = ' '.join(soup(content[0].encode_contents(), "html.parser").stripped_strings)
print(clean_text2)

