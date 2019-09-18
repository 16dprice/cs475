

url1 = "https://www.coolstuffinc.com/a/jimdavis-09162019-sending-the-cats-to-the-astrolabe"
url2 = "https://www.coolstuffinc.com/a/jimdavis-09132019-force-of-negation-is-the-most-important-card-in-modern"
url3 = "https://www.coolstuffinc.com/a/jimdavis-09092019-taking-stoneforge-mystic-in-a-darker-direction-with-stoneforge-pox"

parser = Cool_Stuff_Inc_Article_Parser(url3)

print(parser.get_clean_text())
print(parser.get_tokenized_text())