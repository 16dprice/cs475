from html_parsing import cool_stuff_inc_article_parser

links = [
    # Jim Davis
    "https://www.coolstuffinc.com/a/jimdavis-09202019-love-hates-for-throne-of-eldraine",
    "https://www.coolstuffinc.com/a/jimdavis-09162019-sending-the-cats-to-the-astrolabe",
    "https://www.coolstuffinc.com/a/jimdavis-09132019-force-of-negation-is-the-most-important-card-in-modern",
    "https://www.coolstuffinc.com/a/jimdavis-09092019-taking-stoneforge-mystic-in-a-darker-direction-with-stoneforge-pox",
    "https://www.coolstuffinc.com/a/jimdavis-09062019-faithless-looting-replacements-updating-dredge-izzet-phoenix-and-mardu-pyromancer",
    "https://www.coolstuffinc.com/a/jimdavis-09022019-casting-armageddon-in-standard",
    "https://www.coolstuffinc.com/a/jimdavis-08272019-banning-faithless-looting-in-modern-was-a-mistake",
    "https://www.coolstuffinc.com/a/jimdavis-08262019-breaking-standard-with-four-color-kethis-combo",
    "https://www.coolstuffinc.com/a/jimdavis-08232019-exploring-different-ways-to-play-on-a-digital-playground",
    "https://www.coolstuffinc.com/a/jimdavis-08192019-be-your-own-best-friend-with-chandra-tribal",
    "https://www.coolstuffinc.com/a/jimdavis-08162019-the-future-of-magic-esports-and-organized-play",
    "https://www.coolstuffinc.com/a/jimdavis-08122019-they-said-snow",
    "https://www.coolstuffinc.com/a/jimdavis-08092019-bored-of-standard-and-modern-try-these-brews",
    "https://www.coolstuffinc.com/a/jimdavis-08052019-reanimating-drakuseth-to-burninate-standard",
    "https://www.coolstuffinc.com/a/jimdavis-08022019-mc-prediction-results-and-goblins-sideboard-guide",
    "https://www.coolstuffinc.com/a/jimdavis-08012019-playing-goblins-at-the-mythic-championship",
    "https://www.coolstuffinc.com/a/jimdavis-07252019-five-mythic-championship-iv-predictions",
    "https://www.coolstuffinc.com/a/jimdavis-07222019-playing-hogaak-fair-in-modern-with-hogaak-pox",
    "https://www.coolstuffinc.com/a/jimdavis-07192019-the-verdict-on-cube-cards-from-core-set-2020-and-modern-horizons",
    "https://www.coolstuffinc.com/a/jimdavis-07152019-modern-vial-goblins",
    "https://www.coolstuffinc.com/a/jimdavis-07122019-combos-to-watch-for-in-week-one-of-core-set-2020-standard",
    "https://www.coolstuffinc.com/a/jimdavis-07082019-fate-shift-combo-in-standard",
    "https://www.coolstuffinc.com/a/jimdavis-07052019-ten-new-core-set-2020-standard-brews",
    "https://www.coolstuffinc.com/a/jimdavis-07012019-winning-on-turn-three-with-blister-burn",
    "https://www.coolstuffinc.com/a/jimdavis-06282019-love-hate-for-core-set-2020",
    "https://www.coolstuffinc.com/a/jimdavis-06242019-go-ninja-go-ninja-go",
    "https://www.coolstuffinc.com/a/jimdavis-06212019-first-look-at-core-set-2020",
    "https://www.coolstuffinc.com/a/jimdavis-06172019-arcades-alert-when-walls-attack",
    "https://www.coolstuffinc.com/a/jimdavis-06142019-splinter-twin-is-back-in-modern",
    "https://www.coolstuffinc.com/a/jimdavis-06112019-starting-with-an-idea-modern-four-color-saheeli-combo",
    "https://www.coolstuffinc.com/a/jimdavis-06072019-the-biggest-trends-in-standard-and-modern-for-scgcon",
    "https://www.coolstuffinc.com/a/jimdavis-06032019-putting-the-big-pig-to-work-in-modern",
    "https://www.coolstuffinc.com/a/jimdavis-05312019-the-complete-standard-four-color-command-the-dreadhorde-guide",
    "https://www.coolstuffinc.com/a/jimdavis-05272019-trying-to-punish-those-with-no-basics",
    "https://www.coolstuffinc.com/a/jimdavis-05242019-five-cards-that-are-secretly-excellent-in-standard-right-now",
    "https://www.coolstuffinc.com/a/jimdavis-05202019-a-force-to-be-reckoned-with",
    "https://www.coolstuffinc.com/a/jimdavis-05172019-four-standard-superfriends-brews",
    "https://www.coolstuffinc.com/a/jimdavis-05132019-killing-people-on-turn-two-with-neoform",
    "https://www.coolstuffinc.com/a/jimdavis-05102019-war-of-the-spark-week-one-planeswalker-report-card",
    "https://www.coolstuffinc.com/a/jimdavis-05062019-ahhhh-gruul-monsters",
    # Ali Aintrazi
    "https://www.coolstuffinc.com/a/aliaintrazi-09182019-watch-it-burn-preview-for-throne-of-eldraine",
    "https://www.coolstuffinc.com/a/aliaintrazi-09112019-best-of-one-golos-in-new-standard",
    "https://www.coolstuffinc.com/a/aliaintrazi-09062019-jund-dinosaurs-with-ali-aintrazi",
    "https://www.coolstuffinc.com/a/aliaintrazi-09042019-omniscience-draft-with-ali-aintrazi",
    "https://www.coolstuffinc.com/a/aliaintrazi-08302019-fandom-legends-1st-place-five-color-golos-with-ali-aintrazi",
    "https://www.coolstuffinc.com/a/aliaintrazi-08282019-battle-of-one-grixis-artifacts",
    "https://www.coolstuffinc.com/a/aliaintrazi-08212019-battle-of-one-big-blue",
    "https://www.coolstuffinc.com/a/aliaintrazi-08142019-battle-of-one-kykars-angels",
    "https://www.coolstuffinc.com/a/aliaintrazi-08092019-standard-temur-elementals",
    "https://www.coolstuffinc.com/a/aliaintrazi-08072019-battle-of-one-orzhov-aristocrats"
]

file = open("./for_uta.txt", "w")

for doc in links:
    parser = cool_stuff_inc_article_parser.CoolStuffIncArticleParser(doc)
    file.write(parser.get_clean_text())
    file.write('\n')