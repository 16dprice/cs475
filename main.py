import sys
import os
import time
sys.path.append(os.getcwd() + "/src/")

from html_parsing.parsing_aggregator import ParsingAggregator

start = time.time()

parsing_aggregator = ParsingAggregator()
parsing_aggregator.save_train_corpus("./src/html_parsing")

end = time.time()

print(end - start)
