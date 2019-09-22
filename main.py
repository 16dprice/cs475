import sys
import os
import time
sys.path.append(os.getcwd() + "/src/")

from html_parsing.parsing_aggregator import ParsingAggregator

start = time.time()

parsing_aggregator = ParsingAggregator()
parsing_aggregator.train_doc2vec_model()

parsing_aggregator.save_doc2vec_model()

end = time.time()

print(end - start)
