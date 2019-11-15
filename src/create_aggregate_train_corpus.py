from html_parsing import parsing_aggregator
from pdf_reading import PdfConverter
import os


################################################## Parsing Aggregator ##################################################


aggregator = parsing_aggregator.ParsingAggregator()
aggregator.save_train_corpus(".")


######################################################### PDFs #########################################################


train_corpus = "./aggregate_train_corpus.txt"
pdf_corpus = "./bharatanatyam_pdfs.txt"

train_file = open(train_corpus, "a+")
bharatanatyam_file = open(pdf_corpus, "w")

for file in os.scandir("./pdf_reading/dance_pdfs"):
    converter = PdfConverter.PdfConverter(file.path)
    try:
        train_file.write("\n" + ' '.join(converter.convert_pdf_to_txt().splitlines()))
        bharatanatyam_file.write("\n" + ' '.join(converter.convert_pdf_to_txt().splitlines()))
    except:
        print("Cant write {}".format(file.path))

train_file.close()
bharatanatyam_file.close()