import PdfConverter
import os

converter = PdfConverter.PdfConverter("./dance_pdfs/bharatanatyam_and_nonverbal_communication.pdf")

train_corpus = "../aggregate_train_corpus.txt"
train_file = open(train_corpus, "a+")

# for file in os.scandir("./dance_pdfs"):
#     converter = PdfConverter.PdfConverter(file.path)
#     try:
#         train_file.write("\n" + ' '.join(converter.convert_pdf_to_txt().splitlines()))
#     except:
#         print("Cant write {}".format(file.path))

train_file.close()
