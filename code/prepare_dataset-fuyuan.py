import glob
import itertools
import json
import re
import os
import pandas as pd
import pickle
import xgboost as xgb

venues = ['iclr_2017']
partitions = ['dev', 'test', 'train']
data_folders = ['parsed_pdfs', 'reviews']

data_types = [('parsed_pdfs', r'/([0-9.]+).pdf.json$'), ('reviews', r'/([0-9.]+).json$')]
# raw_data = {'parsed_pdfs': [],'reviews': []}
papers = []
reviews = []

# main_path = 'C:/Users/lfy_1/OneDrive - McGill University/McGill Class/COMP 550 NLP/project/COMP550-project'

for venue in venues:
    for partition in partitions:
        for data_folder in data_folders:
            path = 'C:/Users/lfy_1/OneDrive - McGill University/McGill Class/COMP 550 NLP/project/COMP550-project/dataset/' + venue + '/' + partition + '/' + data_folder + '/'
            for file in os.listdir(path):
                with open(path + file, encoding="utf8") as json_file:
                    text_data = json_file.read()
                    try:
                        json_data = json.loads(text_data)
                        if data_folder == 'parsed_pdfs':
                            temp_id = int(file[0:3])
                            temp_dict = {'id': temp_id,'partition': partition, 'data_folder': data_folder, 'json': json_data}
                            papers.append(temp_dict)
                        elif data_folder == 'reviews':
                            temp_id = int(file[0:3])
                            temp_dict = {'id': temp_id,'partition': partition, 'data_folder': data_folder, 'json': json_data}
                            reviews.append(temp_dict)
                    except:
                        print(partition, data_folder, file)


print("Total papers\t" + str(len(papers)))
print("Total reviews\t" + str(len(reviews)))


train_x = []
train_y = []
test_x = []
test_y = []
dev_x = []
dev_y = []

for index, item in enumerate(papers):
    if item['partition'] == "train":
        train_x.append(papers[index]['json']['metadata']['abstractText'])
        train_y.append(int(reviews[index]['json']['accepted']))
    elif item['partition'] == "test":
        test_x.append(papers[index]['json']['metadata']['abstractText'])
        test_y.append(int(reviews[index]['json']['accepted']))
    elif item['partition'] == 'dev':
        dev_x.append(papers[index]['json']['metadata']['abstractText'])
        dev_y.append(int(reviews[index]['json']['accepted']))

print("Training samples:\t" + str(len(train_x)))
print("Testing samples:\t" + str(len(test_x)))
print("Development samples:\t" + str(len(dev_x)))

dataset = {"train": (train_x, train_y), 'test': (test_x, test_y), 'dev': (dev_x, dev_y)}

with open('../dataset/abstract.pickle', 'wb') as output:
    pickle.dump(dataset, output)





train_x = []
train_y = []
test_x = []
test_y = []
dev_x = []
dev_y = []


for index, item in enumerate(papers):
    text = papers[index]['json']['metadata']['abstractText']
    if papers[index]['json']['metadata']['sections']:
        for item2 in papers[index]['json']['metadata']['sections']:
            text += item2['text']

    if item['partition'] == "train":
        train_x.append(text)
        train_y.append(int(reviews[index]['json']['accepted']))
    elif item['partition'] == "test":
        test_x.append(text)
        test_y.append(int(reviews[index]['json']['accepted']))
    elif item['partition'] == 'dev':
        dev_x.append(text)
        dev_y.append(int(reviews[index]['json']['accepted']))

print("Training samples:\t" + str(len(train_x)))
print("Testing samples:\t" + str(len(test_x)))
print("Development samples:\t" + str(len(dev_x)))

dataset = {"train": (train_x, train_y), 'test': (test_x, test_y), 'dev': (dev_x, dev_y)}

with open('../dataset/all.pickle', 'wb') as output:
    pickle.dump(dataset, output)





train_x = []
train_y = []
test_x = []
test_y = []
dev_x = []
dev_y = []


for index, item in enumerate(papers):
    text = papers[index]['json']['metadata']['abstractText']
    if papers[index]['json']['metadata']['sections']:
        for item2 in papers[index]['json']['metadata']['sections']:
            if item2['heading'] and "introduction" in item2['heading'].lower():
                text += item2['text']

    if item['partition'] == "train":
        train_x.append(text)
        train_y.append(int(reviews[index]['json']['accepted']))
    elif item['partition'] == "test":
        test_x.append(text)
        test_y.append(int(reviews[index]['json']['accepted']))
    elif item['partition'] == 'dev':
        dev_x.append(text)
        dev_y.append(int(reviews[index]['json']['accepted']))

print("Training samples:\t" + str(len(train_x)))
print("Testing samples:\t" + str(len(test_x)))
print("Development samples:\t" + str(len(dev_x)))

dataset = {"train": (train_x, train_y), 'test': (test_x, test_y), 'dev': (dev_x, dev_y)}

with open('../dataset/abstract+introduction.pickle', 'wb') as output:
    pickle.dump(dataset, output)

