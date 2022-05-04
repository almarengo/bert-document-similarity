import transformers as ppb
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from scipy import spatial

def run_main():
    print('Loading the data')
    # Initialize the dictionary
    corpus_dic = {}
    # read each document line
    with open('data/corpus_coding_test.tsv', 'r', encoding='utf8') as lines:
        for line in lines:
            #split hte line at the TAB
            idx, text = line.split('\t')
            temp_list = []
            # Append the text to the temporary list
            temp_list.append(text)
            corpus_dic[idx] = temp_list
    
    print('Loading the model')
    # Load DistilBert Base case uncased
    model = ppb.DistilBertModel
    model = model.from_pretrained('distilbert-base-uncased')
    # Load Tokenizer for DistilBert Base case uncased
    tokenizer = ppb.DistilBertTokenizer
    tokenizer = tokenizer.from_pretrained('distilbert-base-uncased')
    print('Passing the data through the model')
    # Put the model on the device
    model = model.to(device)
    # Set the model in avaluation mode
    model.eval()
    # Loop through the dictionary
    for k, value in tqdm(corpus_dic.items()):
        # for each document we pass through the tokenizer first, max seq length allowed is 512 so we truncate what's longer than that
        tokenizerd = tokenizer.encode(value[0], add_special_tokens=True, truncation=True)
        # Create a tensor of the tokenized input
        input_ids = torch.tensor(np.array(tokenizerd)).type(torch.LongTensor).to(device)
        # Reshape tensor to have a batch size dimension
        input_ids = input_ids.unsqueeze(dim=0)
        # Pass it through the model
        with torch.no_grad():
            last_hidden_states = model(input_ids)
        # Extract CLS token at position 0
        vector = last_hidden_states[0][:, 0, :].cpu().numpy()
        # Append vector to list in dictionary's values
        value.append(vector)
    print('Calculating cosine similarities')
    # Loop through the dictionary
    for key, value in tqdm(corpus_dic.items()):
        # Initialize dictionary
        scores = {}
        # Loops through all the keys 
        for k in corpus_dic.keys():
            # Check that we are not comparing same documents
            if key != k:
                # Calculates cosine similarity
                dist = spatial.distance.cosine(value[1], corpus_dic[k][1])
                # Assignes cosine similarity value to documentID key
                scores[k] = dist
        value.append(scores)
    print('Creating the output')
    # Loop through the dictionary
    for key, value in tqdm(corpus_dic.items()):
        # Create a list of sorted cosine similarity values (third position in the dictionary's value)
        sorted_keys = sorted(value[2], key=value[2].get)
        # Append the top 5 list
        value.append(sorted_keys[:5])

    output = ''
    for key, value in tqdm(corpus_dic.items()):
        output += key
        for id_doc in value[3]:
            output += '\t'+id_doc+':'+str(round(value[2][id_doc], 3))
        output += '\n'
    output = output[:-1]

    with open('data/output.tsv', 'w') as txt_file:
        txt_file.write(output)

if __name__ == '__main__':
    run_main()