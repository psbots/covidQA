import datetime
import random

# import bokeh
# import bokeh.layouts
# import bokeh.models
# import bokeh.plotting
import markdown
# import pandas as pd
import streamlit as st

import requests

import torch
from transformers import BertForQuestionAnswering,BertTokenizer, BertModel, AutoTokenizer, AutoModelForQuestionAnswering
# import umap

tokenizer = AutoTokenizer.from_pretrained("ktrapeznikov/biobert_v1.1_pubmed_squad_v2") 
model = AutoModelForQuestionAnswering.from_pretrained("ktrapeznikov/biobert_v1.1_pubmed_squad_v2")
model.eval()
# url = "https://cord-19.apps.allenai.org/api/meta/search"
url = "https://api.cord19.vespa.ai/search/"
payload = {"query":"+(",
            "summary":"full","hits":20}
headers= {}

def embed_text(text, model):
    input_ids = torch.tensor(tokenizer.encode(text[:512])).unsqueeze(0)  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    return last_hidden_states 

def make_the_embeds(results, key="contents"):
    title_embedding_list = []
    title_list = []
    for hit in results:
        title = hit["fields"]["title-full"]
        data = hit["fields"]["body_text-full"]
        try:
            tensor = make_data_embedding(data)
            title_embedding_list.append(tensor)
            title_list.append(title)
        except:
            print("Invalid title/abstract")
    return torch.cat(title_embedding_list, dim=0), title_list
    
        
def make_data_embedding(article_data, method="mean", dim=1):
    data = article_data
    text = embed_text(data, model)
    if method == "mean":
        return text.mean(dim)

class Format:
    # end = '\033[0m'
    # underline = '\033[4m'
    end = '</b>'
    underline = '<b>'

def answer_question(question, answer_text):
    '''
    Takes a `question` string and an `answer_text` string (which contains the
    answer), and identifies the words within the `answer_text` that are the
    answer. Prints them out.
    '''
    # ======== Tokenize ========
    # Apply the tokenizer to the input text, treating them as a text-pair.
    input_ids = tokenizer.encode(question, answer_text,max_length=512)
    #print("input here: ",answer_text)

    # Report how long the input sequence is.
    #print('Query has {:,} tokens'.format(len(input_ids)))

    # ======== Set Segment IDs ========
    # Search the input_ids for the first instance of the `[SEP]` token.
    sep_index = input_ids.index(tokenizer.sep_token_id)

    # The number of segment A tokens includes the [SEP] token istelf.
    num_seg_a = sep_index + 1

    # The remainder are segment B.
    num_seg_b = len(input_ids) - num_seg_a

    # Construct the list of 0s and 1s.
    segment_ids = [0]*num_seg_a + [1]*num_seg_b

    # There should be a segment_id for every input token.
    assert len(segment_ids) == len(input_ids)

    # ======== Evaluate ========
    # Run our example question through the model.
    start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.
                                    token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text

    # ======== Reconstruct Answer ========
    # Find the tokens with the highest `start` and `end` scores.
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)

    # Get the string versions of the input tokens.
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # Start with the first token.
    answer = ""
    short = ""
    #print(tokens)
    # Select the remaining answer tokens and join them with whitespace.
    for i in range(tokens.index('[SEP]')+1,len(tokens)-1):
        #print(i, " ", tokens[i])
        # If it's a subword token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            if(i>=answer_start and i<=answer_end):
                answer+= Format.underline + tokens[i][2:] + Format.end
                short+=tokens[i][2:]
            else:
                answer+= tokens[i][2:]
        
        # Otherwise, add a space then the token.
        else:
            if(i>=answer_start and i<=answer_end):
                answer+= ' '+ Format.underline + tokens[i] + Format.end
                short+=' ' + tokens[i]
            else:
                answer+= ' ' + tokens[i]
            #tokens[:answer_start] + Format.underline + answer + Format.end + tokens[answer_end:]
    return(answer, short)
    
    #print('Answer: "' + answer + '"')

#Search for documents matching any of query terms (either in title or abstract)
search_request_any = {
  'yql': 'select id,title,abstract,body_text,title_embedding, doi from sources * where userQuery() and has_full_text=true;',
  'hits': 5,
  'summary': 'short',
  'timeout': '1.0s',
  'default-index': 'all',
  'query': 'coronavirus temperature sensitivity',
  'type': 'any',
  'ranking': 'default'
}
endpoint='https://api.cord19.vespa.ai/search/'

search_text = st.text_input("Ask your question related to COVID19")

if search_text:
    payload["query"] = payload["query"] + '"' + '" "'.join(search_text.split(" ")) + '")'
    # st.write(payload["query"])
    response = requests.request("GET", url, headers=headers, params = payload)
    result = response.json()
    st.write(result)
    for hit in result["root"]["children"]:
        # st.text(hit["fields"]["title-full"])
        # st.write(hit["fields"]["abstract-full"].replace("<hi>","<i><b>").replace("</hi>","</i></b>"),unsafe_allow_html=True)
        # st.write(hit["fields"]["body_text-full"],unsafe_allow_html=True)
        body_text = hit["fields"].get("body_text-full", None)
        if body_text:
            st.write("<b>"+hit["fields"]["title-full"]+"</b>",unsafe_allow_html=True)
            ans,short = answer_question(search_text,body_text)
            st.write("Answer : "+"<i><u>"+short+"</u></i>",unsafe_allow_html=True)
            st.write(ans,unsafe_allow_html=True)