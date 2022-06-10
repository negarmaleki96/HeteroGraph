#!/usr/bin/env python
# coding: utf-8

# import libraries
import torch
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
import re
import tensorflow as tf
import tensorflow_hub as hub
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
from torchmetrics import ConfusionMatrix
from sklearn.metrics import classification_report, accuracy_score

# read networkX graphs
G_train = nx.read_gpickle("~/gtrain.gpickle")
G_val = nx.read_gpickle("~/gval.gpickle")
G_test = nx.read_gpickle("~/gtest.gpickle")

# function to get node index
def get_key_node(val, G):
    key_indx_node = []
    for key, value in nx.get_node_attributes(G,'otype').items():
         if val == value:
            key_indx_node.append(key)
    return key_indx_node

# function to get edge index
def get_key_edge(val, G):
    key_indx_edge = []
    for key, value in nx.get_edge_attributes(G,'otype').items():
         if val == value:
            key_indx_edge.append(key)
    return key_indx_edge

# indices for each type of dict
# train
post_train = get_key_node('post', G_train)
comment_train = get_key_node('comment', G_train)
user_train = get_key_node('user', G_train)
vote_train = get_key_edge('vote', G_train)
authored_train = get_key_edge('authored', G_train)
reply_train = get_key_edge('reply', G_train)

# validation
post_val = get_key_node('post', G_val)
comment_val = get_key_node('comment', G_val)
user_val = get_key_node('user', G_val)
vote_val = get_key_edge('vote', G_val)
authored_val = get_key_edge('authored', G_val)
reply_val = get_key_edge('reply', G_val)

# test
post_test = get_key_node('post', G_test)
comment_test = get_key_node('comment', G_test)
user_test = get_key_node('user', G_test)
vote_test = get_key_edge('vote', G_test)
authored_test = get_key_edge('authored', G_test)
reply_test = get_key_edge('reply', G_test)

# Votes and comments between t1 and t2
def time_vote(vote, reply, post, comment, user, authored, G):
    subG_vote = []
    for i in vote:
        if G.edges()[i]['time'] >= G.nodes()[i[1]]['created'] and G.edges()[i]['time'] <= G.nodes()[i[1]]['last_payout']:
            subG_vote.append(i)

    subG_reply = []
    for i in reply:
        if G.edges()[i]['time'] >= G.nodes()[i[0]]['created'] and G.edges()[i]['time'] <= G.nodes()[i[0]]['last_payout']:
            subG_reply.append(i)

    G_node = G.subgraph(post + comment + user)
    Graph = G_node.edge_subgraph(subG_vote + authored + subG_reply)
    return nx.Graph(Graph)

G_train = time_vote(vote_train, reply_train, post_train, comment_train, user_train, authored_train, G_train)
G_val = time_vote(vote_val, reply_val, post_val, comment_val, user_val, authored_val, G_val)
G_test = time_vote(vote_test, reply_test, post_test, comment_test, user_test, authored_test, G_test)

# relabeling the graph after omitting votes and comments that weren't in the given time interval
def relable(G):
    keys = np.array(list(G.nodes))
    values = [int(i) for i in np.arange(0, len(G.nodes))]
    dic = dict(zip(keys, values))
    return nx.relabel_nodes(G, dic)

G_train = relable(G_train)
G_val = relable(G_val)
G_test = relable(G_test)

# Getting features out of each type of node
# train
p_otype_train = []
p_body_train = []
p_payout_train = []

u_otype_train = []

c_otype_train = []
c_body_train = []

for i in range(len(G_train.nodes())):
    if G_train.nodes()[i]['otype']=='post':
        p_otype_train.append(1)
        p_body_train.append(G_train.nodes[i]['body'][0])
        p_payout_train.append(int(G_train.nodes[i]['payout']))

    elif G_train.nodes()[i]['otype']=='user':
        u_otype_train.append(0)

    else:
        c_otype_train.append(2)
        c_body_train.append(G_train.nodes[i]['body'][0])

# validation
p_otype_val = []
p_body_val = []
p_payout_val = []

u_otype_val = []

c_otype_val = []
c_body_val = []

for i in range(len(G_val.nodes())):
    if G_val.nodes()[i]['otype']=='post':
        p_otype_val.append(1)
        p_body_val.append(G_val.nodes[i]['body'][0])
        p_payout_val.append(int(G_val.nodes[i]['payout']))     # [] removed to have y=[] instead of y=[ ,1]

    elif G_val.nodes()[i]['otype']=='user':
        u_otype_val.append(0)

    else:
        c_otype_val.append(2)
        c_body_val.append(G_val.nodes[i]['body'][0])

# test
p_otype_test = []
p_body_test = []
p_payout_test = []

u_otype_test = []

c_otype_test = []
c_body_test = []

for i in range(len(G_test.nodes())):
    if G_test.nodes()[i]['otype']=='post':
        p_otype_test.append(1)
        p_body_test.append(G_test.nodes[i]['body'][0])
        p_payout_test.append(int(G_test.nodes[i]['payout']))     # [] removed to have y=[] instead of y=[ ,1]

    elif G_test.nodes()[i]['otype']=='user':
        u_otype_test.append(0)

    else:
        c_otype_test.append(2)
        c_body_test.append(G_test.nodes[i]['body'][0])

# downloading stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('omw-1.4')

# preprocessing body contents
def preprocessing(list_of_body):
    new_p_body_train = []

    for text in list_of_body:
        text = re.sub("[^a-zA-Z]", " ", str(text))
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))', '', text, flags=re.MULTILINE)
        text = re.sub(r'(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*', '', text)
        text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http', '', text)
        text = re.sub(r'www', '', text)
        text = re.sub(r'com', '', text)
        text = re.sub(r'html', '', text)
        text = re.sub(r'iframe', '', text)
        text = re.sub(r'src', '', text)
        text = re.sub(r'href', '', text)
        text = re.sub(r'jpg', '', text)
        text = re.sub(r'jpeg', '', text)
        text = re.sub(r'png', '', text)
        text = re.sub(r'mkv', '', text)
        text = re.sub(r'[!"?@#$%&()*+,-./:;<=>?[\]^_`{|}~]', ' ', text).lower()

        words = nltk.tokenize.word_tokenize(text)
        words = [w for w in words if w.isalpha()]
        words = [w for w in words if len(w)>2 and w not in stopwords.words('english')]

        lemmatizer = nltk.stem.WordNetLemmatizer()
        words = [lemmatizer.lemmatize(w) for w in words]
        new_p_body_train.append(' '.join(words))

    return new_p_body_train

# train
p_body_train = preprocessing(p_body_train)
c_body_train = preprocessing(c_body_train)

# validation
p_body_val = preprocessing(p_body_val)
c_body_val = preprocessing(c_body_val)

# test
p_body_test = preprocessing(p_body_test)
c_body_test = preprocessing(c_body_test)

# Google Universal Sentence Encoder
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)

# train body embeddings
p_body_train_embeddings = model(p_body_train).numpy()
c_body_train_embeddings = model(c_body_train).numpy()

# validation body embeddings
p_body_val_embeddings = model(p_body_val).numpy()
c_body_val_embeddings = model(c_body_val).numpy()

# test body embeddings
p_body_test_embeddings = model(p_body_test).numpy()
c_body_test_embeddings = model(c_body_test).numpy()

# graph features dataframe
# train
post_df_train = pd.concat([pd.DataFrame(p_otype_train, columns=['otype']), pd.DataFrame(p_body_train_embeddings),
                           pd.DataFrame(p_payout_train, columns=['payout'])], axis=1, ignore_index=True)

comment_df_train = pd.concat([pd.DataFrame(c_otype_train, columns=['otype']), pd.DataFrame(c_body_train_embeddings)],
                             axis=1, ignore_index=True)

user_df_train = pd.DataFrame(u_otype_train, columns =['otype'])
dataset_train = pd.concat([post_df_train, comment_df_train], ignore_index=True, axis=0)

# validation
post_df_val = pd.concat([pd.DataFrame(p_otype_val, columns=['otype']), pd.DataFrame(p_body_val_embeddings),
                           pd.DataFrame(p_payout_val, columns=['payout'])], axis=1, ignore_index=True)

comment_df_val = pd.concat([pd.DataFrame(c_otype_val, columns=['otype']), pd.DataFrame(c_body_val_embeddings)],
                             axis=1, ignore_index=True)

user_df_val = pd.DataFrame(u_otype_val, columns =['otype'])
dataset_val = pd.concat([post_df_val, comment_df_val], ignore_index=True, axis=0)

# test
post_df_test = pd.concat([pd.DataFrame(p_otype_test, columns=['otype']), pd.DataFrame(p_body_test_embeddings),
                           pd.DataFrame(p_payout_test, columns=['payout'])], axis=1, ignore_index=True)

comment_df_test = pd.concat([pd.DataFrame(c_otype_test, columns=['otype']), pd.DataFrame(c_body_test_embeddings)],
                             axis=1, ignore_index=True)

user_df_test = pd.DataFrame(u_otype_test, columns =['otype'])
dataset_test = pd.concat([post_df_test, comment_df_test], ignore_index=True, axis=0)

# type of node dataframe
df_otype_train = pd.concat([pd.DataFrame(p_otype_train, columns=['otype']), pd.DataFrame(c_otype_train, columns=['otype']),
                     pd.DataFrame(u_otype_train, columns=['otype'])],ignore_index=True, axis=0)
df_otype_val = pd.concat([pd.DataFrame(p_otype_val, columns=['otype']), pd.DataFrame(c_otype_val, columns=['otype']),
                     pd.DataFrame(u_otype_val, columns=['otype'])],ignore_index=True, axis=0)
df_otype_test = pd.concat([pd.DataFrame(p_otype_test, columns=['otype']), pd.DataFrame(c_otype_test, columns=['otype']),
                     pd.DataFrame(u_otype_test, columns=['otype'])],ignore_index=True, axis=0)

# creating instance of one-hot-encoder
# train
enc_train = OneHotEncoder(handle_unknown='ignore')
enc_df_train = pd.DataFrame(enc_train.fit_transform(df_otype_train).toarray())
df_otype_train = df_otype_train.join(enc_df_train)

# validation
enc_val = OneHotEncoder(handle_unknown='ignore')
enc_df_val = pd.DataFrame(enc_val.fit_transform(df_otype_val).toarray())
df_otype_val = df_otype_val.join(enc_df_val)

# test
enc_test = OneHotEncoder(handle_unknown='ignore')
enc_df_test = pd.DataFrame(enc_test.fit_transform(df_otype_test).toarray())
df_otype_test = df_otype_test.join(enc_df_test)

# creating list of features fpr nodes
# train
p_otype_u_train = []
p_otype_p_train = []
p_otype_c_train = []
p_body_train = []
p_payout_train = []

u_otype_u_train = []
u_otype_p_train = []
u_otype_c_train = []

c_otype_u_train = []
c_otype_p_train = []
c_otype_c_train = []
c_body_train = []

for i in range(post_df_train.shape[0]):
    p_otype_u_train.append([df_otype_train.iloc[i, 1]])
    p_otype_p_train.append([df_otype_train.iloc[i, 2]])
    p_otype_c_train.append([df_otype_train.iloc[i, 3]])
    p_body_train.append(dataset_train.iloc[i, 1:513])
    p_payout_train.append(dataset_train.iloc[i, 513])

for i in range(len(p_body_train)):
    p_body_train[i] = p_body_train[i].tolist()

for i in range(len(dataset_train),len(df_otype_train)):
    u_otype_u_train.append([df_otype_train.iloc[i, 1]])
    u_otype_p_train.append([df_otype_train.iloc[i, 2]])
    u_otype_c_train.append([df_otype_train.iloc[i, 3]])

for i in range(post_df_train.shape[0], len(dataset_train)):
    c_otype_u_train.append([df_otype_train.iloc[i, 1]])
    c_otype_p_train.append([df_otype_train.iloc[i, 2]])
    c_otype_c_train.append([df_otype_train.iloc[i, 3]])
    c_body_train.append(dataset_train.iloc[i, 1:513])

for i in range(len(c_body_train)):
    c_body_train[i] = c_body_train[i].tolist()

# validation
p_otype_u_val = []
p_otype_p_val = []
p_otype_c_val = []
p_body_val = []
p_payout_val = []

u_otype_u_val = []
u_otype_p_val = []
u_otype_c_val = []

c_otype_u_val = []
c_otype_p_val = []
c_otype_c_val = []
c_body_val = []

for i in range(post_df_val.shape[0]):
    p_otype_u_val.append([df_otype_val.iloc[i, 1]])
    p_otype_p_val.append([df_otype_val.iloc[i, 2]])
    p_otype_c_val.append([df_otype_val.iloc[i, 3]])
    p_body_val.append(dataset_val.iloc[i, 1:513])
    p_payout_val.append(dataset_val.iloc[i, 513])

for i in range(len(p_body_val)):
    p_body_val[i] = p_body_val[i].tolist()

for i in range(len(dataset_val),len(df_otype_val)):
    u_otype_u_val.append([df_otype_val.iloc[i, 1]])
    u_otype_p_val.append([df_otype_val.iloc[i, 2]])
    u_otype_c_val.append([df_otype_val.iloc[i, 3]])

for i in range(post_df_val.shape[0], len(dataset_val)):
    c_otype_u_val.append([df_otype_val.iloc[i, 1]])
    c_otype_p_val.append([df_otype_val.iloc[i, 2]])
    c_otype_c_val.append([df_otype_val.iloc[i, 3]])
    c_body_val.append(dataset_val.iloc[i, 1:513])

for i in range(len(c_body_val)):
    c_body_val[i] = c_body_val[i].tolist()

# test
p_otype_u_test = []
p_otype_p_test = []
p_otype_c_test = []
p_body_test = []
p_payout_test = []

u_otype_u_test = []
u_otype_p_test = []
u_otype_c_test = []

c_otype_u_test = []
c_otype_p_test = []
c_otype_c_test = []
c_body_test = []

for i in range(post_df_test.shape[0]):
    p_otype_u_test.append([df_otype_test.iloc[i, 1]])
    p_otype_p_test.append([df_otype_test.iloc[i, 2]])
    p_otype_c_test.append([df_otype_test.iloc[i, 3]])
    p_body_test.append(dataset_test.iloc[i, 1:513])
    p_payout_test.append(dataset_test.iloc[i, 513])

for i in range(len(p_body_test)):
    p_body_test[i] = p_body_test[i].tolist()

for i in range(len(dataset_test),len(df_otype_test)):
    u_otype_u_test.append([df_otype_test.iloc[i, 1]])
    u_otype_p_test.append([df_otype_test.iloc[i, 2]])
    u_otype_c_test.append([df_otype_test.iloc[i, 3]])

for i in range(post_df_test.shape[0], len(dataset_test)):
    c_otype_u_test.append([df_otype_test.iloc[i, 1]])
    c_otype_p_test.append([df_otype_test.iloc[i, 2]])
    c_otype_c_test.append([df_otype_test.iloc[i, 3]])
    c_body_test.append(dataset_test.iloc[i, 1:513])

for i in range(len(c_body_test)):
    c_body_test[i] = c_body_test[i].tolist()

# target value (payouts)
# train
for i in range(len(p_payout_train)):
    if p_payout_train[i] < 5:
        p_payout_train[i] = 0
    elif p_payout_train[i] >= 5 and p_payout_train[i] < 40:
        p_payout_train[i] = 1
    else:
        p_payout_train[i] = 2
p_payout_train = torch.tensor(p_payout_train).long()

# validation
for i in range(len(p_payout_val)):
    if p_payout_val[i] < 5:
        p_payout_val[i] = 0
    elif p_payout_val[i] >= 5 and p_payout_val[i] < 40:
        p_payout_val[i] = 1
    else:
        p_payout_val[i] = 2
p_payout_val = torch.tensor(p_payout_val).long()

# test
for i in range(len(p_payout_test)):
    if p_payout_test[i] < 5:
        p_payout_test[i] = 0
    elif p_payout_test[i] >= 5 and p_payout_test[i] < 40:
        p_payout_test[i] = 1
    else:
        p_payout_test[i] = 2
p_payout_test = torch.tensor(p_payout_test).long()

# convert type of lists to tensor
p_otype_u_train = torch.tensor(p_otype_u_train).float()
p_otype_p_train = torch.tensor(p_otype_p_train).float()
p_otype_c_train = torch.tensor(p_otype_c_train).float()
p_body_train = torch.tensor(p_body_train).float()
p_payout_train = torch.tensor(p_payout_train).float()

p_otype_u_val = torch.tensor(p_otype_u_val).float()
p_otype_p_val = torch.tensor(p_otype_p_val).float()
p_otype_c_val = torch.tensor(p_otype_c_val).float()
p_body_val = torch.tensor(p_body_val).float()
p_payout_val = torch.tensor(p_payout_val).float()

p_otype_u_test = torch.tensor(p_otype_u_test).float()
p_otype_p_test = torch.tensor(p_otype_p_test).float()
p_otype_c_test = torch.tensor(p_otype_c_test).float()
p_body_test = torch.tensor(p_body_test).float()
p_payout_test = torch.tensor(p_payout_test).float()

u_otype_u_train = torch.tensor(u_otype_u_train).float()
u_otype_p_train = torch.tensor(u_otype_p_train).float()
u_otype_c_train = torch.tensor(u_otype_c_train).float()
u_otype_u_val = torch.tensor(u_otype_u_val).float()
u_otype_p_val = torch.tensor(u_otype_p_val).float()
u_otype_c_val = torch.tensor(u_otype_c_val).float()
u_otype_u_test = torch.tensor(u_otype_u_test).float()
u_otype_p_test = torch.tensor(u_otype_p_test).float()
u_otype_c_test = torch.tensor(u_otype_c_test).float()

c_otype_u_train = torch.tensor(c_otype_u_train).float()
c_otype_p_train = torch.tensor(c_otype_p_train).float()
c_otype_c_train = torch.tensor(c_otype_c_train).float()
c_body_train = torch.tensor(c_body_train).float()

c_otype_u_val = torch.tensor(c_otype_u_val).float()
c_otype_p_val = torch.tensor(c_otype_p_val).float()
c_otype_c_val = torch.tensor(c_otype_c_val).float()
c_body_val = torch.tensor(c_body_val).float()

c_otype_u_test = torch.tensor(c_otype_u_test).float()
c_otype_p_test = torch.tensor(c_otype_p_test).float()
c_otype_c_test = torch.tensor(c_otype_c_test).float()
c_body_test = torch.tensor(c_body_test).float()


# create lists of edges
# train
vote_p_i_train = []
vote_p_j_train = []
vote_c_i_train = []
vote_c_j_train = []
authored_p_i_train = []
authored_p_j_train = []
authored_c_i_train = []
authored_c_j_train = []
reply_i_train = []
reply_j_train = []
v_p_otype_train = []
a_p_otype_train = []
v_c_otype_train = []
a_c_otype_train = []
r_otype_train = []
v_p_time_train = []
a_p_time_train = []
v_c_time_train = []
a_c_time_train = []
r_time_train = []

post_train, comment_train, user_train = {}, {}, {}
post_idx_train, comment_idx_train, user_idx_train = 0, 0, 0
for i in range(len(G_train.nodes())):
    if G_train.nodes()[i]['otype'] == 'user':
        user_train[i] = user_idx_train
        user_idx_train += 1
    elif G_train.nodes()[i]['otype'] == 'post':
        post_train[i] = post_idx_train
        post_idx_train += 1
    else:
        comment_train[i] = comment_idx_train
        comment_idx_train += 1

for i in range(len(G_train.nodes())):
    if G_train.nodes()[i]['otype'] == 'user':
        for j in range(len(G_train.nodes())):
            if (i,j) in G_train.edges():
                if G_train.edges[i,j]['otype'] == 'vote':
                    if G_train.nodes()[j]['otype'] == 'post':
                        v_p_otype_train.append([torch.tensor(1)])
                        v_p_time_train.append([int(G_train.edges[i,j]['time'])])
                        vote_p_i_train.append(user_train[i])
                        vote_p_j_train.append(post_train[j])
                    elif G_train.nodes()[j]['otype'] == 'comment':
                        v_c_otype_train.append([torch.tensor(1)])
                        v_c_time_train.append([int(G_train.edges[i,j]['time'])])
                        vote_c_i_train.append(user_train[i])
                        vote_c_j_train.append(comment_train[j])

                elif G_train.edges[i,j]['otype'] == 'authored':
                    if G_train.nodes()[j]['otype'] == 'post':
                        a_p_otype_train.append([torch.tensor(0)])
                        a_p_time_train.append([int(G_train.edges[i,j]['time'])])
                        authored_p_i_train.append(user_train[i])
                        authored_p_j_train.append(post_train[j])
                    elif G_train.nodes()[j]['otype'] == 'comment':
                        a_c_otype_train.append([torch.tensor(0)])
                        a_c_time_train.append([int(G_train.edges[i,j]['time'])])
                        authored_c_i_train.append(user_train[i])
                        authored_c_j_train.append(comment_train[j])

    elif G_train.nodes()[i]['otype'] == 'post':
        for j in range(len(G_train.nodes())):
            if (i,j) in G_train.edges():
                if G_train.edges[i,j]['otype'] == 'reply':
                    reply_i_train.append(post_train[i])
                    reply_j_train.append(comment_train[j])
                    r_otype_train.append([torch.tensor(2)])
                    r_time_train.append([int(G_train.edges[i,j]['time'])])

# validation
vote_p_i_val = []
vote_p_j_val = []
vote_c_i_val = []
vote_c_j_val = []
authored_p_i_val = []
authored_p_j_val = []
authored_c_i_val = []
authored_c_j_val = []
reply_i_val = []
reply_j_val = []
v_p_otype_val = []
a_p_otype_val = []
v_c_otype_val = []
a_c_otype_val = []
r_otype_val = []
v_p_time_val = []
a_p_time_val = []
v_c_time_val = []
a_c_time_val = []
r_time_val = []

post_val, comment_val, user_val = {}, {}, {}
post_idx_val, comment_idx_val, user_idx_val = 0, 0, 0
for i in range(len(G_val.nodes())):
    if G_val.nodes()[i]['otype'] == 'user':
        user_val[i] = user_idx_val
        user_idx_val += 1
    elif G_val.nodes()[i]['otype'] == 'post':
        post_val[i] = post_idx_val
        post_idx_val += 1
    else:
        comment_val[i] = comment_idx_val
        comment_idx_val += 1

for i in range(len(G_val.nodes())):
    if G_val.nodes()[i]['otype'] == 'user':
        for j in range(len(G_val.nodes())):
            if (i,j) in G_val.edges():
                if G_val.edges[i,j]['otype'] == 'vote':
                    if G_val.nodes()[j]['otype'] == 'post':
                        v_p_otype_val.append([torch.tensor(1)])
                        v_p_time_val.append([int(G_val.edges[i,j]['time'])])
                        vote_p_i_val.append(user_val[i])
                        vote_p_j_val.append(post_val[j])
                    elif G_val.nodes()[j]['otype'] == 'comment':
                        v_c_otype_val.append([torch.tensor(1)])
                        v_c_time_val.append([int(G_val.edges[i,j]['time'])])
                        vote_c_i_val.append(user_val[i])
                        vote_c_j_val.append(comment_val[j])

                elif G_val.edges[i,j]['otype'] == 'authored':
                    if G_val.nodes()[j]['otype'] == 'post':
                        a_p_otype_val.append([torch.tensor(0)])
                        a_p_time_val.append([int(G_val.edges[i,j]['time'])])
                        authored_p_i_val.append(user_val[i])
                        authored_p_j_val.append(post_val[j])
                    elif G_val.nodes()[j]['otype'] == 'comment':
                        a_c_otype_val.append([torch.tensor(0)])
                        a_c_time_val.append([int(G_val.edges[i,j]['time'])])
                        authored_c_i_val.append(user_val[i])
                        authored_c_j_val.append(comment_val[j])

    elif G_val.nodes()[i]['otype'] == 'post':
        for j in range(len(G_val.nodes())):
            if (i,j) in G_val.edges():
                if G_val.edges[i,j]['otype'] == 'reply':
                    reply_i_val.append(post_val[i])
                    reply_j_val.append(comment_val[j])
                    r_otype_val.append([torch.tensor(2)])
                    r_time_val.append([int(G_val.edges[i,j]['time'])])

# test
vote_p_i_test = []
vote_p_j_test = []
vote_c_i_test = []
vote_c_j_test = []
authored_p_i_test = []
authored_p_j_test = []
authored_c_i_test = []
authored_c_j_test = []
reply_i_test = []
reply_j_test = []
v_p_otype_test = []
a_p_otype_test = []
v_c_otype_test = []
a_c_otype_test = []
r_otype_test = []
v_p_time_test = []
a_p_time_test = []
v_c_time_test = []
a_c_time_test = []
r_time_test = []

post_test, comment_test, user_test = {}, {}, {}
post_idx_test, comment_idx_test, user_idx_test = 0, 0, 0
for i in range(len(G_test.nodes())):
    if G_test.nodes()[i]['otype'] == 'user':
        user_test[i] = user_idx_test
        user_idx_test += 1
    elif G_test.nodes()[i]['otype'] == 'post':
        post_test[i] = post_idx_test
        post_idx_test += 1
    else:
        comment_test[i] = comment_idx_test
        comment_idx_test += 1

for i in range(len(G_test.nodes())):
    if G_test.nodes()[i]['otype'] == 'user':
        for j in range(len(G_test.nodes())):
            if (i,j) in G_test.edges():
                if G_test.edges[i,j]['otype'] == 'vote':
                    if G_test.nodes()[j]['otype'] == 'post':
                        v_p_otype_test.append([torch.tensor(1)])
                        v_p_time_test.append([int(G_test.edges[i,j]['time'])])
                        vote_p_i_test.append(user_test[i])
                        vote_p_j_test.append(post_test[j])
                    elif G_test.nodes()[j]['otype'] == 'comment':
                        v_c_otype_test.append([torch.tensor(1)])
                        v_c_time_test.append([int(G_test.edges[i,j]['time'])])
                        vote_c_i_test.append(user_test[i])
                        vote_c_j_test.append(comment_test[j])

                elif G_test.edges[i,j]['otype'] == 'authored':
                    if G_test.nodes()[j]['otype'] == 'post':
                        a_p_otype_test.append([torch.tensor(0)])
                        a_p_time_test.append([int(G_test.edges[i,j]['time'])])
                        authored_p_i_test.append(user_test[i])
                        authored_p_j_test.append(post_test[j])
                    elif G_test.nodes()[j]['otype'] == 'comment':
                        a_c_otype_test.append([torch.tensor(0)])
                        a_c_time_test.append([int(G_test.edges[i,j]['time'])])
                        authored_c_i_test.append(user_test[i])
                        authored_c_j_test.append(comment_test[j])

    elif G_test.nodes()[i]['otype'] == 'post':
        for j in range(len(G_test.nodes())):
            if (i,j) in G_test.edges():
                if G_test.edges[i,j]['otype'] == 'reply':
                    reply_i_test.append(post_test[i])
                    reply_j_test.append(comment_test[j])
                    r_otype_test.append([torch.tensor(2)])
                    r_time_test.append([int(G_test.edges[i,j]['time'])])

# convert type of lists to tensor
# train
vote_p_i_train = torch.tensor(vote_p_i_train)
vote_p_j_train = torch.tensor(vote_p_j_train)
vote_c_i_train = torch.tensor(vote_c_i_train)
vote_c_j_train = torch.tensor(vote_c_j_train)
authored_p_i_train = torch.tensor(authored_p_i_train)
authored_p_j_train = torch.tensor(authored_p_j_train)
authored_c_i_train = torch.tensor(authored_c_i_train)
authored_c_j_train = torch.tensor(authored_c_j_train)
reply_i_train = torch.tensor(reply_i_train)
reply_j_train = torch.tensor(reply_j_train)
v_p_otype_train = torch.tensor(v_p_otype_train)
a_p_otype_train = torch.tensor(a_p_otype_train)
v_c_otype_train = torch.tensor(v_c_otype_train)
a_c_otype_train = torch.tensor(a_c_otype_train)
r_otype_train = torch.tensor(r_otype_train)
v_p_time_train = torch.tensor(v_p_time_train)
a_p_time_train = torch.tensor(a_p_time_train)
v_c_time_train = torch.tensor(v_c_time_train)
a_c_time_train = torch.tensor(a_c_time_train)
r_time_train = torch.tensor(r_time_train)

# validation
vote_p_i_val = torch.tensor(vote_p_i_val)
vote_p_j_val = torch.tensor(vote_p_j_val)
vote_c_i_val = torch.tensor(vote_c_i_val)
vote_c_j_val = torch.tensor(vote_c_j_val)
authored_p_i_val = torch.tensor(authored_p_i_val)
authored_p_j_val = torch.tensor(authored_p_j_val)
authored_c_i_val = torch.tensor(authored_c_i_val)
authored_c_j_val = torch.tensor(authored_c_j_val)
reply_i_val = torch.tensor(reply_i_val)
reply_j_val = torch.tensor(reply_j_val)
v_p_otype_val = torch.tensor(v_p_otype_val)
a_p_otype_val = torch.tensor(a_p_otype_val)
v_c_otype_val = torch.tensor(v_c_otype_val)
a_c_otype_val = torch.tensor(a_c_otype_val)
r_otype_val = torch.tensor(r_otype_val)
v_p_time_val = torch.tensor(v_p_time_val)
a_p_time_val = torch.tensor(a_p_time_val)
v_c_time_val = torch.tensor(v_c_time_val)
a_c_time_val = torch.tensor(a_c_time_val)
r_time_val = torch.tensor(r_time_val)

# test
vote_p_i_test = torch.tensor(vote_p_i_test)
vote_p_j_test = torch.tensor(vote_p_j_test)
vote_c_i_test = torch.tensor(vote_c_i_test)
vote_c_j_test = torch.tensor(vote_c_j_test)
authored_p_i_test = torch.tensor(authored_p_i_test)
authored_p_j_test = torch.tensor(authored_p_j_test)
authored_c_i_test = torch.tensor(authored_c_i_test)
authored_c_j_test = torch.tensor(authored_c_j_test)
reply_i_test = torch.tensor(reply_i_test)
reply_j_test = torch.tensor(reply_j_test)
v_p_otype_test = torch.tensor(v_p_otype_test)
a_p_otype_test = torch.tensor(a_p_otype_test)
v_c_otype_test = torch.tensor(v_c_otype_test)
a_c_otype_test = torch.tensor(a_c_otype_test)
r_otype_test = torch.tensor(r_otype_test)
v_p_time_test = torch.tensor(v_p_time_test)
a_p_time_test = torch.tensor(a_p_time_test)
v_c_time_test = torch.tensor(v_c_time_test)
a_c_time_test = torch.tensor(a_c_time_test)
r_time_test = torch.tensor(r_time_test)

# create edges for heterograph
# train
vote_p_train = torch.cat((vote_p_i_train,vote_p_j_train)).reshape(-1,len(vote_p_i_train)).long()
vote_c_train = torch.cat((vote_c_i_train,vote_c_j_train)).reshape(-1,len(vote_c_i_train)).long()
authored_p_train = torch.cat((authored_p_i_train,authored_p_j_train)).reshape(-1,len(authored_p_i_train)).long()
authored_c_train = torch.cat((authored_c_i_train,authored_c_j_train)).reshape(-1,len(authored_c_i_train)).long()
reply_train = torch.cat((reply_i_train,reply_j_train)).reshape(-1,len(reply_i_train)).long()

# validation
vote_p_val = torch.cat((vote_p_i_val,vote_p_j_val)).reshape(-1,len(vote_p_i_val)).long()
vote_c_val = torch.cat((vote_c_i_val,vote_c_j_val)).reshape(-1,len(vote_c_i_val)).long()
authored_p_val = torch.cat((authored_p_i_val,authored_p_j_val)).reshape(-1,len(authored_p_i_val)).long()
authored_c_val = torch.cat((authored_c_i_val,authored_c_j_val)).reshape(-1,len(authored_c_i_val)).long()
reply_val = torch.cat((reply_i_val,reply_j_val)).reshape(-1,len(reply_i_val)).long()

# test
vote_p_test = torch.cat((vote_p_i_test,vote_p_j_test)).reshape(-1,len(vote_p_i_test)).long()
vote_c_test = torch.cat((vote_c_i_test,vote_c_j_test)).reshape(-1,len(vote_c_i_test)).long()
authored_p_test = torch.cat((authored_p_i_test,authored_p_j_test)).reshape(-1,len(authored_p_i_test)).long()
authored_c_test = torch.cat((authored_c_i_test,authored_c_j_test)).reshape(-1,len(authored_c_i_test)).long()
reply_test = torch.cat((reply_i_test,reply_j_test)).reshape(-1,len(reply_i_test)).long()

# Creatte heterograph in PyTorch
data_train = HeteroData()
data_val = HeteroData()
data_test = HeteroData()

# adding node features to heterograph
# train
# Post features
data_train['post'].x = torch.cat([p_otype_u_train, p_otype_p_train, p_otype_c_train, p_body_train], dim=-1)
data_train['post'].y = p_payout_train

# User features
data_train['user'].x = torch.cat([u_otype_u_train, u_otype_p_train, u_otype_c_train], dim=-1)

# Comment features
data_train['comment'].x = torch.cat([c_otype_u_train, c_otype_p_train, c_otype_c_train, c_body_train], dim=-1)

# validation
# Post features
data_val['post'].x = torch.cat([p_otype_u_val, p_otype_p_val, p_otype_c_val, p_body_val], dim=-1)
data_val['post'].y = p_payout_val

# User features
data_val['user'].x = torch.cat([u_otype_u_val, u_otype_p_val, u_otype_c_val], dim=-1)

# Comment features
data_val['comment'].x = torch.cat([c_otype_u_val, c_otype_p_val, c_otype_c_val, c_body_val], dim=-1)

# test
# Post features
data_test['post'].x = torch.cat([p_otype_u_test, p_otype_p_test, p_otype_c_test, p_body_test], dim=-1)
data_test['post'].y = p_payout_test

# User features
data_test['user'].x = torch.cat([u_otype_u_test, u_otype_p_test, u_otype_c_test], dim=-1)

# Comment features
data_test['comment'].x = torch.cat([c_otype_u_test, c_otype_p_test, c_otype_c_test, c_body_test], dim=-1)

# adding edges to heterograph
# train
data_train['user', 'authored_post', 'post'].edge_index = authored_p_train
data_train['user', 'authored_comment', 'comment'].edge_index = authored_c_train
data_train['user', 'vote_post', 'post'].edge_index = vote_p_train
data_train['user', 'vote_comment', 'comment'].edge_index = vote_c_train
data_train['post', 'reply', 'comment'].edge_index = reply_train

data_train['user', 'authored_post', 'post'].edge_attr = torch.cat([a_p_otype_train, a_p_time_train], dim=-1)
data_train['user', 'authored_comment', 'comment'].edge_attr = torch.cat([a_c_otype_train, a_c_time_train], dim=-1)
data_train['user', 'vote_post', 'post'].edge_attr = torch.cat([v_p_otype_train, v_p_time_train], dim=-1)
data_train['user', 'vote_comment', 'comment'].edge_attr = torch.cat([v_c_otype_train, v_c_time_train], dim=-1)
data_train['post', 'reply', 'comment'].edge_attr = torch.cat([r_otype_train, r_time_train], dim=-1)

# validation
data_val['user', 'authored_post', 'post'].edge_index = authored_p_val
data_val['user', 'authored_comment', 'comment'].edge_index = authored_c_val
data_val['user', 'vote_post', 'post'].edge_index = vote_p_val
data_val['user', 'vote_comment', 'comment'].edge_index = vote_c_val
data_val['post', 'reply', 'comment'].edge_index = reply_val

data_val['user', 'authored_post', 'post'].edge_attr = torch.cat([a_p_otype_val, a_p_time_val], dim=-1)
data_val['user', 'authored_comment', 'comment'].edge_attr = torch.cat([a_c_otype_val, a_c_time_val], dim=-1)
data_val['user', 'vote_post', 'post'].edge_attr = torch.cat([v_p_otype_val, v_p_time_val], dim=-1)
data_val['user', 'vote_comment', 'comment'].edge_attr = torch.cat([v_c_otype_val, v_c_time_val], dim=-1)
data_val['post', 'reply', 'comment'].edge_attr = torch.cat([r_otype_val, r_time_val], dim=-1)

# test
data_test['user', 'authored_post', 'post'].edge_index = authored_p_test
data_test['user', 'authored_comment', 'comment'].edge_index = authored_c_test
data_test['user', 'vote_post', 'post'].edge_index = vote_p_test
data_test['user', 'vote_comment', 'comment'].edge_index = vote_c_test
data_test['post', 'reply', 'comment'].edge_index = reply_test

data_test['user', 'authored_post', 'post'].edge_attr = torch.cat([a_p_otype_test, a_p_time_test], dim=-1)
data_test['user', 'authored_comment', 'comment'].edge_attr = torch.cat([a_c_otype_test, a_c_time_test], dim=-1)
data_test['user', 'vote_post', 'post'].edge_attr = torch.cat([v_p_otype_test, v_p_time_test], dim=-1)
data_test['user', 'vote_comment', 'comment'].edge_attr = torch.cat([v_c_otype_test, v_c_time_test], dim=-1)
data_test['post', 'reply', 'comment'].edge_attr = torch.cat([r_otype_test, r_time_test], dim=-1)

# Transform the edeges
data_train = T.ToUndirected()(data_train)
data_val = T.ToUndirected()(data_val)
data_test = T.ToUndirected()(data_test)

# Training the model
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

def accuracy(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()

model = GNN(hidden_channels=512, out_channels=3)
model = to_hetero(model, data_train.metadata(), aggr='sum')

# Loss and optimizer
num_epochs = 3000
learning_rate = 0.001
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # Forward pass and loss
    y_pred = model(data_train.x_dict, data_train.edge_index_dict)
    loss = criterion(y_pred['post'], data_train['post'].y.type(torch.LongTensor))

    # Backward pass and update
    loss.backward()
    optimizer.step()

    # zero grad before new step
    optimizer.zero_grad()
    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
        acc = accuracy(y_pred['post'].argmax(dim=1), data_train['post'].y.type(torch.LongTensor))
        print(f'accuracy train: {acc*100:.4f}')


with torch.no_grad():
    model.eval()
    y_predicted_val = model(data_val.x_dict, data_val.edge_index_dict)
    acc_val = accuracy(y_predicted_val['post'].argmax(dim=1), data_val['post'].y.type(torch.LongTensor))
    print(f'accuracy validation: {acc_val*100:.4f}')

with torch.no_grad():
    model.eval()
    y_predicted_test = model(data_test.x_dict, data_test.edge_index_dict)
    acc_test = accuracy(y_predicted_test['post'].argmax(dim=1), data_test['post'].y.type(torch.LongTensor))
    print(f'accuracy test: {acc_test*100:.4f}')

# confusion matrix
confmat = ConfusionMatrix(num_classes=3)
cm = confmat(y_predicted_test['post'].argmax(dim=1), data_test['post'].y.type(torch.LongTensor))
print("Confusion Matrix: ", cm)

# classification report
pd.DataFrame(y_predicted_test['post'].argmax(dim=1)).to_csv("~/model2_3class_testpred.csv")
pd.DataFrame(data_test['post'].y.type(torch.LongTensor)).to_csv("~/model2_3class_testactual.csv")
model2_3C_pred = pd.read_csv("~/model2_3class_testpred.csv")
model2_3C_act = pd.read_csv("~/model2_3class_testactual.csv")
cr = classification_report(model2_3C_act.iloc[:,0], model2_3C_pred.iloc[:,0])
print("Classification Report: ", cr)
