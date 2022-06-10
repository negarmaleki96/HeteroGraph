# import libraries
import json
import ast
import math
import time
import numpy as np
import pandas as pd
import networkx as nx
from steem import Steem
from datetime import datetime

# import dataset
aug_sep_2019 = pd.read_csv('~/data_AugSep2019.csv')
aug_sep_2019.drop('Unnamed: 0', axis = 1, inplace=True)
aug_sep_2019.drop_duplicates('permlink', ignore_index=True, inplace=True)
aug_sep_2019.dropna(subset=['permlink'], axis = 0, inplace=True)

# create a Graph in networkX
G = nx.DiGraph()
for i in range(len(aug_sep_2019)):
    # create user nodes
    # create unique user node (if it isn't already created)
    if aug_sep_2019.iloc[i,1] not in list(G.nodes):
        G.add_node(aug_sep_2019.iloc[i,1])
        # adding attributes to the created user node
        G.nodes[aug_sep_2019.iloc[i,1]]['username'] = aug_sep_2019.iloc[i,1]
        G.nodes[aug_sep_2019.iloc[i,1]]['otype'] = 'user'
    # create post nodes
    G.add_node(aug_sep_2019.iloc[i,2])
    # adding attributes to the created post node
    G.nodes[aug_sep_2019.iloc[i,2]]['permlink'] = aug_sep_2019.iloc[i,2]
    G.nodes[aug_sep_2019.iloc[i,2]]['otype'] = 'post'
    G.nodes[aug_sep_2019.iloc[i,2]]['category'] = aug_sep_2019.iloc[i,3]
    G.nodes[aug_sep_2019.iloc[i,2]]['title'] = [aug_sep_2019.iloc[i,6]]
    G.nodes[aug_sep_2019.iloc[i,2]]['body'] = [aug_sep_2019.iloc[i,7]]
    # some posts do not have 'tags'
    try:
        G.nodes[aug_sep_2019.iloc[i,2]]['tag'] = json.loads(aug_sep_2019.iloc[i,8])['tags']
    except:
        pass
    # all datetime attributes converted to the timestamp
    G.nodes[aug_sep_2019.iloc[i,2]]['last_update'] = [int(datetime.timestamp(datetime.fromisoformat(aug_sep_2019.iloc[i,9])))]
    G.nodes[aug_sep_2019.iloc[i,2]]['created'] = int(aug_sep_2019.iloc[i,46])
    G.nodes[aug_sep_2019.iloc[i,2]]['last_payout'] = int(datetime.timestamp(datetime.fromisoformat(aug_sep_2019.iloc[i,12])))
    G.nodes[aug_sep_2019.iloc[i,2]]['payout'] = float(aug_sep_2019.iloc[i,23].split()[0])+float(aug_sep_2019.iloc[i,24].split()[0])
    G.nodes[aug_sep_2019.iloc[i,2]]['net_rshares'] = int(aug_sep_2019.iloc[i,15])
    G.nodes[aug_sep_2019.iloc[i,2]]['abs_rshares'] = int(aug_sep_2019.iloc[i,16])
    G.nodes[aug_sep_2019.iloc[i,2]]['vote_rshares'] = int(aug_sep_2019.iloc[i,17])
    G.nodes[aug_sep_2019.iloc[i,2]]['author_rewards'] = int(aug_sep_2019.iloc[i,25])
    G.nodes[aug_sep_2019.iloc[i,2]]['url'] = 'https://steemit.com'+aug_sep_2019.iloc[i,35]
    # author reputation calculated based on the formula in "https://steemit.com/basictraining/@vaansteam/how-does-steemit-reputation-work-ultimate-guide"
    try:
        G.nodes[aug_sep_2019.iloc[i,2]]['author_reputation'] = math.floor((math.log10(int(aug_sep_2019.iloc[i,41]))-9)*9+25)
    except:
        pass
    # create edge between user and their posts (time attribute is in form of timestamp)
    G.add_edge(aug_sep_2019.iloc[i,1], aug_sep_2019.iloc[i,2], time=int(aug_sep_2019.iloc[i,46]), otype='authored')
    # create node and edge for vote
    if len(aug_sep_2019.iloc[i,39]) != 0:
        for j in range(len(ast.literal_eval(aug_sep_2019.iloc[i,39]))):
            try:
                # create vote nodes by searching in active_votes column json
                if ast.literal_eval(aug_sep_2019.iloc[i,39])[j]['voter'] not in list(G.nodes):
                    G.add_node(ast.literal_eval(aug_sep_2019.iloc[i,39])[j]['voter'])
                    G.nodes[ast.literal_eval(aug_sep_2019.iloc[i,39])[j]['voter']]['username'] = ast.literal_eval(aug_sep_2019.iloc[i,39])[j]['voter']
                    G.nodes[ast.literal_eval(aug_sep_2019.iloc[i,39])[j]['voter']]['otype'] = 'user'
                # create edge between voters and posts (time attribute is in form of timestamp)
                G.add_edge(ast.literal_eval(aug_sep_2019.iloc[i,39])[j]['voter'], aug_sep_2019.iloc[i,2], time=int(datetime.timestamp(datetime.fromisoformat(ast.literal_eval(aug_sep_2019.iloc[i,39])[j]['time']))), otype='vote')
            except:
                pass

# import comment dataset
replies_aug_sep_2019 = pd.read_csv('~/replies_AugSep2019.csv')
replies_aug_sep_2019.drop('Unnamed: 0', axis = 1, inplace=True)

# Adding other nodes to the graph
for i in range(len(replies_aug_sep_2019)):
    for j in range(replies_aug_sep_2019.shape[1]):
        if type(replies_aug_sep_2019.iloc[i,j]) == dict:
            # create user nodes
            # create unique user node (if it isn't already created)
            if replies_aug_sep_2019.iloc[i,j]['author'] not in list(G.nodes):
                G.add_node(replies_aug_sep_2019.iloc[i,j]['author'])
                # adding attributes to the created user node
                G.nodes[replies_aug_sep_2019.iloc[i,j]['author']]['username'] = replies_aug_sep_2019.iloc[i,j]['author']
                G.nodes[replies_aug_sep_2019.iloc[i,j]['author']]['otype'] = 'user'
            # create comment nodes
            G.add_node(replies_aug_sep_2019.iloc[i,j]['permlink'])
            # adding attributes to the created post node
            G.nodes[replies_aug_sep_2019.iloc[i,j]['permlink']]['permlink'] = replies_aug_sep_2019.iloc[i,j]['permlink']
            G.nodes[replies_aug_sep_2019.iloc[i,j]['permlink']]['otype'] = 'comment'
            G.nodes[replies_aug_sep_2019.iloc[i,j]['permlink']]['category'] = replies_aug_sep_2019.iloc[i,j]['category']
            G.nodes[replies_aug_sep_2019.iloc[i,j]['permlink']]['title'] = [replies_aug_sep_2019.iloc[i,j]['title']]
            G.nodes[replies_aug_sep_2019.iloc[i,j]['permlink']]['body'] = [replies_aug_sep_2019.iloc[i,j]['body']]
            G.nodes[replies_aug_sep_2019.iloc[i,j]['permlink']]['parent_permlink'] = replies_aug_sep_2019.iloc[i,j]['parent_permlink']
            # some posts do not have 'tags'
            try:
                G.nodes[replies_aug_sep_2019.iloc[i,j]['permlink']]['tag'] = json.loads(replies_aug_sep_2019.iloc[i,j]['json_metadata'])['tags']
            except:
                pass
            # all datetime attributes converted to the timestamp
            G.nodes[replies_aug_sep_2019.iloc[i,j]['permlink']]['last_update'] = [int(datetime.timestamp(datetime.fromisoformat(replies_aug_sep_2019.iloc[i,j]['last_update'])))]
            G.nodes[replies_aug_sep_2019.iloc[i,j]['permlink']]['created'] = int(datetime.timestamp(datetime.fromisoformat(replies_aug_sep_2019.iloc[i,j]['created'])))
            G.nodes[replies_aug_sep_2019.iloc[i,j]['permlink']]['last_payout'] = int(datetime.timestamp(datetime.fromisoformat(replies_aug_sep_2019.iloc[i,j]['last_payout'])))
            G.nodes[replies_aug_sep_2019.iloc[i,j]['permlink']]['payout'] = float((replies_aug_sep_2019.iloc[i,j]['total_payout_value']).split()[0])+float((replies_aug_sep_2019.iloc[i,j]['curator_payout_value']).split()[0])
            G.nodes[replies_aug_sep_2019.iloc[i,j]['permlink']]['net_rshares'] = int(replies_aug_sep_2019.iloc[i,j]['net_rshares'])
            G.nodes[replies_aug_sep_2019.iloc[i,j]['permlink']]['abs_rshares'] = int(replies_aug_sep_2019.iloc[i,j]['abs_rshares'])
            G.nodes[replies_aug_sep_2019.iloc[i,j]['permlink']]['vote_rshares'] = int(replies_aug_sep_2019.iloc[i,j]['vote_rshares'])
            G.nodes[replies_aug_sep_2019.iloc[i,j]['permlink']]['author_rewards'] = int(replies_aug_sep_2019.iloc[i,j]['author_rewards'])
            G.nodes[replies_aug_sep_2019.iloc[i,j]['permlink']]['url'] = 'https://steemit.com'+replies_aug_sep_2019.iloc[i,j]['url']
            # author reputation calculated based on the formula in "https://steemit.com/basictraining/@vaansteam/how-does-steemit-reputation-work-ultimate-guide"
            try:
                G.nodes[replies_aug_sep_2019.iloc[i,j]['permlink']]['author_reputation'] = math.floor((math.log10(int(replies_aug_sep_2019.iloc[i,j]['author_reputation']))-9)*9+25)
            except:
                pass
            # create edge between user and their comments (time attribute is in form of timestamp)
            G.add_edge(replies_aug_sep_2019.iloc[i,j]['author'], replies_aug_sep_2019.iloc[i,j]['permlink'], time=int(datetime.timestamp(datetime.fromisoformat(replies_aug_sep_2019.iloc[i,j]['created']))), otype='authored')
            # create node and edge for vote
            if len(replies_aug_sep_2019.iloc[i,j]['active_votes']) != 0:
                for k in range(len(ast.literal_eval(replies_aug_sep_2019.iloc[i,j]['active_votes']))):
                    try:
                        # create vote nodes by searching in active_votes column json
                        if ast.literal_eval(replies_aug_sep_2019.iloc[i,j]['active_votes'])[k]['voter'] not in list(G.nodes):
                            G.add_node(ast.literal_eval(replies_aug_sep_2019.iloc[i,j]['active_votes'])[k]['voter'])
                            G.nodes[ast.literal_eval(replies_aug_sep_2019.iloc[i,j]['active_votes'])[k]['voter']]['username'] = ast.literal_eval(replies_aug_sep_2019.iloc[i,j]['active_votes'])[k]['voter']
                            G.nodes[ast.literal_eval(replies_aug_sep_2019.iloc[i,j]['active_votes'])[k]['voter']]['otype'] = 'user'
                        # create edge between voters and posts (time attribute is in form of timestamp)
                        G.add_edge(ast.literal_eval(replies_aug_sep_2019.iloc[i,j]['active_votes'])[k]['voter'], replies_aug_sep_2019.iloc[i,j]['permlink'], time=int(datetime.timestamp(datetime.fromisoformat(ast.literal_eval(replies_aug_sep_2019.iloc[i,j]['active_votes'])[j]['time']))), otype='vote')
                    except:
                        pass
            # create edge between post and comment
            if aug_sep_2019.iloc[i,2] == replies_aug_sep_2019.iloc[i,j]['parent_permlink']:
                G.add_edge(aug_sep_2019.iloc[i,2], replies_aug_sep_2019.iloc[i,j]['permlink'], time=int(datetime.timestamp(datetime.fromisoformat(replies_aug_sep_2019.iloc[i,j]['created']))), otype='reply')

# save the graph in pickle format
nx.write_gpickle(G, "~/AugSep2019_all_org.gpickle")

# relable the graph
keys = np.array(list(G.nodes))
values = [int(i) for i in np.arange(0, len(G.nodes))]
dic = dict(zip(keys, values))
H = nx.relabel_nodes(G, dic)

# save the relabeled graph in pickle format
nx.write_gpickle(H, "~/AugSep2019_all.gpickle")
