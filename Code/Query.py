#!/usr/bin/env python
# coding: utf-8

# import libraries
import time
import numpy as np
import networkx as nx

# import the networkX heterograph
G = nx.read_gpickle("~/AugSep2019_all.gpickle")

# getting nodes indices
p_idx, c_idx, u_idx = [], [], []
for i in range(len(G.nodes())):
    if G.nodes()[i]['otype'] == 'post':
        p_idx.append(i)
    elif G.nodes()[i]['otype'] == 'comment':
        c_idx.append(i)
    elif G.nodes()[i]['otype'] == 'user':
        u_idx.append(i)

# number of posts at each subgraph
train = []
for i in range(len(p_idx)):
    if i < 10:
        train.append(p_idx[i])

# drop duplicates in list
def unique(list1):
    unique_list = []
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

# creating sub-graph with given nmber of posts
uvp_train, uap_train, prc_train, uac_train, uvc_train = [], [], [], [], []
uvp_node_train, uap_node_train, prc_node_train, uac_node_train, uvc_node_train = [], [], [], [], []
for j in range(len(train)):
    for i in range(len(G.nodes())):
        if (i,train[j]) in G.edges() and G.edges[i,train[j]]['otype'] == 'vote':
            uvp_train.append((i,train[j]))
            uvp_node_train.append(i)
            uvp_node_train.append(train[j])
        elif (train[j],i) in G.edges() and G.edges[train[j], i]['otype'] == 'vote':
            uvp_train.append((i,train[j]))
            uvp_node_train.append(i)
            uvp_node_train.append(train[j])
        elif (i,train[j]) in G.edges() and G.edges[i,train[j]]['otype'] == 'authored':
            uap_train.append((i,train[j]))
            uap_node_train.append(i)
            uap_node_train.append(train[j])
        elif (train[j],i) in G.edges() and G.edges[train[j],i]['otype'] == 'reply':
            prc_train.append((train[j],i))
            prc_node_train.append(i)
            prc_node_train.append(train[j])
            for k in range(len(G.nodes())):
                if (k,i) in G.edges() and G.edges[k,i]['otype'] == 'authored':
                    uac_train.append((k,i))
                    uac_node_train.append(i)
                    uac_node_train.append(k)
                elif (k,i) in G.edges() and G.edges[k,i]['otype'] == 'vote':
                    uvc_train.append((k,i))
                    uvc_node_train.append(i)
                    uvc_node_train.append(k)
                elif (i,k) in G.edges() and G.edges[i,k]['otype'] == 'vote':
                    uvc_train.append((k,i))
                    uvc_node_train.append(i)
                    uvc_node_train.append(k)

train_nodes = unique(train + uvp_node_train + uap_node_train + prc_node_train + uac_node_train + uvc_node_train)
train_edges = unique(uvp_train + uap_train + prc_train + uac_train + uvc_train)

# creating sub-graph
train_G = G.subgraph(train_nodes)
G_train = train_G.edge_subgraph(train_edges)

# relabeling the sub-graph, so indices start from zero
keys = np.array(list(G_train.nodes))
values = [int(i) for i in np.arange(0, len(G_train.nodes))]
dic = dict(zip(keys, values))
G = nx.relabel_nodes(G_train, dic)

# Creating subgraph between t1 and t2
start = time.time()

# function to get node index
def get_key_node(val):
    key_indx_node = []
    for key, value in nx.get_node_attributes(G,'otype').items():
         if val == value:
            key_indx_node.append(key)
    return key_indx_node

# function to get edge index
def get_key_edge(val):
    key_indx_edge = []
    for key, value in nx.get_edge_attributes(G,'otype').items():
         if val == value:
            key_indx_edge.append(key)
    return key_indx_edge

# indices for each type of dict
post = get_key_node('post')
comment = get_key_node('comment')
user = get_key_node('user')
vote = get_key_edge('vote')
authored = get_key_edge('authored')
reply = get_key_edge('reply')

# set t1 and t2
t1 = 1564632000
t2 = 1565409600

# getting indices for nodes and edges between t1 and t2
subG_post = []
for i in post:
    if G.nodes()[i]['created'] >= t1 and G.nodes()[i]['created'] <= t2:
        subG_post.append(i)

subG_comment = []
for i in comment:
    if G.nodes()[i]['created'] >= t1 and G.nodes()[i]['created'] <= t2:
        subG_comment.append(i)

subG_vote = []
for i in vote:
    if G.edges()[i]['time'] >= t1 and G.edges()[i]['time'] <= t2:
        subG_vote.append(i)

subG_authored = []
for i in authored:
    if G.edges()[i]['time'] >= t1 and G.edges()[i]['time'] <= t2:
        subG_authored.append(i)

subG_reply = []
for i in reply:
    if G.edges()[i]['time'] >= t1 and G.edges()[i]['time'] <= t2:
        subG_reply.append(i)

# Creating the subgraph for all nodes, and then keep those that are between in t1 and t2
G_node = G.subgraph(subG_post+subG_comment+user)
G_post_t1_t2 = G_node.edge_subgraph(subG_vote + subG_authored + subG_reply)
end = time.time()
print("First query takes ", end - start, " sec.")

# Filter posts based on category
start = time.time()

# define category
category = 'steemit'

# function to get node index
def get_key_node_(val):
    key_indx_node = []
    for key, value in nx.get_node_attributes(G_post_t1_t2,'otype').items():
         if val == value:
            key_indx_node.append(key)
    return key_indx_node

# indices for post type dict
post_ = get_key_node_('post')

# creating the subgraph for specified category
subG_t1_t2_post = []
for i in post_:
    if G_post_t1_t2.nodes()[i]['category'] == category:
        subG_t1_t2_post.append(i)

# finding related nodes for the posts' category
related_node = []
related_edge = []
comment_related = []
for i in subG_t1_t2_post:
    for j in range(len(subG_vote)):
        if i == subG_vote[j][1]:
            related_node.append(subG_vote[j][0])
            related_edge.append(subG_vote[j])
    for j in range(len(subG_authored)):
        if i == subG_authored[j][1]:
            related_node.append(subG_authored[j][0])
            related_edge.append(subG_authored[j])
    for j in range(len(subG_reply)):
        if i == subG_reply[j][0]:
            related_node.append(subG_reply[j][1])
            related_edge.append(subG_reply[j])
            comment_related.append(subG_reply[j][1])

# finding related nodes for comments of the posts' category
for i in comment_related:
    for j in range(len(subG_vote)):
        if i == subG_vote[j][1]:
            related_node.append(subG_vote[j][0])
            related_edge.append(subG_vote[j])
    for j in range(len(subG_authored)):
        if i == subG_authored[j][1]:
            related_node.append(subG_authored[j][0])
            related_edge.append(subG_authored[j])

# drop duplicates
cat_node = []
for i in related_node:
    if i not in cat_node:
        cat_node.append(i)

# Creating the subgraph
G_post_t1_t2_cat = G_post_t1_t2.subgraph(cat_node+subG_t1_t2_post)
end = time.time()
print("Second query takes ", end - start, " sec.")

# Sort posts based on vote and comment velocity
start = time.time()

vote_count = []
comment_count = []
for i in G_post_t1_t2.nodes():
    vote = 0
    comment = 0
    if G_post_t1_t2.nodes()[i]['otype'] == 'post':
        for j in G_post_t1_t2.out_edges(i):
            if G_post_t1_t2.edges()[i,j[1]]['otype'] == 'reply':
                comment += 1
        comment_count.append([i,comment])
        for j in G_post_t1_t2.in_edges(i):
            if G_post_t1_t2.edges()[j[0],i]['otype'] == 'vote':
                vote += 1
        vote_count.append([i,vote])

# number of votes and comments all together
sum_cm_vote = []
for i in range(len(comments_sorted)):
    for j in range(len(votes_sorted)):
        if comments_sorted[i][0] == votes_sorted[j][0]:
            sum_cm_vote.append([comments_sorted[i][0], comments_sorted[i][1], votes_sorted[j][1],comments_sorted[i][1]+votes_sorted[j][1]])

# sort descending
comments_sorted = sorted(comment_count, key=lambda x: x[1], reverse = True)
votes_sorted = sorted(vote_count, key=lambda x: x[1], reverse = True)
sum_cm_vote_sorted = sorted(sum_cm_vote, key=lambda x: x[1], reverse = True)
end = time.time()
print("Third query takes ", end - start, " sec.")

# top 10 posts with highest number of votes and comments
win = []
for p in range(10):
    win.append([G_post_t1_t2.nodes()[sum_cm_vote_sorted[p][0]]['title'],G_post_t1_t2.nodes()[sum_cm_vote_sorted[p][0]]['payout'], sum_cm_vote_sorted[p][1], sum_cm_vote_sorted[p][2],sum_cm_vote_sorted[p][3]])

# Sorting users who are actively engaged based on their activity
start = time.time()
user_activity = []
for i in range(len(user)):
    if user[i] in G_post_t1_t2.nodes():
        user_activity.append([user[i], len(G_post_t1_t2.out_edges(user[i]))])

# sort descending
user_activity_sorted = sorted(user_activity, key=lambda x: x[1], reverse = True)
end = time.time()
print("Forth query takes ", end - start, " sec.")

# top 10 users with highest active engagement
user_list_win = []
for i in range(10):
    vote_u, author_u = 0, 0
    for j in range(len(G_post_t1_t2.nodes)):
        try:
            if G_post_t1_t2.edges()[user_activity_sorted[i][0],j]['otype'] == 'vote':
                vote_u += 1
            elif G_post_t1_t2.edges()[user_activity_sorted[i][0],j]['otype'] == 'authored':
                author_u += 1
        except:
            pass
    user_list_win.append([G_post_t1_t2.nodes()[user_activity_sorted[i][0]]['username'], vote_u, author_u])
