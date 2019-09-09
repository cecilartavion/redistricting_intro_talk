# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 07:24:17 2019

@author: jasplund
"""
#json_cb_path = 'C:/Users/jasplund/Downloads/cb_files/census_block_json/'
##json_cb_path = 'C:/Users/jasplund/Downloads/cb_files/census_grab_json/demo_json_city_names_nonplanar/'
#shp_path = 'C:/Users/jasplund/Downloads/'
#temp_BK = gpd.read_file("zip:"+shp_path+'tl_2010_48_tabblock10.zip')
#g = Graph.from_json(json_cb_path+"BLOCK_{}.json".format(st))
#new_g = g.copy()
#
#temp_cb_data = pd.DataFrame([new_g.nodes[node] for node in new_g.nodes()])
#n1 = temp_cb_data[temp_cb_data['BLOCKID10']=='482599703012035'].index[0]
#g.nodes[list(new_g.neighbors(n1))[0]]
#g.nodes[list(new_g.neighbors(n1))[1]]
#temp_cb_data[~temp_cb_data['BLOCKID10'].isin(BK['GEOID10'])]

import geopandas as gpd
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain,
                        proposals, updaters, constraints, accept, Election)
from gerrychain.updaters import Tally, cut_edges
from gerrychain.proposals import recom
from functools import partial
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
import os

#Build the congressional districts and plot the shape with colors for each
# congressional district.
shp_path = 'directory where shapefiles exists'
BK = gpd.read_file(shp_path+'tl_2016_us_cd115/tl_2016_us_cd115.shp') 
count = 0
for idx in list(BK[BK['STATEFP']=='24'].index):
    BK[BK['STATEFP']=='24'].loc[[idx],'geometry'].plot()
    plt.savefig("MD_cd_"+str(count)+".png", bbox_inches='tight', dpi=600)
    count+=1
BK[BK['STATEFP']=='24'].plot(column='GEOID', cmap='seismic', alpha=0.8)
plt.savefig("MD_state_legislature_upper_senate.png", bbox_inches='tight', dpi=600)

#Plot the census blocks for Maryland with one color for all blocks. 
shp_path = 'directory for original census block shapefiles/tl_2010_24_tabblock10/'
BK = gpd.read_file(shp_path+'tl_2010_24_tabblock10.shp') 
BK.columns
BK.plot(column='STATEFP10', cmap='gnuplot2', alpha=0.8)
plt.savefig("MD_census_blocks.png", bbox_inches='tight', dpi=400)


#plot the virginia precincts
shp_path = 'directory for precincts of Virgina/VA_precincts/'
#BK = gpd.read_file(shp_path+'tl_2010_24_tabblock10.shp') 
BK = gpd.read_file(shp_path+'VA_precincts.shp') 
BK.columns
BK['pct'] = BK['G16RPRS'].astype(float).div(np.sum(BK[['G16DPRS','G16RPRS']].astype(float),axis=1))
BK.plot(column='pct', cmap='seismic', alpha=0.8)
plt.savefig("VA_PR16_gen_election.png", bbox_inches='tight', dpi=600)



shp_path1 = 'directory for precincts of Virginia/VA_precincts/'
va_precincts = gpd.read_file(shp_path1+'VA_precincts.shp') 
#va_precincts['G16DPRS'] = va_precincts['G16DPRS'].astype(float)
#va_precincts['G16RPRS'] = va_precincts['G16RPRS'].astype(float)
#centroids = va_precincts.centroid
#va_precincts["C_X"] = centroids.x
#va_precincts["C_Y"] = centroids.y 
#pos = {index:(row['C_X'],row['C_Y']) for index,row in va_precincts.iterrows()}
graph = Graph.from_file(shp_path1+'VA_precincts.shp')
#change strings to floats
graph.to_json(shp_path1+'VA_precincts.json')
for node in graph.nodes():
    graph.nodes[node]['G16DPRS'] = float(graph.nodes[node]['G16DPRS'])
    graph.nodes[node]['G16RPRS'] = float(graph.nodes[node]['G16RPRS'])
    graph.nodes[node]['G18DSEN'] = float(graph.nodes[node]['G18DSEN'])
    graph.nodes[node]['G18RSEN'] = float(graph.nodes[node]['G18RSEN'])
    



#Create dataframe of the data on each node.
g_df = pd.DataFrame([graph.nodes[node] for node in graph.nodes()])
cd_verts = []
#Set list of colors.
colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown',
          'tab:pink','tab:gray','tab:olive','tab:cyan','darkblue','springgreen',
          'peru','dodgerblue','mediumslateblue','darkorchid','aqua']
pos = {node:(float(graph.nodes[node]['C_X']),float(graph.nodes[node]['C_Y'])) for node in graph.nodes}
#Build the dual graph for each congressional district.
count = 0 
#va_precincts.plot(color='white',edgecolor='white')
for cd in g_df.CD_16.unique():
    cd_verts.append(list(g_df[g_df['CD_16']==cd].index))
    nx.draw(graph.subgraph(list(g_df[g_df['CD_16']==cd].index)), pos=pos,node_size=2,node_color=colors[count],edge_color=colors[count])
    plt.savefig("va_precinct_"+str(count+1)+".png", bbox_inches='tight', dpi=400)
    plt.clf()
    count+=1
plt.savefig("VA_precinct_graph.png", bbox_inches='tight', dpi=400)
#CD.plot(column='CD108FP', cmap='seismic')
real_life_plan = Partition(graph, "CD")
plt.axis('off')
plt.show()
plt.savefig("va_precinct_graph.png", bbox_inches='tight', dpi=400)

#Gerrychain on 2016 election for virginia.
graph.nodes[0]
election = Election("PRES16", {"Dem": "G16DPRS", "Rep": "G16RPRS"})
initial_partition = Partition(
    graph,
    assignment="CD_16",
    updaters={
        "cut_edges": cut_edges,
        "population": Tally("TOTPOP", alias="population"),
        "PRES16": election
    }
)
for district, pop in initial_partition["population"].items():
    print("District {}: {}".format(district, pop))
    
from gerrychain import MarkovChain
from gerrychain.constraints import single_flip_contiguous
from gerrychain.proposals import propose_random_flip
from gerrychain.accept import always_accept

chain = MarkovChain(
    proposal=propose_random_flip,
    constraints=[single_flip_contiguous],
    accept=always_accept,
    initial_state=initial_partition,
    total_steps=1000
)
for partition in chain:
    print(sorted(partition["PRES16"].percents("Dem")))
d_percents = [sorted(partition["PRES16"].percents("Dem")) for partition in chain]

data = pd.DataFrame(d_percents)
import matplotlib.pyplot as plt

ax = data.boxplot(positions=range(len(data.columns)))
plt.plot(data.iloc[0], "ro")

plt.show()

#Output files to new directory.
newdir = "./Outputs/"
os.makedirs(os.path.dirname(newdir + "init.txt"), exist_ok=True)
with open(newdir + "init.txt", "w") as f:
    f.write("Created Folder")


#df["plot" + str(t)] = df["GEOID10"].map(dict(part.assignment))
#df.plot(column="plot" + str(t), cmap="tab20")
    
#Construct plot of congressional districts for each part assignment.       
count=0
for partition in chain:
    count+=1

#    print(count)
    partition.plot(geometries=va_precincts.geometry,cmap='tab20')
    plt.savefig(newdir + "plot" + str(count) + ".png")
    plt.close()


#Use ReCom to build gerrychain and plot.
elections = [
            Election("PRES16", {"Dem": "G16DPRS", "Rep": "G16RPRS"}),
            Election("SEN12", {"Dem": "G18DSEN", "Rep": "G18RSEN"})
            ]
my_updaters = {"population": updaters.Tally("TOTPOP", alias="population")}
election_updaters = {election.name: election for election in elections}
my_updaters.update(election_updaters)
initial_partition = GeographicPartition(graph, assignment="CD_16", updaters=my_updaters)
# The ReCom proposal needs to know the ideal population for the districts so that
# we can improve speed by bailing early on unbalanced partitions.

ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)

# We use functools.partial to bind the extra parameters (pop_col, pop_target, epsilon, node_repeats)
# of the recom proposal.
proposal = partial(recom,
                   pop_col="TOTPOP",
                   pop_target=ideal_population,
                   epsilon=0.03,
                   node_repeats=2
                  )
compactness_bound = constraints.UpperBound(
    lambda p: len(p["cut_edges"]),
    2*len(initial_partition["cut_edges"])
)


pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, 0.03)
chain1 = MarkovChain(
    proposal=proposal,
    constraints=[
        pop_constraint,
        compactness_bound
    ],
    accept=accept.always_accept,
    initial_state=initial_partition,
    total_steps=50000
)

data = pd.DataFrame(
    partition["SEN12"].mean_median()
    for partition in chain.with_progress_bar()
)
temp = [partition["PRES16"].mean_median() for partition in chain]
temp1 = [partition["PRES16"].efficiency_gap() for partition in chain]
plt.hist(temp ,bins=200)

count=0
for partition in chain1:
    count+=1
    if count<10:
        print(count)
        partition.plot()

partition['PRES16'].mean_median()

fig, ax = plt.subplots(figsize=(8, 6))

# Draw 50% line
ax.axhline(0.5, color="#cccccc")

# Draw boxplot
data.boxplot(ax=ax, positions=range(len(data.columns)))

# Draw initial plan's Democratic vote %s (.iloc[0] gives the first row)
plt.plot(data.iloc[0], "ro")

# Annotate
ax.set_title("Comparing the 2011 plan to an ensemble")
ax.set_ylabel("Democratic vote % (Senate 2012)")
ax.set_xlabel("Sorted districts")
ax.set_ylim(0, 1)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])

plt.show()


