from dash import Dash
import time
import dash_daq as daq
from itertools import combinations, product
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import base64
import io
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State

rk_nodes = []
with open("rk_nodes.txt") as f:
    for line in f: rk_nodes.append(line.strip())

rk_edges = []
with open("rk_edges.txt") as f:
    for line in f: rk_edges.append(line.strip())

app = Dash(__name__)
start_drop = dcc.Dropdown(
    id="start-dropdown",
   options=[
       {'label':x, 'value':x} for x in rk_nodes 
   ],
   value="chemical_substance",
   clearable=False
)

tail_drop = dcc.Dropdown(
    id="tail-dropdown",
   options=[
       {'label':x, 'value':x} for x in rk_nodes 
   ]#,
   #value="disease"
)

node_drop = dcc.Dropdown(
    id="node-dropdown",
   options=[
       {'label':x, 'value':x} for x in rk_nodes 
       #{'label': 'New York City', 'value': 'NYC'},
       #{'label': 'Montreal', 'value': 'MTL'},
       #{'label': 'San Francisco', 'value': 'SF'},
   ],
   multi=True
)
k_drop = []
for k in range(1,6):
    drop = html.Div([
        html.B(children='Level k-%i:'%k),
        dcc.Dropdown(
        id="node-dropdown-%i" % k,
            options=[
                {'label':x, 'value':x} for x in rk_nodes 
       #{'label': 'New York City', 'value': 'NYC'},
       #{'label': 'Montreal', 'value': 'MTL'},
       #{'label': 'San Francisco', 'value': 'SF'},
            ],
        multi=True
    )],
    id="node-div-%i"%k,
    style={'display':'block'}
    )
    k_drop.append(drop)

edge_drop = dcc.Dropdown(
    id="edge-dropdown",
   options=[
       {'label':x, 'value':x} for x in rk_edges 
       #{'label': 'New York City', 'value': 'NYC'},
       #{'label': 'Montreal', 'value': 'MTL'},
       #{'label': 'San Francisco', 'value': 'SF'},
   ],
   multi=True
)

k_sel = daq.NumericInput(
   id="k-select",
   min=1,
   max=5,
   value=2
) 

pos_pairs = html.Div([
    html.Div(html.B(children='Positive Pairs:')),
    dcc.Textarea(
        id='positive_pairs',
        value='aspirin, ibuprofen',
        style={'width': '20%', 'height': 300},
)])

neg_pairs = html.Div([
    html.Div(html.B(children='Negative Pairs:\n')),
    dcc.Textarea(
        id='negative_pairs',
        value='aspirin, ibuprofen',
        style={'width': '20%', 'height': 300},
)])

app.layout = html.Div([
    html.B(children='Start Node:'),
    start_drop,
    html.B(children='Tail Node:'),
    tail_drop,
    html.B(children='K Value:'),
    k_sel,
#    html.H4(children='Pathway Node Labels:'),
#    node_drop,
    k_drop[0],
    k_drop[1],
    k_drop[2],
    k_drop[3],
    k_drop[4],
#    html.H4(children='Pathway Edge Labels:'),
#    edge_drop,
#    html.Div(id='edge-output-container'),
#    html.Div(dcc.Input(id='input-on-submit', type='text')),
    html.Div(id='dd-output-container'),
    html.Img(id='img-test'),
    pos_pairs, 
    neg_pairs,
    html.Button('Submit', id='submit-val', n_clicks=0),
    html.Div(id='container-button-basic', children='Enter a value and press submit')
])


selected_nodes = []



selected_edges = []



@app.callback(
    Output('container-button-basic', 'children'),
    Input('submit-val', 'n_clicks'),
    [State('positive_pairs', 'value'), State('negative_pairs','value')]#,
#    State('edge-dropdown', 'edges')
)
#1def update_output(n_clicks, nodes, edges):
def update_output(n_clicks,pos_pairs,neg_pairs):#, sel_nodes, sel_edges):
    x = pos_pairs + neg_pairs
    y = "HI"

    return 'The input value was "{}" "{}" and the button has been clicked {} times'.format(
        x,
        y,
        n_clicks
    )

#@app.callback(
#    Output('img-test', 'src'),
#    Input('submit-val', 'n_clicks'),
#    [State('start-dropdown','value'),State('tail-dropdown','value'),State('node-dropdown', 'value'), State('edge-dropdown','value')]#,
#)
#def update_output(n_clicks,s,t,nodes,edges):
#    if(nodes==None or nodes==[]):return "data:image/png;base64,{}".format('')
#    fig, ax = plt.subplots(1,1)
#    buf = io.BytesIO() # in-memory files
#    k = 3
#    pairs_list = []
#    for i in range(k-1):
#        pairs_list.append(list(product(nodes,nodes)))
#    COP_Visualize(ax, s, t, pairs_list)
#        
#    fig.savefig(buf, format = "png") # save to the above file object
#    plt.close()
#    data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
#    return "data:image/png;base64,{}".format(data)
def buildCypher(s,t,k_nodes,k_val):
    query = "MATCH "
    query += "(s:`%s`)" % s
    for k in range(k_val):
        query += "--(x%i)" % (k+1)
    if(t!=None): query+= "--(t:`%s`)" % t

    label_clauses = []
    for k in range(k_val):
        if(k_nodes[k]==None or len(k_nodes[k])==0):continue
        #where_str += ( ) 
        clauses = ["x%i:`%s`" % (k+1,label) for label in k_nodes[k]]
        clause_str = " ( " + " OR ".join(clauses) + " ) "
        label_clauses.append(clause_str)
    if(len(label_clauses)!=0):
        query += " WHERE "
        query += " AND ".join(label_clauses)
    query += " RETURN * " 
    return query




@app.callback(
    [
        Output('img-test', 'src'),
        Output('dd-output-container', 'children')
    ],
    [
        Input("start-dropdown", 'value'), 
        Input("tail-dropdown", 'value'), 
        Input("node-dropdown-1", 'value'), 
        Input("node-dropdown-2", 'value'), 
        Input("node-dropdown-3", 'value'), 
        Input("node-dropdown-4", 'value'), 
        Input("node-dropdown-5", 'value'),
        Input('k-select', 'value')
    ]
)
def update_output(s,t,k1_nodes,k2_nodes,k3_nodes,k4_nodes,k5_nodes,k_val):
    if(s=="chemical_substance" and k1_nodes==None and k2_nodes==None and k_val==2):
        query = buildCypher(s,t,[k1_nodes,k2_nodes,k3_nodes,k4_nodes,k5_nodes],k_val)
        with open("base_image.txt") as f:
            image = f.read()
        return (image,query)
       # return "data:image/png;base64,{}".format(''), ""
    node_layers = []
    if(k_val >= 1): 
        if(k1_nodes!=None and len(k1_nodes)>0):node_layers.append(k1_nodes)
        else: node_layers.append(["*"])
    if(k_val >= 2): 
        if(k2_nodes!=None and len(k2_nodes)>0):node_layers.append(k2_nodes)
        else: node_layers.append(["*"])
    if(k_val >= 3): 
        if(k3_nodes!=None and len(k3_nodes)>0):node_layers.append(k3_nodes)
        else: node_layers.append(["*"])
    if(k_val >= 4): 
        if(k4_nodes!=None and len(k4_nodes)>0):node_layers.append(k4_nodes)
        else: node_layers.append(["*"])
    if(k_val >= 5): 
        if(k5_nodes!=None and len(k5_nodes)>0):node_layers.append(k5_nodes)
        else: node_layers.append(["*"])
    fig, ax = plt.subplots(1,1)
    buf = io.BytesIO() # in-memory files
    pairs_list = []
    print("NODE LAYERS:" , node_layers)
    if(k_val==1):
        pairs_list.append([(n,None) for n in node_layers[0]])
    else:
        for i in range(k_val-1):
            pairs_list.append(list(product(node_layers[i],node_layers[i+1])))
            #pairs_list.append(list(product(node_layers[i],node_layers[i+1])))
    COP_Visualize(ax, s, t, pairs_list)
        
    fig.savefig(buf, format = "png") # save to the above file object
    plt.close()
    data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    image = "data:image/png;base64,{}".format(data)
    query = buildCypher(s,t,[k1_nodes,k2_nodes,k3_nodes,k4_nodes,k5_nodes],k_val)
    return (image, query)


def COP_Visualize(ax,start,tail,pairs_list):
    nodes = set()
    nodes.add(start)
    if(tail!=None):
        nodes.add(tail)
    for count, pairs in enumerate(pairs_list):
        depth = count+1 #Want a depth value that shows how far we are from start. Each pair in pairs_list is the next layer of depth (with seed as 0th layer).
        for (x,y) in pairs:
            nodes.add(x+str(depth))
            if(y!=None): nodes.add(y+str(depth+1))

    nodes = list(nodes)

    colors = []
    pos_dict={start:(0,0)}
    for n in nodes:
        if(n.endswith("1")): 
            colors.append("lightblue")
        elif(n.endswith("2")): 
            colors.append("pink")
        elif(n.endswith("3")): 
            colors.append("lightgreen")
        elif(n.endswith("4")): 
            colors.append("lightcoral")
        elif(n.endswith("5")): 
            colors.append("plum")
        else: colors.append("tan")
    G=nx.DiGraph()
    G.add_nodes_from(nodes)
    print(G.nodes())
    
    for count, pairs in enumerate(pairs_list):
        depth = count+1 #Want a depth value that shows how far we are from start. Each pair in pairs_list is the next layer of depth (with seed as 0th layer).
        print('len',len(pairs_list))
        print('dep',depth)
        print('tail',tail)
        if(depth==1 and len(pairs_list)==1):
            for (x,y) in pairs:
                G.add_edge(start,x+str(depth))
                if(y==None and tail!=None): G.add_edge(x+str(depth),tail)
                if(y!=None): G.add_edge(x+str(depth),y+str(depth+1))
                if(y!=None and tail!=None): G.add_edge(y+str(depth+1),tail)
        elif(depth==1):
            for (x,y) in pairs:
                G.add_edge(start,x+str(depth))
                if(y!=None): G.add_edge(x+str(depth),y+str(depth+1))
        elif(depth==len(pairs_list) and tail!=None):
            print("HI")
            for (x,y) in pairs:
                G.add_edge(x+str(depth),y+str(depth+1))
                G.add_edge(y+str(depth+1),tail)
        else:
            for (x,y) in pairs:
                G.add_edge(x+str(depth),y+str(depth+1))

    pos = graphviz_layout(G, prog="dot",root=start)
    print(nx.is_tree(G))
    print(colors)
#    nx.draw(G, with_labels = True, pos=pos, ax=ax, node_color=colors, node_size=1000)
    nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color=colors, node_size=1000)
    nx.draw_networkx_labels(G, pos=pos)
    nx.draw_networkx_edges(G, pos=pos)
#    nx.draw_networkx_edge_labels(G, pos=pos,edge_labels={("Seed","Gene1"):"HI"})
    ax.set_xlim(ax.get_xlim()[0]-15,ax.get_xlim()[1]+15)


@app.callback(
    [
    Output("node-div-1",'style'),
    Output("node-div-2",'style'),
    Output("node-div-3",'style'),
    Output("node-div-4",'style'),
    Output("node-div-5",'style'),
    ],
    Input('k-select', 'value')
)
def update_output(value):
    k = value
    print('hi')
    style_1 = {'display':'block'} if k>=1 else {'display':'None'}
    style_2 = {'display':'block'} if k>=2 else {'display':'None'}
    style_3 = {'display':'block'} if k>=3 else {'display':'None'}
    style_4 = {'display':'block'} if k>=4 else {'display':'None'}
    style_5 = {'display':'block'} if k>=5 else {'display':'None'}

    return style_1, style_2, style_3, style_4, style_5

if __name__ == '__main__':
    app.run_server(host="0.0.0.0",port=80,debug=True)
