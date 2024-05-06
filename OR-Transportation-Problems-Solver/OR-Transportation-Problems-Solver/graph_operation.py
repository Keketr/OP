import networkx as nx
from tabulate import tabulate
import copy
from random import *
import math
from read_and_display import *

# takes all the processed data, starts in the north-west corner and put the maximum possible allocation after a comparison between order and provision (minimum)
# Sets provision or order completely used to 0 temporarily to keep going through the matrix
def NorthWest(datalist,order,provision):
    ordercp=copy.deepcopy(order)
    provisioncp=copy.deepcopy(provision)
    data=[]
    for i in datalist:
        temp=[]
        for j in i:
            temp.append(0)
        data.append(temp)
    i=0
    j=0
    while(i<len(provisioncp) and j<len(ordercp)):
        quantity=min(ordercp[j],provisioncp[i])
        data[i][j]=quantity
        provisioncp[i]-=quantity
        ordercp[j]-=quantity
        if provisioncp[i]==0:
            i+=1
        if ordercp[j]==0:
            j+=1
        printing(data,ordercp,provisioncp)
    return(data,ordercp,provisioncp)

# Compute an original proposition using Vogel algorithm
def BalasHammer(datalist,order,provision,costlist):
    datalistcp=copy.deepcopy(datalist)
    ordercp=copy.deepcopy(order)
    provisioncp=copy.deepcopy(provision)
    
    data=[]
    for i in datalist:
        temp=[]
        for j in i:
            temp.append(None)
        data.append(temp)
    cpt=0
    while(notallzero(ordercp)==0 and notallzero(provisioncp)==0):
        cpt+=1
        linePenalty=[]
        columnPenalty=[]
        
        for i in datalistcp:
            min1=None
            min2=None
            for j in i:
                if min1==None:
                    min1=j
                elif min2==None:
                    if j<min1:
                        min2=min1
                        min1=j
                    else:
                        min2=j
                else:
                    if j<min1:
                        min2=min1
                        min1=j
                    elif j<min2:
                        min2=j
            if min2==100000000000 and min1!=100000000000:
                linePenalty.append(min1)
            else:
                linePenalty.append(min2-min1)
        for j in range(len(datalistcp[0])):
            min1 = None
            min2 = None
    
            for i in range(len(datalistcp)):
                if min1 is None:
                    min1 = datalistcp[i][j]
                elif min2 is None:
                    if datalistcp[i][j] < min1:
                        min2 = min1
                        min1 = datalistcp[i][j]
                    else:
                        min2 = datalistcp[i][j]
                else:
                    if datalistcp[i][j] < min1:
                        min2 = min1
                        min1 = datalistcp[i][j]
                    elif datalistcp[i][j] < min2:
                        min2 = datalistcp[i][j]
            if min2==100000000000 and min1!=100000000000:
                columnPenalty.append(min1)
            else:
                columnPenalty.append(min2 - min1)
        maxiPenalty=-1
        cost=None
        for i in range(len(linePenalty)):
            if cost==None:
                maxiPenalty=linePenalty[i]
                changei=i
                for j in range(len(costlist[i])):
                    if data[i][j]==None:
                        if cost==None:
                            cost=costlist[i][j]
                            changej=j
                            tempcost=cost
                        elif cost>costlist[i][j]:
                            cost=costlist[i][j]
                            tempcost=cost
                            changej=j
            elif linePenalty[i]>maxiPenalty:
                maxiPenalty=linePenalty[i]
                changei=i
                cost=None
                for j in range(len(costlist[i])):
                     if data[i][j]==None:
                         if cost==None:
                            cost=costlist[i][j]
                            tempcost=cost
                            changej=j
                         elif cost>costlist[i][j]:
                            cost=costlist[i][j]
                            changej=j
                            tempcost=cost
            elif linePenalty[i]==maxiPenalty:
                cost=tempcost
                for j in range(len(costlist[i])):
                     if data[i][j]==None:
                         if cost>costlist[i][j]:
                            cost=costlist[i][j]
                            changei=i
                            changej=j
                            tempcost=cost
        for j in range(len(columnPenalty)):
            
            if columnPenalty[j]>maxiPenalty:
                maxiPenalty=columnPenalty[j]
                changej=j
                cost=None
                for i in range(len(costlist)):
                     if data[i][j]==None:
                         if cost==None:
                            cost=costlist[i][j]
                            tempcost=cost
                            changei=i
                                
                         elif cost>costlist[i][j]:
                            cost=costlist[i][j]
                            tempcost=cost
                            changei=i
            elif columnPenalty[j]==maxiPenalty:
                cost=tempcost
                for i in range(len(costlist)):
                     if data[i][j]==None:
                         if cost>costlist[i][j]:
                            cost=costlist[i][j]
                            changei=i
                            changej=j
                            tempcost=cost

                
        if data[changei][changej]==None:
            orderprint=copy.deepcopy(ordercp)
            provisionprint=copy.deepcopy(provisioncp)
            quantity=min(ordercp[changej],provisioncp[changei])
            data[changei][changej]=quantity
            datalistcp[changei][changej]=100000000000
            provisioncp[changei]-=quantity
            ordercp[changej]-=quantity
    
            if provisioncp[changei]==0:
                for j in range(len(data[changei])):
                    if data[changei][j]==None:
                        data[changei][j]=0
                        datalistcp[changei][j]=100000000000
            if ordercp[changej]==0:
                for i in range(len(data)):
                    if data[i][changej]==None:
                        data[i][changej]=0
                        datalistcp[i][changej]=100000000000
            printingPenalty(data, orderprint, provisionprint,linePenalty,columnPenalty)
    return(data,ordercp,provisioncp)

# Calculates the total cost of a proposal
def Costculation(transportdata,datalist,p=True):
    cal=0
    for i in range(len(transportdata)):
        for j in range(len(transportdata[i])):
            cal+=transportdata[i][j]*datalist[i][j]
    if p:
        print(cal)
    return cal

# True if the list is full of 0. False otherwise.
def notallzero(liste):
    for i in liste:
        if i!=0:
            return 0
    return 1

# Using library nx, checks whether there is a cycle in a graph. If there is, returns it
def testcircular(g,p=False):
    try:
        cycle=nx.find_cycle(g, orientation='original')
        string=""
        for i in cycle:
            string=string + " ==> "+ i[0]
        a=cycle
    except:
        a=1
    return a

# Removes a cycle from a graph
def rectifCircular(test,transportdata, graph,order,provision,upgrade):
    transporcp=copy.deepcopy(transportdata) 
    
    do=None
    mini=None
    # Finds the minimum edge in the cycle
    for data in test:
        if "P" in data[1]:
            i=int(data[1][1:])-1
            j=int(data[0][1:])-1
            if mini==None:
                do="P"
                mini=transporcp[i][j]
            if transporcp[i][j]<mini:
                do="P"
                mini=transporcp[i][j]
        else:
            i=int(data[0][1:])-1
            j=int(data[1][1:])-1
            if mini==None:
                do="C"
                mini=transporcp[i][j]
            if transporcp[i][j]<mini:
                do="C"
                mini=transporcp[i][j]
    quantity=None
    
    # Finds the minimal quantity among edges that will decrease
    for data in test:
        if do=="C":
            i=int(data[1][1:])-1
            j=int(data[0][1:])-1
            if "P" in data[1]:
                if quantity ==None:
                    
                    quantity=transporcp[i][j]
                if quantity>transporcp[i][j]:
                    quantity=transporcp[i][j]
        else:
            i=int(data[0][1:])-1
            j=int(data[1][1:])-1
            if "C" in data[1]:
                if quantity ==None:
                    quantity=transporcp[i][j]
                if quantity>transporcp[i][j]:
                    quantity=transporcp[i][j]
    
    # Deletes an edge with quantity = 0 that is part of the edges that should decrease to fix the cycle
    if quantity==0:
        for data in test:
            if "C" in data[0]:
                i=int(data[1][1:])-1
                j=int(data[0][1:])-1
            else:
                i=int(data[0][1:])-1
                j=int(data[1][1:])-1
            if transporcp[i][j]==0:
                if upgrade[1]!=i or upgrade[0]!=j:
                    graph.remove_edge(data[0],data[1])

    # Shifts the values in the cycle        
    else:
        for data in test:
            if do=="C":
                i=int(data[1][1:])-1
                j=int(data[0][1:])-1
                if "P" in data[1]:
                    transporcp[i][j]-=quantity
                else:
                    transporcp[j][i]+=quantity
            else:
                i=int(data[1][1:])-1
                j=int(data[0][1:])-1
                if "P" in data[1]:
                    transporcp[i][j]+=quantity
                else:
                    transporcp[j][i]-=quantity
        transportdata=transporcp
        
        # Removes edges that are now 0 in quantity
        for data in test:
            
            if "P" in data[1]:
                i=int(data[1][1:])-1
                j=int(data[0][1:])-1
                if transporcp[i][j]==0:
                    graph.remove_edge(data[0],data[1])
            else:
                
                i=int(data[0][1:])-1
                j=int(data[1][1:])-1
                if transporcp[i][j]==0:
                    graph.remove_edge(data[0],data[1])
    return(transportdata,graph)

# True if the graph is connected, false otherwise 
def can_reach_all_nodes(graph):
    num_nodes = len(graph.nodes())
    
    # Go through every node of a graph
    for node in graph.nodes():
        visited = set()  # Every visited node
        
        def dfs(current_node):
            visited.add(current_node)
            
            # Go through every neighbour of current node
            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited:
                    dfs(neighbor)
        
        dfs(node)
        
        # Check that all nodes have been visited
        if len(visited) != num_nodes:
            return False  
    
    return True

# Resolves the system for potential of each node
# Goes through the dictionnary and check whether one of the two nodes of an edge has been found. if one has been found, calculates the other
def calculate_potentials(graph, costs):
    num_nodes = len(graph.nodes())
    potentials = {}
    for i in graph.nodes:
        potentials[i]=None
    
    start_node = list(graph.nodes())[0]
    potentials[start_node] = 0
    todo=copy.deepcopy(graph.edges)
    while None in potentials.values():
        for i in todo:
    
            if potentials[i[0]]!=None and potentials[i[1]]==None:
                if "C" in i[0]:
                    ical=int(i[1][1:])-1
                    jcal=int(i[0][1:])-1
                    potentials[i[1]]=costs[ical][jcal]+potentials[i[0]]
                else:
                    ical=int(i[0][1:])-1
                    jcal=int(i[1][1:])-1
                    potentials[i[1]]=potentials[i[0]]-costs[ical][jcal]
            elif potentials[i[1]]!=None and potentials[i[0]]==None:
                if "C" in i[0]:
                    ical=int(i[1][1:])-1
                    jcal=int(i[0][1:])-1
                    potentials[i[0]]=-costs[ical][jcal]+potentials[i[1]]
                else:
                    ical=int(i[0][1:])-1
                    jcal=int(i[1][1:])-1
                    potentials[i[0]]=potentials[i[1]]-costs[ical][jcal]
    return potentials

# Uses potentials to compute potential costs matrix
def calculate_potential_costs(table, potentials):

    potential_costs = []
    tableC=[]
    tableP=[]
            
    for i in range(len(table)):
        tableP.append(i)
    for i in range(len(table[0])):
        tableC.append(i)
    for i in range(len(tableP)):
        temp=[]
        for j in range(len(tableC)):
            temp.append(potentials["P"+str(i+1)]-potentials["C"+str(j+1)])
        potential_costs.append(temp)
    return potential_costs

# Uses cost and potential cost tables to create marginal costs table (Costs - potential costs)
def calculate_marginal_costs(costs, potential_costs):
    marginal_costs=[]
    for i in range(len(potential_costs)):
        temp=[]
        for j in range(len(potential_costs[i])):
            temp.append(costs[i][j]-potential_costs[i][j])
        marginal_costs.append(temp)
    return(marginal_costs)

# detect the greatest negative number among marginal costs
def detect_best_improvement(marginal_costs,transportdata):
    maxi=0
    for i in marginal_costs:
        for j in i:
            if transportdata[marginal_costs.index(i)][i.index(j)]==0:
                if maxi>j:
                    maxi=j
                    imaxi=marginal_costs.index(i)
                    jmaxi=i.index(j)
    if maxi!=0:
        return (jmaxi,imaxi)

# From a list of every subgraph, add edges with minimal cost that allow the graph to be connected
def testContinuity(g,cost):
    while not nx.is_connected(g):
        connected_components = list(nx.connected_components(g))
        i1=[]
        j1=[]
        i2=[]
        j2=[]
        for i in connected_components[0]:
            if "C" in i:
                j1.append(int(i[1:])-1)
            else:
                i1.append(int(i[1:])-1)
        for i in connected_components[1]:
            if "C" in i:
                j2.append(int(i[1:])-1)
            else:
                i2.append(int(i[1:])-1)
        mini=None
        for i in i1:
            for j in j2:
                if mini==None:
                    doi=i
                    doj=j
                    mini=cost[i][j]
                elif mini>cost[i][j]:
                    doi=i
                    doj=j
                    mini=cost[i][j]
        for i in i2:
            for j in j1:
                if mini==None:
                    doi=i
                    doj=j
                    mini=cost[i][j]
                elif mini>cost[i][j]:
                    doi=i
                    doj=j
                    mini=cost[i][j]
        g.add_edge("P"+str(doi+1),"C"+str(doj+1))
    return g

def randTable(size):
    datalist=[]
    order=[]
    sizeorder=0
    provision=[]
    for i in range(size):
        temp=[]
        ordertemp=randint(1,100)
        sizeorder+=ordertemp
        order.append(ordertemp)
        for j in range(size):
            temp.append(randint(1,100))
        datalist.append(temp)
    provision=copy.deepcopy(order)
    shuffle(provision)
    return(datalist,order,provision)

def printingMarginal(marginal_costs):
    for i in range(len(marginal_costs)):
        for j in range(len(marginal_costs[i])):
            marginal_costs[i][j]=math.sqrt(marginal_costs[i][j]**2)
    print(tabulate(marginal_costs, tablefmt="grid"))
