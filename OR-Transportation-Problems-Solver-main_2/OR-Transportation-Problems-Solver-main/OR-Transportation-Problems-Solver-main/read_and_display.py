from tabulate import tabulate
import copy
import networkx as nx
import matplotlib.pyplot as plt
from random import *
import math
import time


def display_graph(datalist,p=True):
    G = nx.Graph()
    
    cols = len(datalist[0])
    for j in range(cols):
        G.add_node(f"C{j+1}", pos=(j, 1))  
    
    rows = len(datalist)
    for i in range(rows):
        G.add_node(f"P{i+1}", pos=(i, 0))
    
   
    for i in range(rows):
        for j in range(cols):
            if datalist[i][j] != 0:
                G.add_edge(f"P{i+1}", f"C{j+1}", weight=datalist[i][j])
    
    return G

   

def readfile(url):
    try:
        with open(f"Data\{url}", 'r') as file:
    
            data=[]
            provision=[]
            order=[]
            lines=file.readlines()
            high=1
            lenght=1
            for i in range(len(lines)):
                if i==0:
                    temp=lines[i].split()
                    high=int(temp[0])
                    lenght=int(temp[1])
                elif i<high+1:
                    temp=[]
                    d=lines[i].split()
                    for j in range(lenght):
                        temp.append(int(d[j]))
                    data.append(temp)
                    provision.append(int(d[lenght]))
                else:
                    d=lines[i].split()
                    for i in d:
                        order.append(int(i))
            for i in range(len(provision)):
                provision[i]=int(provision[i])                
            return (data,order,provision)
    except FileNotFoundError:
        print(f"Le fichier {url} n'a pas été trouvé.")
        return None
    

def printing(datalist, order, provision):
    datalist_copy = copy.deepcopy(datalist)
    order_copy = copy.deepcopy(order)
    provision_copy = copy.deepcopy(provision)

    headers = ["-"]
    for i in range(len(datalist_copy[0])):
        headers.append(f"C{i+1}")
    headers.append("Provision")

    for i in range(len(datalist_copy)):
        datalist_copy[i].insert(0, f"P{i+1}")
        datalist_copy[i].append(provision_copy[i])
    order_copy.insert(0, "Order")
    order_copy.append("-")
    datalist_copy.append(order_copy)

    table = tabulate(datalist_copy, headers=headers, tablefmt="grid")
    print(table)
    
def printingPenalty(datalist, order, provision,linePenalty,columnPenalty):
    datalist_copy = copy.deepcopy(datalist)
    order_copy = copy.deepcopy(order)
    provision_copy = copy.deepcopy(provision)
    columnPenaltycp= copy.deepcopy(columnPenalty)
    linePenaltycp= copy.deepcopy(linePenalty)
    headers = ["-"]
    for i in range(len(datalist_copy[0])):
        headers.append(f"C{i+1}")
    headers.append("Provision")
    headers.append("Penalty")
    for i in range(len(datalist_copy)):
        datalist_copy[i].insert(0, f"P{i+1}")
        datalist_copy[i].append(provision_copy[i])
        if provision_copy[i]==0:
            datalist_copy[i].append("-")
        else:
            datalist_copy[i].append(linePenaltycp[i])
    order_copy.insert(0, "Order")
    order_copy.append("-")
    columnPenaltycp.insert(0,"Penalty")
    columnPenaltycp.append("-")
    for i in range(len(order_copy)):
        if order_copy[i]==0:
            columnPenaltycp[i]="-"
    datalist_copy.append(order_copy)
    datalist_copy.append(columnPenaltycp)
    table = tabulate(datalist_copy, headers=headers, tablefmt="grid")
    print(table)
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
    while(asnozero(ordercp)==0 and asnozero(provisioncp)==0):
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
def Costculation(transportdata,datalist,p=True):
    cal=0
    for i in range(len(transportdata)):
        for j in range(len(transportdata[i])):
            cal+=transportdata[i][j]*datalist[i][j]
    if p:
        print(cal)
    return cal
def asnozero(liste):
    for i in liste:
        if i!=0:
            return 0
    return 1
def testcircular(g,p=False):
    """
    Thanks to networkx librairy we can directly check if there is a cycle in our graph
    """
    try:
        cycle=nx.find_cycle(g, orientation='original')
        if p:
            print("There is a cycle in the graphe")
        string=""
        for i in cycle:
            string=string + " ==> "+ i[0]
        if p!=False:
            print(string)
        a=cycle
    except:
        a=1
        if p:
            print("there is no cycle in this graph")
    return a
    
def rectifCircular(test,transportdata, graph,order,provision,upgrade):
    transporcp=copy.deepcopy(transportdata) 
    
    do=None
    mini=None
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
def can_reach_all_nodes(graph):
    num_nodes = len(graph.nodes())
    
    # Parcourir tous les nœuds du graphe
    for node in graph.nodes():
        visited = set()  # Pour stocker les nœuds visités
        
        # Fonction DFS pour explorer le graphe à partir du nœud actuel
        def dfs(current_node):
            visited.add(current_node)
            
            # Parcourir tous les voisins du nœud actuel
            for neighbor in graph.neighbors(current_node):
                if neighbor not in visited:
                    dfs(neighbor)
        
        # Exécuter DFS à partir du nœud actuel
        dfs(node)
        
        # Vérifier si tous les nœuds ont été visités
        if len(visited) != num_nodes:
            return False  # Si tous les nœuds ne sont pas accessibles, retourner False
    
    return True

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


    
def display_potentials(potentials):
    print("Potentiels par sommet :")
    for node, potential in potentials.items():
        print(f"Sommet {node}: {potential}")
    print()
    
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

def calculate_marginal_costs(costs, potential_costs):
    marginal_costs=[]
    for i in range(len(potential_costs)):
        temp=[]
        for j in range(len(potential_costs[i])):
            temp.append(costs[i][j]-potential_costs[i][j])
        marginal_costs.append(temp)
    return(marginal_costs)

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
    """for idx in range(size-1):
        provision.append(randint(1,sizeorder-sum(provision)-size+idx))
    provision.append(sizeorder-sum(provision))
    shuffle(provision)
    """
    provision=copy.deepcopy(order)
    shuffle(provision)
    return(datalist,order,provision)
def printingMarginal(marginal_costs):
    for i in range(len(marginal_costs)):
        for j in range(len(marginal_costs[i])):
            marginal_costs[i][j]=math.sqrt(marginal_costs[i][j]**2)
    print(tabulate(marginal_costs, tablefmt="grid"))




timeavant=time.time()   
timemethode=0 
for i in range(0,101):
    i=0
    while i==0:
        print("Hello\nHow are you\nDo you want to test a file or to generate a random one?(1,2)")
        k=2
        if k==1:
            print("Which file do you want to test?")
            a=str(input())
            data=readfile(a)
            if data!=None:
                i=1
        elif k==2:
            print("Which size do you want to do?")
            #il faut modifier en dessous la size
            size=4
            data=randTable(size)
            i=1
    
    i=0
    
    while i!="2" and i!="1":
        print("Which method do you want to use? 1 for North West, 2 for Balas")
        #modifier i pour faire balas ou northwest
        i="2"
    
    (datalist,order,provision)=data
    printing(datalist,order,provision)
    timeavantmethode=time.time() 
    if i=="1":   
         (transportdata,transportorder,transportprovision)=NorthWest(datalist, order, provision)
    else:
        (transportdata,transportorder,transportprovision)=BalasHammer(datalist,order,provision,datalist)
    timemethode+=time.time()-timeavantmethode
    a=0
    printing(transportdata,transportorder,transportprovision)
    g=display_graph(transportdata)
    best_improvement=None
    err=0
    precost=Costculation(transportdata,datalist,p=False)
    while a==0 :
        
        print("start test")
        test=testcircular(g,p=True)
        if can_reach_all_nodes(g)==False:
            print("The graph is degenerate")
        while test!=1 or can_reach_all_nodes(g)==False:
            
            if test!=1:
                (transportdata,g)=rectifCircular(test,transportdata, g,order,provision,best_improvement)
            if can_reach_all_nodes(g)==False:
                g=testContinuity(g,datalist)
            
        print("Test finish")
        potentials = calculate_potentials(g, datalist)
        
        display_potentials(potentials)
        potential_costs = calculate_potential_costs(transportdata, potentials)
        marginal_costs = calculate_marginal_costs(datalist, potential_costs)
        print('the potential costs is:')
        print(tabulate(potential_costs, tablefmt="grid"))
        print('the marginal costs is:')
        print(tabulate(marginal_costs, tablefmt="grid"))
        best_improvement=detect_best_improvement(marginal_costs,transportdata)
        if best_improvement is not None:
            # Ici, vous pouvez ajouter du code pour effectuer les modifications nécessaires dans votre réseau de transport
            # Par exemple, augmenter la capacité de l'arête identifiée comme amélioration potentielle
            
            if (err==10):
                print("Aucune amélioration potentielle détectée.")
                a=1
                cost=0
                printingMarginal(marginal_costs)
                for i in range(len(transportdata)):
                    for j in range(len(transportdata[i])):
                        cost+=transportdata[i][j]*datalist[i][j]
                print(cost)
                break 
            else:
                
                print(tabulate(marginal_costs, tablefmt="grid"))
                print("Meilleure amélioration détectée :", best_improvement)
                print("The current cost is")
                postcost=Costculation(transportdata,datalist)
                g.add_edge("P"+str(best_improvement[1]+1),"C"+str(best_improvement[0]+1))
                if precost==postcost:
                    err+=1
                else:
                    err=1
                precost=postcost
                
            #printing(transportdata,transportorder,transportprovision)
        else:
            print("Aucune amélioration potentielle détectée.")
            a=1
            cost=0
            for i in range(len(transportdata)):
                for j in range(len(transportdata[i])):
                    cost+=transportdata[i][j]*datalist[i][j]
            print(cost)
            printing(transportdata,order,provision)
    z=None
    while z!="1" and z!="2":
        print("do you want to continue ? 1 for yes 2 for no")
        z="1"
    if z=="2":
        break
print('le temps est de ' + str(time.time()-timeavant))
print("le temps pour faire la methode est de "+ str(timemethode))