import networkx as nx
from tabulate import tabulate
import copy

# Creates all nodes then creates all edges from a proposal in the form of a matrix.
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

# Read data from file. First line gives the size nm of the matrix. Last column stored in provision. Last line stored in order.
# Every other data added in the created matrix
def readfile(file_number):
    try:
        with open(f"./Problems/{file_number}.txt", 'r') as file:
    
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
        print(f"Le fichier {file_number} n'a pas été trouvé.")
        return None
    
# Display with tabulate by giving data (cost or proposal), order and provision
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

# Same as printing but with penalties added to the sides
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

# Display the result of the potentials per node    
def display_potentials(potentials):
    print("Potentiels par sommet :")
    for node, potential in potentials.items():
        print(f"Sommet {node}: {potential}")
    print()
