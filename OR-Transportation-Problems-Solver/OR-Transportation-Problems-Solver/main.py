from tabulate import tabulate
import copy
import networkx as nx
from random import *
import math
from read_and_display import *
from graph_operation import *
import sys

# Used to make traces
# Redirects output to both file and console
"""class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("./Traces/int3-7_trace_placeholder-bh.txt", "w")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass    

sys.stdout = Logger()"""

while True:
    i=0
    while i==0:
        print("Operation Research Project\nWhat do you want to do?\n\t1-Test a file\n\t2-Generate a random one\n")
        k=int(input())
        if k==1:
            print("Which file do you want to test? (1 to 12)\n")
            a=str(input())
            data=readfile(a)
            if data!=None:
                i=1
        elif k==2:
            print("Enter the size of your square matrix: ")
            size=int(input())
            data=randTable(size)
            i=1
    
    i=0
    while i!="2" and i!="1":
        print("Which method do you want to use?\n\t1-North West\n\t2-Balas\n")
        i=str(input())
    
    (datalist,order,provision)=data
    printing(datalist,order,provision) 
    if i=="1":   
         (transportdata,transportorder,transportprovision)=NorthWest(datalist, order, provision)
    else:
        (transportdata,transportorder,transportprovision)=BalasHammer(datalist,order,provision,datalist)
    a=0
    printing(transportdata,transportorder,transportprovision)
    g=display_graph(transportdata)
    best_improvement=None
    err=0
    precost=Costculation(transportdata,datalist,p=False)
    while a==0 :
        
        test=testcircular(g,p=True)
        if test!=1:
            print("There is a cycle: ", test[0][0], end="")
            for edge in test:
                print(" ==> ", edge[1], end="")
            print("\n")
        if can_reach_all_nodes(g)==False:
            print("The graph is degenerate")
            sub_graphs = list(nx.connected_components(g))
            for graph in sub_graphs:
                print(f"Sub graph: {graph}")
        while test!=1 or can_reach_all_nodes(g)==False:
            
            if test!=1:
                (transportdata,g)=rectifCircular(test,transportdata, g,order,provision,best_improvement)
            if can_reach_all_nodes(g)==False:
                g =testContinuity(g,datalist)
                
            test=testcircular(g)
        potentials = calculate_potentials(g, datalist)
        
        display_potentials(potentials)
        potential_costs = calculate_potential_costs(transportdata, potentials)
        marginal_costs = calculate_marginal_costs(datalist, potential_costs)
        print('The potential costs is:')
        print(tabulate(potential_costs, tablefmt="grid"))
        print('The marginal costs is:')
        print(tabulate(marginal_costs, tablefmt="grid"))
        best_improvement=detect_best_improvement(marginal_costs,transportdata)
        if best_improvement is not None:
            
            if (err==10):
                printingMarginal(marginal_costs)
                print("No improvement detected.")
                a=1
                cost=0
                for i in range(len(transportdata)):
                    for j in range(len(transportdata[i])):
                        cost+=transportdata[i][j]*datalist[i][j]
                print("Total cost: ",cost)
                break 
            else:
                print("Improvement detected :", best_improvement)
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
            print("No improvement detected.")
            a=1
            cost=0
            for i in range(len(transportdata)):
                for j in range(len(transportdata[i])):
                    cost+=transportdata[i][j]*datalist[i][j]
            printing(transportdata, order, provision)
            print("Total cost: ",cost)
    z=None
    while z!="1" and z!="2":
        print("Do you want to continue ?\n\t1-Yes\n\t2-No\n")
        z=str(input())
    if z=="2":
        break
print("Goodbye!")
