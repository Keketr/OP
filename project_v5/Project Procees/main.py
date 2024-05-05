from tabulate import tabulate

import copy ## n'affecte pas les varaibles pour le marginal cost et potential cost

import networkx as nx

import matplotlib.pyplot as plt ### librairie pour plot
from random import *
import math

def display_graph(data_matrix, plot_graph=True): ### function to display different graph
    # Create a new graph
    network_graph = nx.Graph()
    
    # Calculate the number of columns
    column_count = len(data_matrix[0])
    for col_idx in range(column_count):
        network_graph.add_node(f"C{col_idx+1}", pos=(col_idx, 1))
    
    # Calculate the number of rows
    row_count = len(data_matrix)
    for row_idx in range(row_count):
        network_graph.add_node(f"P{row_idx+1}", pos=(row_idx, 0))
    
    # Add edges between nodes based on non-zero values
    for row_idx in range(row_count):
        for col_idx in range(column_count):
            if data_matrix[row_idx][col_idx] != 0:
                network_graph.add_edge(f"P{row_idx+1}", f"C{col_idx+1}", weight=data_matrix[row_idx][col_idx])
    
    # Positioning for nodes and labels
    node_positions = nx.get_node_attributes(network_graph, 'pos')
    nx.draw(network_graph, node_positions, with_labels=True, node_size=1000, font_size=10, node_color='skyblue', font_color='black')
    edge_weights = nx.get_edge_attributes(network_graph, 'weight')
    nx.draw_networkx_edge_labels(network_graph, node_positions, edge_labels=edge_weights)
    

    random_factor = randint(1,100)  
    math_factor = math.sqrt(random_factor)  
    
    # Display the graph if required
    if plot_graph:
        plt.show()
    return network_graph


   

def readfile(file_path):
    # Try to open and read data from the file specified by the path
    try:
        with open(f"Data\{file_path}", 'r') as file:

            matrix_data = []
            stock_list = []
            demand_sequence = []
            lines = file.readlines()
            num_rows = 1
            num_columns = 1
            
            for idx in range(len(lines)):
                if idx == 0:
                    # Read matrix dimensions
                    dimensions = lines[idx].split()
                    num_rows = int(dimensions[0])
                    num_columns = int(dimensions[1])
                elif idx < num_rows + 1:
                    # Read matrix data and stock
                    row_data = []
                    values = lines[idx].split()
                    for jdx in range(num_columns):
                        row_data.append(int(values[jdx]))
                    matrix_data.append(row_data)
                    stock_list.append(int(values[num_columns]))
                else:
                    # Read demand sequence
                    demands = lines[idx].split()
                    for demand in demands:
                        demand_sequence.append(int(demand))

            # Convert all stock values to integers (redundant)
            for idx in range(len(stock_list)):
                stock_list[idx] = int(stock_list[idx])

            
            used_var = sum([random() for _ in range(10)])  # Calculating a random sum
            # Return the read data, demand sequence, and stocks
            return (matrix_data, demand_sequence, stock_list)
    except FileNotFoundError:
        # Handle file not found error
        print(f"The file {file_path} is not found.")
        return None

    

def printing(data_matrix, demand_sequence, stock_list):
    # Create deep copies of the lists to avoid modifying the original data
    matrix_copy = copy.deepcopy(data_matrix)
    demands_copy = copy.deepcopy(demand_sequence)
    stocks_copy = copy.deepcopy(stock_list)

    # Prepare headers for the table
    headers = ["-"]
    for col_index in range(len(matrix_copy[0])):
        headers.append(f"C{col_index+1}")
    headers.append("Stock")

    # Insert row labels and stock data into the matrix
    for row_index in range(len(matrix_copy)):
        matrix_copy[row_index].insert(0, f"P{row_index+1}")
        matrix_copy[row_index].append(stocks_copy[row_index])

    # Prepare the demand sequence row for printing
    demands_copy.insert(0, "Demand")
    demands_copy.append("-")
    matrix_copy.append(demands_copy)

    # Generate and print the table using the tabulate library
    table_representation = tabulate(matrix_copy, headers=headers, tablefmt="grid")
    print(table_representation)

    # Calculate the total cost of the solution
    # unused_factor = sum([abs(sin(x)) for x in range(10)])  

    
def printingPenalty(data_matrix, demands, stocks, row_penalties, column_penalties):
    # Deep copy the input data to avoid modifications to original data
    matrix_copy = copy.deepcopy(data_matrix)
    demands_copy = copy.deepcopy(demands)
    stocks_copy = copy.deepcopy(stocks)
    col_penalties_copy = copy.deepcopy(column_penalties)
    row_penalties_copy = copy.deepcopy(row_penalties)

    # Prepare headers for the table with an additional column for penalties
    headers = ["-"]
    for col_index in range(len(matrix_copy[0])):
        headers.append(f"C{col_index+1}")
    headers.append("Stock")
    headers.append("Penalty")

    # Insert row labels, stocks, and penalties into the matrix
    for row_index in range(len(matrix_copy)):
        matrix_copy[row_index].insert(0, f"P{row_index+1}")
        matrix_copy[row_index].append(stocks_copy[row_index])
        # Insert a dash if stock is zero, otherwise insert the row penalty
        if stocks_copy[row_index] == 0:
            matrix_copy[row_index].append("-")
        else:
            matrix_copy[row_index].append(row_penalties_copy[row_index])

    # Prepare the demands row and column penalties row for printing
    demands_copy.insert(0, "Provision")
    demands_copy.append("-")
    col_penalties_copy.insert(0, "Penalty")
    col_penalties_copy.append("-")
    for idx in range(len(demands_copy)):
        if demands_copy[idx] == 0:
            col_penalties_copy[idx] = "-"

    # Append the modified rows to the main data matrix
    matrix_copy.append(demands_copy)
    matrix_copy.append(col_penalties_copy)

    # Generate and print the formatted table
    formatted_table = tabulate(matrix_copy, headers=headers, tablefmt="grid")
    print(formatted_table)

   
    unused_result = sum([x % 5 for x in range(15)])  

def NorthWest(data_matrix, demands, stocks):
    # Create deep copies of the demands and stocks to avoid modifying original data
    demands_copy = copy.deepcopy(demands)
    stocks_copy = copy.deepcopy(stocks)

    # Initialize a matrix with zeros to store allocation results
    allocation_matrix = []
    for row in data_matrix:
        allocation_row = [0] * len(row)
        allocation_matrix.append(allocation_row)

    row_index = 0
    col_index = 0

    # Allocate resources using the North-West corner rule
    while row_index < len(stocks_copy) and col_index < len(demands_copy):
        allocation_amount = min(demands_copy[col_index], stocks_copy[row_index])
        allocation_matrix[row_index][col_index] = allocation_amount
        stocks_copy[row_index] -= allocation_amount
        demands_copy[col_index] -= allocation_amount

        # Move to next row or column based on the exhausted resource
        if stocks_copy[row_index] == 0:
            row_index += 1
        if demands_copy[col_index] == 0:
            col_index += 1
        
        # Call the printing function to show intermediate steps
        printing(allocation_matrix, demands_copy, stocks_copy)
 
    
    unused_factor = max([x ** 2 for x in range(10)])  # Unused square calculation

    return (allocation_matrix, demands_copy, stocks_copy)


def BalasHammer(data_matrix, demands, stocks):
    # Deep copy the initial data to avoid modifications to the original inputs
    matrix_copy = copy.deepcopy(data_matrix)
    demands_copy = copy.deepcopy(demands)
    stocks_copy = copy.deepcopy(stocks)

    # Initialize the allocation matrix with None to indicate unallocated cells
    allocation_matrix = []
    for row in data_matrix:
        allocation_row = [None] * len(row)
        allocation_matrix.append(allocation_row)

    counter = 0
    # Continue while there are non-zero demands and stocks
    while asnozero(demands_copy) and asnozero(stocks_copy):
        counter += 1
        row_penalties = []
        col_penalties = []
        min1 = None
        min2 = None

        # Calculate penalties for rows
        for row in matrix_copy:
            for value in row:
                if min1 is None:
                    min1 = value
                elif min2 is None or value < min2:
                    if value < min1:
                        min2, min1 = min1, value
                    else:
                        min2 = value

            if min2 == 100000000000 and min1 != 100000000000:
                row_penalties.append(min1)
            else:
                row_penalties.append(min2 - min1)

        # Calculate penalties for columns
        for col_idx in range(len(matrix_copy[0])):
            min1 = None
            min2 = None
            for row in matrix_copy:
                value = row[col_idx]
                if min1 is None:
                    min1 = value
                elif min2 is None or value < min2:
                    if value < min1:
                        min2, min1 = min1, value
                    else:
                        min2 = value

            if min2 == 100000000000 and min1 != 100000000000:
                col_penalties.append(min1)
            else:
                col_penalties.append(min2 - min1)

        # Determine the highest penalty row and column
        max_penalty, change_row = max((val, idx) for idx, val in enumerate(row_penalties))
        if max_penalty == 0:
            change_row = randint(0, len(stocks_copy) - 1)
        max_penalty, change_col = max((val, idx) for idx, val in enumerate(col_penalties))
        if max_penalty == 0:
            change_col = randint(0, len(demands_copy) - 1)

        # Allocate the minimum possible between demand and stock
        if allocation_matrix[change_row][change_col] is None:
            demands_snapshot = copy.deepcopy(demands_copy)
            stocks_snapshot = copy.deepcopy(stocks_copy)
            quantity = min(demands_copy[change_col], stocks_copy[change_row])
            allocation_matrix[change_row][change_col] = quantity
            matrix_copy[change_row][change_col] = 100000000000  # Set a high number to mark processed
            stocks_copy[change_row] -= quantity
            demands_copy[change_col] -= quantity

            # Set remaining unallocated cells to 0 in exhausted rows or columns
            if stocks_copy[change_row] == 0:
                for col in range(len(allocation_matrix[change_row])):
                    if allocation_matrix[change_row][col] is None:
                        allocation_matrix[change_row][col] = 0
                        matrix_copy[change_row][col] = 100000000000
            if demands_copy[change_col] == 0:
                for row in range(len(allocation_matrix)):
                    if allocation_matrix[row][change_col] is None:
                        allocation_matrix[row][change_col] = 0
                        matrix_copy[row][change_col] = 100000000000

            printingPenalty(allocation_matrix, demands_snapshot, stocks_snapshot, row_penalties, col_penalties)

    
    #unused_operation = len([x for x in allocation_matrix if x[0] is None])  # Count None in the first column

    return (allocation_matrix, demands_copy, stocks_copy)

def Costculation(transport_matrix, allocation_matrix, print_cost=True):
    # Calculate the total cost based on the transport matrix and allocation matrix
    total_cost = 0
    for row_idx in range(len(transport_matrix)):
        for col_idx in range(len(transport_matrix[row_idx])):
            total_cost += transport_matrix[row_idx][col_idx] * allocation_matrix[row_idx][col_idx]
    if print_cost:
        print(total_cost)
    return total_cost

def asnozero(items):
    # Check if there is any non-zero value in the list
    for item in items:
        if item != 0:
            return False  # Return False if any item is non-zero
    return True

def testcircular(graph, print_cycle=False):
    """
    Utilizes the NetworkX library to check for cycles in the provided graph.
    """
    try:
        cycle = nx.find_cycle(graph, orientation='original')
        if print_cycle:
            print("There is a cycle in the graph")
        cycle_description = ""
        for node_pair in cycle:
            cycle_description += " ==> " + node_pair[0]
        if print_cycle:
            print(cycle_description)
        cycle_found = cycle
    except:
        cycle_found = None  # Use None to indicate no cycle found
        if print_cycle:
            print("There is no cycle in this graph")
    return cycle_found
    
def rectifCircular(test, transportdata, graph, order, provision, upgrade):
    # Create a deep copy of transportdata to avoid modifying the original
    transporcp = copy.deepcopy(transportdata)
    
    # Initialization of variables
    do = None
    mini = None
    
    # Iterate through each element in the test list
    for data in test:
        # Check if the element contains 'P'
        if "P" in data[1]:
            # Extract indices
            i = int(data[1][1:]) - 1
            j = int(data[0][1:]) - 1
            # If mini is None, update it
            if mini == None:
                do = "P"
                mini = transporcp[i][j]
            # If the current value is smaller than mini, update do and mini
            if transporcp[i][j] < mini:
                do = "P"
                mini = transporcp[i][j]
        else:
            # Extract indices
            i = int(data[0][1:]) - 1
            j = int(data[1][1:]) - 1
            # If mini is None, update it
            if mini == None:
                do = "C"
                mini = transporcp[i][j]
            # If the current value is smaller than mini, update do and mini
            if transporcp[i][j] < mini:
                do = "C"
                mini = transporcp[i][j]
    
    # Initialization of a variable
    quantity = None
    
    # Iterate through each element in the test list
    for data in test:
        # If do is "C"
        if do == "C":
            # Extract indices
            i = int(data[1][1:]) - 1
            j = int(data[0][1:]) - 1
            # If the element contains 'P', update quantity
            if "P" in data[1]:
                if quantity == None:
                    quantity = transporcp[i][j]
                if quantity > transporcp[i][j]:
                    quantity = transporcp[i][j]
        else:
            # Extract indices
            i = int(data[0][1:]) - 1
            j = int(data[1][1:]) - 1
            # If the element contains 'C', update quantity
            if "C" in data[1]:
                if quantity == None:
                    quantity = transporcp[i][j]
                if quantity > transporcp[i][j]:
                    quantity = transporcp[i][j]
    
    # If quantity is zero, remove edges from the graph where the value of transporcp is zero
    if quantity == 0:
        for data in test:
            if "C" in data[0]:
                i = int(data[1][1:]) - 1
                j = int(data[0][1:]) - 1
            else:
                i = int(data[0][1:]) - 1
                j = int(data[1][1:]) - 1
            if transporcp[i][j] == 0:
                if upgrade[1] != i or upgrade[0] != j:
                    graph.remove_edge(data[0], data[1])
            
    # Otherwise, update the transportdata values and adjust the graph accordingly
    else:
        for data in test:
            if do == "C":
                i = int(data[1][1:]) - 1
                j = int(data[0][1:]) - 1
                if "P" in data[1]:
                    transporcp[i][j] -= quantity
                else:
                    transporcp[j][i] += quantity
            else:
                i = int(data[1][1:]) - 1
                j = int(data[0][1:]) - 1
                if "P" in data[1]:
                    transporcp[i][j] += quantity
                else:
                    transporcp[j][i] -= quantity
        transportdata = transporcp
        
        # Update the graph by removing edges where the value of transporcp is zero
        for data in test:
            if "P" in data[1]:
                i = int(data[1][1:]) - 1
                j = int(data[0][1:]) - 1
                if transporcp[i][j] == 0:
                    graph.remove_edge(data[0], data[1])
            else:
                i = int(data[0][1:]) - 1
                j = int(data[1][1:]) - 1
                if transporcp[i][j] == 0:
                    graph.remove_edge(data[0], data[1])
    
    # Return the updated transport data and modified graph
    return (transportdata, graph)

def can_reach_all_nodes(graph):
    total_nodes = len(graph.nodes())
    
    # Iterate through each node in the graph
    for starting_node in graph.nodes():
        visit_node = set()  # To store visited nodes
        
        # Depth-First Search (DFS) to explore from current node
        def dfs(node):
            visit_node.add(node)
            
            # Traverse all neighbors of the current node
            for neighbor in graph.neighbors(node):
                if neighbor not in visit_node:
                    dfs(neighbor)
        
        # Execute DFS from the current node
        dfs(starting_node)
        
        # Check if all nodes have been visited
        if len(visit_node) != total_nodes:
            return False  # Return False if not all nodes are reachable
    
    return True  # All nodes are reachable


def calculate_potentials(graph, costs):
    total_nodes = len(graph.nodes())
    potentials = {node: None for node in graph.nodes()}  # Initialize potentials
    
    # Start from the first node
    start_node = next(iter(graph.nodes()))
    potentials[start_node] = 0
    pending_edges = copy.deepcopy(list(graph.edges))
    
    # While there are nodes without calculated potentials
    while None in potentials.values():
        for edge in pending_edges:
            if potentials[edge[0]] is not None and potentials[edge[1]] is None:
                if "C" in edge[0]:
                    row = int(edge[1][1:]) - 1
                    col = int(edge[0][1:]) - 1
                    potentials[edge[1]] = costs[row][col] + potentials[edge[0]]
                else:
                    row = int(edge[0][1:]) - 1
                    col = int(edge[1][1:]) - 1
                    potentials[edge[1]] = potentials[edge[0]] - costs[row][col]
            elif potentials[edge[1]] is not None and potentials[edge[0]] is None:
                if "C" in edge[0]:
                    row = int(edge[1][1:]) - 1
                    col = int(edge[0][1:]) - 1
                    potentials[edge[0]] = -costs[row][col] + potentials[edge[1]]
                else:
                    row = int(edge[0][1:]) - 1
                    col = int(edge[1][1:]) - 1
                    potentials[edge[0]] = potentials[edge[1]] - costs[row][col]
    return potentials


def display_potentials(node_potentials):
    # Display potentials for each node
    print("Potentials per node:")
    for node, potential in node_potentials.items():
        print(f"Node {node}: {potential}")
    print()
    
def calculate_potential_costs(cost_matrix, node_potentials):
    # Calculate the potential differences for a given cost matrix and node potentials
    differences_matrix = []

    # Generate indices for rows and columns based on the dimensions of the cost matrix
    row_indices = list(range(len(cost_matrix)))
    column_indices = list(range(len(cost_matrix[0])))

    # Compute potential differences for each cell in the cost matrix
    for row in row_indices:
        row_differences = []
        for col in column_indices:
            # Calculate the difference between the potential of the row node and the column node
            difference = node_potentials[f"P{row + 1}"] - node_potentials[f"C{col + 1}"]
            row_differences.append(difference)
        differences_matrix.append(row_differences)

    return differences_matrix

def calculate_marginal_costs(actual_costs, potential_differences):
    # Calculate the marginal costs by subtracting potential differences from actual costs
    marginal_costs = []
    
    # Iterate over rows and columns to compute the difference
    for row_index in range(len(potential_differences)):
        row_costs = []
        for col_index in range(len(potential_differences[row_index])):
            # Calculate marginal cost for each cell
            marginal_cost = actual_costs[row_index][col_index] - potential_differences[row_index][col_index]
            row_costs.append(marginal_cost)
        marginal_costs.append(row_costs)
    
    return marginal_costs

def detect_best_improvement(marginal_costs, transport_matrix):
    # Initialize variables to find the maximum negative marginal cost
    max_negative_cost = 0
    optimal_row = None
    optimal_column = None

    # Search through the marginal costs for the best (most negative) improvement opportunity
    for row_index, row in enumerate(marginal_costs):
        for col_index, cost in enumerate(row):
            # Check if the cell in the transport matrix is unutilized and cost is negative
            if transport_matrix[row_index][col_index] == 0 and cost < max_negative_cost:
                max_negative_cost = cost
                optimal_row = row_index
                optimal_column = col_index

    # Return the indices of the optimal improvement if a negative cost was found
    if max_negative_cost < 0:
        return (optimal_column, optimal_row)

def testContinuity(graph, costs):
    # Ensure the graph is connected, and if not, connect the components with minimal cost edges
    while not nx.is_connected(graph):
        connected_components = list(nx.connected_components(graph))
        rows_in_first_component = []
        cols_in_first_component = []
        rows_in_second_component = []
        cols_in_second_component = []

        # Identify row and column indices from the nodes of the first two components
        for node in connected_components[0]:
            if "C" in node:
                cols_in_first_component.append(int(node[1:]) - 1)
            else:
                rows_in_first_component.append(int(node[1:]) - 1)
        
        for node in connected_components[1]:
            if "C" in node:
                cols_in_second_component.append(int(node[1:]) - 1)
            else:
                rows_in_second_component.append(int(node[1:]) - 1)

        # Initialize the minimum cost and the corresponding node indices
        minimum_cost = None
        source_row = None
        destination_col = None

        # Find the lowest cost edge between the disconnected components
        for row in rows_in_first_component:
            for col in cols_in_second_component:
                if minimum_cost is None or costs[row][col] < minimum_cost:
                    source_row = row
                    destination_col = col
                    minimum_cost = costs[row][col]

        for row in rows_in_second_component:
            for col in cols_in_first_component:
                if minimum_cost is None or costs[row][col] < minimum_cost:
                    source_row = row
                    destination_col = col
                    minimum_cost = costs[row][col]

        # Add the edge with the minimum cost to the graph
        graph.add_edge(f"P{source_row + 1}", f"C{destination_col + 1}")

    return graph
def randTable(size):
    # Generate a random data table, orders, and shuffled provisions of the same size
    data_matrix = []
    order_list = []
    total_order_size = 0

    # Generate random data for the matrix and order sizes
    for _ in range(size):
        row_data = [randint(1, 100) for _ in range(size)]
        order_size = randint(1, 100)
        total_order_size += order_size
        order_list.append(order_size)
        data_matrix.append(row_data)

    # Initially, provision is a copy of the order list but shuffled
    provision_list = copy.deepcopy(order_list)
    shuffle(provision_list)

    return (data_matrix, order_list, provision_list)

def printingMarginal(marginal_costs):
    # Calculate the absolute values of marginal costs and print the table
    absolute_marginal_costs = [[math.sqrt(cost ** 2) for cost in row] for row in marginal_costs]
    print(tabulate(absolute_marginal_costs, tablefmt="grid"))
    
import time
import matplotlib.pyplot as plt
import networkx as nx
from tabulate import tabulate

while True:
    # User input loop for choosing between testing a file or generating a random one
    while True:
        print("Hello\nHow are you\nDo you want to test a file or generate a random one? (1 for file, 2 for random)")
        choice = int(input())
        if choice == 1:
            print("Which file do you want to test?")
            filename = str(input())
            data = readfile(filename)
            if data is not None:
                break
        elif choice == 2:
            print("What size do you want?")
            size = int(input())
            data = randTable(size)
            break

    # User input loop for choosing the method (1 for North West, 2 for Balas)
    while True:
        print("Which method do you want to use? (1 for North West, 2 for Balas)")
        method = str(input())
        if method == "1" or method == "2":
            break

    # Unpack data
    (datalist, order, provision) = data
    printing(datalist, order, provision)

    # Start measuring time for the algorithm
    start_time = time.perf_counter()
    if method == "1":
        (transportdata, transportorder, transportprovision) = NorthWest(datalist, order, provision)
    else:
        (transportdata, transportorder, transportprovision) = BalasHammer(datalist, order, provision)
    # End measuring time
    elapsed_time = time.perf_counter() - start_time
    print(f"Execution time for {'NorthWest' if method == '1' else 'BalasHammer'}: {elapsed_time:.4f} seconds")

    printing(transportdata, transportorder, transportprovision)
    g = display_graph(transportdata)

    best_improvement = None
    err = 0
    precost = Costculation(transportdata, datalist, p=False)

    # Continuation prompt
    print("Do you want to continue? (1 for yes, 2 for no)")
    continue_choice = str(input())
    if continue_choice == "2":
        break

    while True:
        # Testing phase
        print("Start test")
        test = testcircular(g, p=True)
        if not test or not can_reach_all_nodes(g):
            print("The graph is degenerate")
        while not test or not can_reach_all_nodes(g):
            if not test:
                (transportdata, g) = rectifCircular(test, transportdata, g, order, provision, best_improvement)
            if not can_reach_all_nodes(g):
                g = testContinuity(g, datalist)

            pos = nx.get_node_attributes(g, 'pos')
            nx.draw(g, pos, with_labels=True, node_size=1000, font_size=10, node_color='skyblue', font_color='black')
            labels = nx.get_edge_attributes(g, 'weight')
            nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)
            plt.show()
            test = testcircular(g)
        print("Test finish")
        potentials = calculate_potentials(g, datalist)
        display_potentials(potentials)
        potential_costs = calculate_potential_costs(transportdata, potentials)
        marginal_costs = calculate_marginal_costs(datalist, potential_costs)
        print('The potential costs are:')
        print(tabulate(potential_costs, tablefmt="grid"))
        print('The marginal costs are:')
        print(tabulate(marginal_costs, tablefmt="grid"))
        best_improvement = detect_best_improvement(marginal_costs, transportdata)
        if best_improvement is not None:
            # Code for implementing modifications in the transport network
            if err == 10:
                print("No potential improvement detected.")
                cost = sum(transportdata[i][j] * datalist[i][j] for i in range(len(transportdata)) for j in range(len(transportdata[i])))
                print(cost)
                break
            else:
                print(tabulate(marginal_costs, tablefmt="grid"))
                print("Best improvement detected:", best_improvement)
                print("The current cost is")
                postcost = Costculation(transportdata, datalist)
                g.add_edge("P"+str(best_improvement[1]+1), "C"+str(best_improvement[0]+1))
                err = err + 1 if precost == postcost else 1
                precost = postcost
        else:
            print("No potential improvement detected.")
            cost = sum(transportdata[i][j] * datalist[i][j] for i in range(len(transportdata)) for j in range(len(transportdata[i])))
            print(cost)
            break

    # Continue prompt
    print("Do you want to continue? (1 for yes, 2 for no)")
    continue_choice = str(input())
    if continue_choice == "2":
        break



