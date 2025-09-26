import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import math
import seaborn as sns
import json
import time
import os

# K-COLORS SHORTEST PATH PROBLEM
# https://networkx.org/documentation/stable/tutorial.html

#----------------------------------------------------------------------------------------------------------------------------------------
# Generazione dei colori del grafo
def rgb_to_hex(rgb):
    '''Conversione da RGB (fra 0 e 1) a formato HEX'''
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    )

def generate_distinct_colors(n):
    return sns.color_palette("husl", n)


#----------------------------------------------------------------------------------------------------------------------------------------
# Creazione del grafo e caricamento dati


def generate_grid_graph(num_nodes, num_edges):
    if num_nodes <= 0 or num_edges <= 0:
        raise ValueError("Il numero di nodi e archi deve essere maggiore di 0.")

    print(f"Generazione di un grid graph con {num_nodes} nodi e {num_edges} archi...")
    
    # Trova dimensioni ottimali per il grid graph
    best_r, best_c = None, None
    min_error = float('inf')

    for r in range(1, int(math.sqrt(num_nodes)) + 2):
        if num_nodes % r == 0:
            c = num_nodes // r
            max_edges = r * (c - 1) + c * (r - 1)
            error = abs(max_edges - num_edges)
            if error < min_error:
                best_r, best_c = r, c
                min_error = error

    if best_r is None or best_c is None:
        raise ValueError("Non è stato possibile trovare una configurazione valida.")
    
    print(f"Configurazione ottimale trovata: {best_r} righe, {best_c} colonne.")
    G = nx.grid_2d_graph(best_r, best_c, periodic=False, create_using=nx.Graph())
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    
    # Verifica del numero di archi attualmente presenti
    current_edges = G.number_of_edges()
    print(f"Grafo generato con {current_edges} archi. Obiettivo: {num_edges} archi.")

    # Controllo del limite massimo di archi possibili
    max_possible_edges = num_nodes * (num_nodes - 1) // 2
    if num_edges > max_possible_edges:
        raise ValueError(
            f"Impossibile creare un grafo con {num_nodes} nodi e {num_edges} archi. "
            f"Il massimo numero di archi possibili è {max_possible_edges}."
        )

    # Gestione degli archi mancanti o in eccesso
    if current_edges < num_edges:
        add_edges = num_edges - current_edges
        print(f"Aggiungendo {add_edges} archi...")
        
        # Aggiungi archi casuali uno alla volta
        while add_edges > 0:
            u = random.randint(0, num_nodes - 1)
            v = random.randint(0, num_nodes - 1)
            if u != v and not G.has_edge(u, v):
                G.add_edge(u, v)
                add_edges -= 1

    elif current_edges > num_edges:
        remove_edges = current_edges - num_edges
        print(f"Rimuovendo {remove_edges} archi...")
        edges_to_remove = random.sample(list(G.edges()), remove_edges)
        G.remove_edges_from(edges_to_remove)

    print(f"Grafo finale con {G.number_of_edges()} archi.")
    return G


def select_source_and_destination(G):
    """
    Seleziona un nodo sorgente (source) e un nodo destinazione (destination) 
    in modo che siano i più distanti possibili all'interno del grafo.
    """
    print("Selezionando source e destination ...")
    
    # Ci si assicura che il grafo sia connesso, altrimenti si usa la componente connessa più grande
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        print(f"Grafo non connesso. Usata la componente più grande con {G.number_of_nodes()} nodi.")

    # Step 1: si sceglie un nodo casuale
    first_node = next(iter(G.nodes))

    # Step 2: si trova il nodo più distante usando BFS
    farthest_node = nx.single_source_shortest_path_length(G, first_node)
    farthest_node = max(farthest_node, key=farthest_node.get)

    # Step 3: si trova il nodo più distante dal nodo precedente
    distances = nx.single_source_shortest_path_length(G, farthest_node)
    destination = max(distances, key=distances.get)

    # Step 4: restituisce i nodi più distanti come source e destination
    source = farthest_node
    return source, destination

#----------------------------------------------------------------------------------------------------------------------------------------
#Algoritmo di riduzione del grafo

def graph_reduction_algorithm(G, source, destination, initial_solution, K):
    try:
        distances_from_source = nx.single_source_dijkstra_path_length(G, source, weight='weight')
        distances_to_destination = nx.single_source_dijkstra_path_length(G, destination, weight='weight')
    except nx.NetworkXNoPath:
        return G.copy()  # Restituisci il grafo originale se non ci sono percorsi validi

    # Calcola la lunghezza della soluzione iniziale
    upper_bound = sum(
        G[initial_solution[i]][initial_solution[i + 1]]['weight'] for i in range(len(initial_solution) - 1) if G.has_edge(initial_solution[i], initial_solution[i + 1])
    )
    print("Upper bound generato: ", upper_bound)

    # Riduzione del grafo
    nodes_to_remove = []
    for n in G.nodes:
        if n not in initial_solution and n != source and n != destination:
            # Calcola il percorso totale passando per il nodo `n`
            length_to_n = distances_from_source.get(n, float('inf'))
            length_from_n = distances_to_destination.get(n, float('inf'))
            total_length = length_to_n + length_from_n

            # Se il percorso è più lungo della soluzione attuale, rimuovi il nodo
            if total_length > upper_bound:
                nodes_to_remove.append(n)

    # Rimuovi i nodi non necessari
    G_reduced = G.copy()
    G_reduced.remove_nodes_from(nodes_to_remove)

    print(f"Nodi rimossi: {nodes_to_remove}")
    print(f"Numero di nodi rimanenti: {G_reduced.number_of_nodes()}")
    print(f"Numero di archi rimanenti: {G_reduced.number_of_edges()}")

    return G_reduced

def save_graph_to_json(G, file_path):

    data = {
        "nodes": [
            {"id": node, "attributes": G.nodes[node]}
            for node in G.nodes
        ],
        "edges": [
            {"source": u, "target": v, "attributes": G[u][v]}
            for u, v in G.edges
        ]
    }

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Grafo salvato con successo in {file_path}")


def load_grid_graph_from_json(file_path):
  
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    G = nx.Graph()
    
    for node in data['nodes']:
        node_id = node['id']
        attributes = node.get('attributes', {})
        G.add_node(node_id, **attributes)
    
    for edge in data['edges']:
        source = edge['source']
        target = edge['target']
        attributes = edge.get('attributes', {})
        G.add_edge(source, target, **attributes)
    
    return G


# Definizione della funzione obiettivo

def objective_function(graph:nx.Graph, solution, K:int):
    cost = 0
    edge_colors = set()

    # Assicurati che 'solution' sia una lista di nodi
    if isinstance(solution, list):
        # Esplora gli archi nel cammino
        for i in range(len(solution) - 1):
            if len(edge_colors) > K:
                # Se il numero di colori è superiore al limite, penalizza
                return float('inf')
            u, v = solution[i], solution[i + 1]
            
            # Controlla che esista un arco tra i nodi u e v
            if graph.has_edge(u, v):
                cost += graph[u][v]['weight']  # Aggiungi il peso dell'arco
                edge_colors.add(graph[u][v]['color'])  # Aggiungi il colore dell'arco
            else:
                # Penalizzazione alta per cammini non validi
                return float('inf')
    else:
        print("ERRORE: la soluzione non è una lista di nodi valida.")
        return float('inf')
        
    return cost

def initialize_solution(G, source, target, K, max_attempts=20, reducing=False):
    best_path = None
    best_cost = float('inf')
    print("REDUCING?", reducing)
    if (reducing==False):
        for _ in range(max_attempts):
            path = generate_initial_path_with_backtracking(G, source, target, K)
            if path is not None:
                cost = calculate_path_cost(G, path)
                if cost < best_cost:
                    best_cost = cost
                    best_path = path           
    else:
        target_distance = dict(nx.single_source_dijkstra_path_length(G, target, weight='weight'))

        for _ in range(max_attempts):
            path = generate_initial_path_with_backtracking_shortest(G, source, target, K, target_distance)
            if path is not None:
                cost = calculate_path_cost(G, path)
                if cost < best_cost:
                    best_cost = cost
                    best_path = path

    return best_path

def generate_initial_path_with_backtracking(G, source, target, K):
    """
    Genera una singola soluzione iniziale in maniera randomica e sfrutta backtracking per esplorare alternative.
    """
    path = [source]
    current_node = source
    visited_colors = set()
    visited_nodes = {source}  # Evita cicli

    while current_node != target:
        # Se esiste un arco diretto verso il target, lo si utilizza, se è vaido
        if G.has_edge(current_node, target) and G[current_node][target]['color'] not in visited_colors:
            path.append(target)
            return path

        # Trova i vicini validi (rispettando il vincolo sui colori)
        neighbors = [
            n for n in G.neighbors(current_node)
            if n not in visited_nodes and
               len(visited_colors | {G[current_node][n]['color']}) <= K
        ]

        if not neighbors:
            # Se non ci sono vicini validi, fa backtracking
            if len(path) > 1:
                path.pop()  
                current_node = path[-1]  # Passa al nodo precedente
                continue
            else:
                return None 

        next_node = random.choice(neighbors)

        edge_color = G[current_node][next_node]['color']
        visited_nodes.add(next_node)
        visited_colors.add(edge_color)
        path.append(next_node)
        current_node = next_node

    return path if current_node == target else None


def generate_initial_path_with_backtracking_shortest(G, source, target, K, target_distance):
    """
    Genera una singola soluzione iniziale in maniera randomica e sfrutta backtracking, 
    ottimizzando la scelta dei vicini usando la distanza più breve dal target.
    """
    path = [source]
    current_node = source
    visited_colors = set()
    visited_nodes = {source}  # Evita cicli

    while current_node != target:

        # Se esiste un arco diretto verso il target e non è stato visitato
        if G.has_edge(current_node, target) and G[current_node][target]['color'] not in visited_colors:
            path.append(target)
            return path

        # Trova i vicini validi (rispettando il vincolo sui colori)
        neighbors = [
            n for n in G.neighbors(current_node)
            if n not in visited_nodes and
               len(visited_colors | {G[current_node][n]['color']}) <= K
        ]

        if not neighbors:
            # Se non ci sono vicini validi, fa backtracking
            if len(path) > 1:
                path.pop()  
                current_node = path[-1]  # Passa al nodo precedente
                continue
            else:
                return None 

        # Ordina i vicini per la distanza stimata dal target
        neighbors.sort(key=lambda n: target_distance.get(n, float('inf')))

        next_node = neighbors[0]  # Prendi il vicino più promettente

        edge_color = G[current_node][next_node]['color']
        visited_nodes.add(next_node)
        visited_colors.add(edge_color)
        path.append(next_node)
        current_node = next_node

    return path if current_node == target else None

def calculate_path_cost(G, path):
    cost = 0
    for i in range(len(path) - 1):
        cost += G[path[i]][path[i + 1]]['weight']
    return cost


def check_if_optimal_solution(G, source, target, K):
    try:
        path = nx.shortest_path(G, source, target, weight='weight')
        if len(set(G[u][v]['color'] for u, v in zip(path, path[1:]))) <= K:
            return path
        else:
            return None
    except nx.NetworkXNoPath:
        return None
    
def replace_edge_with_new_color(path, G, colors_in_path):
    """
    Sostituisce un arco nel percorso corrente con un altro compatibile, mantenendo il vincolo sui colori.
    """
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        current_color = G[u][v]['color']
        
        # Cerca un arco alternativo con colore diverso
        for neighbor in G[u]:
            if neighbor != v:  # Evita l'arco corrente
                edge_data = G[u][neighbor]
                if edge_data['color'] not in colors_in_path:  # Deve essere un colore nuovo
                    # Sostituisci (u, v) con (u, neighbor)
                    new_path = path[:i + 1] + nx.shortest_path(G, neighbor, path[-1], weight='weight')
                    return new_path
    
    return None 


def insert_intermediate_node(path, G, colors_in_path):
    """
    Inserisce un nodo intermedio nel percorso corrente, con un arco di colore nuovo.
    """
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        
        # Prova a inserire un nodo intermedio tra u e v
        for intermediate in G[u]:
            if intermediate != v and G[u][intermediate]['color'] not in colors_in_path:
                new_path = path[:i + 1] + [intermediate] + path[i + 1:]
                return new_path
    return None

def swap_subpath(path, G, colors_in_path):
    """
    Sostituisce un sottopercorso con uno alternativo che utilizza colori compatibili.
    """
    for i in range(len(path) - 2):
        for j in range(i + 2, len(path)):
            # Cerca un nuovo percorso tra path[i] e path[j]
            subpath_start, subpath_end = path[i], path[j]
            try:
                new_subpath = nx.shortest_path(G, subpath_start, subpath_end, weight='weight')
                
                # Verifica i colori del nuovo sottopercorso
                new_colors = {G[new_subpath[k]][new_subpath[k + 1]]['color'] for k in range(len(new_subpath) - 1)}
                if not new_colors.intersection(colors_in_path):
                    # Sostituisci il sottopercorso
                    new_path = path[:i] + new_subpath + path[j + 1:]
                    return new_path
            except nx.NetworkXNoPath:
                continue
    return None 


def generate_neighborhood(current_solution, G, K, tabu_list):
    colors_in_path = {G[current_solution[i]][current_solution[i + 1]]['color'] for i in range(len(current_solution) - 1)}
    neighborhood = []

    new_solution = replace_edge_with_new_color(current_solution, G, colors_in_path)
    if new_solution and new_solution not in tabu_list:
        neighborhood.append(new_solution)

    new_solution = insert_intermediate_node(current_solution, G, colors_in_path)
    if new_solution and new_solution not in tabu_list:
        neighborhood.append(new_solution)

    new_solution = swap_subpath(current_solution, G, colors_in_path)
    if new_solution and new_solution not in tabu_list:
        neighborhood.append(new_solution)

    return neighborhood

#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------
# Define the Tabu Search algorithm

def tabu_search_with_restart(G, source, destination, K, max_iterations, max_restarts=200, initial_tabu_list_size=15, max_no_improvement=20):
    # Inizializzazione della soluzione
    best_solution = initialize_solution(G, source, destination, K)
    actual_tries = 0
    no_improvement_count = 0
    restarts = 0
    tabu_list_size = initial_tabu_list_size
    
    while best_solution is None and actual_tries <= 15:
        print("Reinizializzazione...")
        best_solution = initialize_solution(G, source, destination, K)
        actual_tries += 1

    if best_solution is None:
        print("Impossibile generare una soluzione iniziale!")
        return None

    current_solution = best_solution
    current_cost = objective_function(G, current_solution, K)
    tabu_list = []

    print(f"Soluzione iniziale trovata: {current_solution}, costo: {current_cost}")

    for iteration in range(max_iterations):
        print(f"Iterazione: {iteration}")

        all_neighbors = generate_neighborhood(current_solution, G, K, tabu_list)

        # Filtra i vicini tabu
        all_neighbors = [n for n in all_neighbors if n not in tabu_list]

        best_neighbor = None
        best_neighbor_fitness = float('inf')

        # Esplorazione dei vicini e calcolo della funzione obiettivo
        for neighbor in all_neighbors:
            neighbor_fitness = objective_function(G, neighbor, K)
            
            # Applicazione dell'aspirazione
            if (neighbor_fitness < best_neighbor_fitness or 
                (neighbor_fitness < objective_function(G, best_solution, K) and neighbor not in tabu_list)):
                best_neighbor = neighbor
                best_neighbor_fitness = neighbor_fitness

        if best_neighbor is None:
            print("Nessun vicino valido, termina la ricerca.")
            break

        # Aggiorna la soluzione corrente
        current_solution = best_neighbor
        tabu_list.append(best_neighbor)

        # Gestione dimensione dinamica della lista tabu
        if no_improvement_count > 5:
            tabu_list_size = min(20, tabu_list_size + 1)  # Intensificazione: incrementa la lista tabu
        else:
            tabu_list_size = max(5, tabu_list_size - 1)  # Diversificazione: riduci la lista tabu

        if len(tabu_list) > tabu_list_size:
            tabu_list.pop(0)  # Rimuove la mossa più vecchia dalla lista tabu

        # Controllo del miglioramento
        if best_neighbor_fitness < objective_function(G, best_solution, K):
            print("Miglioramento trovato, reset del counter ...")
            best_solution = best_neighbor
            no_improvement_count = 0  # Reset del contatore
        else:
            print("Miglioramento non trovato ...")
            no_improvement_count += 1

        # Condizione per il restart
        if no_improvement_count >= max_no_improvement:
            if restarts >= max_restarts:
                print("Numero massimo di restart raggiunto.")
                break
            print(f"Restart #{restarts + 1}: Ricerca in stallo.")
            restarts += 1

            # Diversificazione: Genera una nuova soluzione iniziale
            new_solution = initialize_solution(G, source, destination, K)
            if new_solution:
                current_solution = new_solution
                tabu_list = []  # Pulisci la lista tabu per esplorare nuove regioni
                print(f"Nuova soluzione iniziale: {new_solution}")
            else:
                print("Impossibile generare una nuova soluzione iniziale durante il restart.")

    return best_solution

#---------------------------------------------------------------------------------------------------------------

max_iterations = 1000
tabu_list_size = 30

G = nx.Graph()

try:
    load_graph=(input("Caricare un grafo in JSON?(Y/N)"))
    if (load_graph=='Y' or load_graph=='y'):
        json_graph_name=input("Inserire il nome del file JSON (senza estensione): ")
        file_path = json_graph_name + ".json"
        if not os.path.exists(file_path):
            print(f"Errore: Il file {file_path} non esiste nella cartella corrente.")   
        else:
            print("Caricando il file ...")
    if (load_graph=='Y' or load_graph=='y'):
        G= load_grid_graph_from_json(file_path)

        num_nodes=len(G.nodes)
        print(num_nodes)
        for edge in G.edges():
            weight = G[edge[0]][edge[1]]['weight']
            color = G[edge[0]][edge[1]]['color']
            print(f"Arco {edge}: Peso = {weight}, Colore = {color}")
        edge_colors = [G.edges[edge]["color"] for edge in G.edges()]
        print("Grafo caricato con successo:", G)
        print("Numero di nodi:",len(G.nodes))
        print("Numero di archi:",len(G.edges))
        manual_source_and_destination=input("Scegliere manualmente source e destination?(Y/N)")
        if (manual_source_and_destination=='N' or manual_source_and_destination=='n'):
            source, destination = select_source_and_destination(G)
        else:
            try:
                source=int(input("Inserire un numero intero per identificare il numero del nodo sorgente: "))
                destination=int(input("Inserire un numero intero per identificare il numero del nodo destinazione: "))
            except:
                print("Errore durante l'inserimento dell'input")
                exit()

        if (file_path=='A-G1.json'):
            K=195
        elif(file_path=='A-G4.json'):
            K=290
        elif(file_path=='A-G9.json'):
            K=1005
        elif(file_path=='B-G2.json'):
            K=174
        elif(file_path=='B-G6.json'):
            K=480
        elif(file_path=='B-G10.json'):
            K=985
        elif(file_path=='A-R7.json'):
            K=6
        elif(file_path=='A-R9.json'):
            K=5
        elif(file_path=='A-R11.json'):
            K=6
        else:
            K=int(input("Inserire il numero K di colori massimo: "))

        for i in range(10):
            initial_solution=initialize_solution(G, source, destination, K, max_attempts=15, reducing=True)
            if (initial_solution is not None):
                break
        print("Soluzione iniziale generata, riducendo ")

        try:
            G_initial= nx.Graph()
            G_initial = G.copy()

            G = graph_reduction_algorithm(G, source, destination, initial_solution, K)
            percent = 100 - (100 * G.number_of_nodes()/num_nodes)
            print("Number of nodes: ", num_nodes)
            print("New number of nodes:", G.number_of_nodes())
            print("Grafo ridotto del ", percent, "% rispetto all'originale!")
            input("Premere invio per continuare ...")
        except Exception as e:
            print("ERROR:",e )

    elif (load_graph=='N' or load_graph=='n'):
        try:
            num_nodes= int(input("Inserire un numero n per definire il numero di nodi del grafo: "))
            num_edges= int(input("Inserire un numero intero per definire il numero di archi del grafo: "))
            min_distance=int(input("Inserire un numero intero per definire la minima distanza tra due nodi: "))
            max_distance=int(input("Inserire un numero intero per definire la massima distanza tra due nodi: "))
            manual_source_and_destination=input("Scegliere manualmente source e destination?(Y/N): ")
            if (manual_source_and_destination=="Y" or manual_source_and_destination=='y'):
                source=int(input("Inserire un numero intero per identificare il numero del nodo sorgente: "))
                destination=int(input("Inserire un numero intero per identificare il numero del nodo destinazione: "))
                print("SCELTO MANUALMENTE")
            else:
                print("SCELTO AUTOMATICAMENTE")
            number_of_colors=int(input("Inserire il numero di colori da generare:"))
            K=int(input("Inserire il numero K di colori massimo: "))
        except:
            print("Uno dei valori inseriti non è valido, inserire un numero intero!")
            default_params=input("Utilizzare valori di default?(Y/N)")
            if (default_params=='Y' or default_params=='y'):
                num_nodes=100
                num_edges=70
                min_distance=1
                max_distance=1000
                source=0
                destination=num_nodes-1
                K=3
                number_of_colors=8
            else:
                exit()

        colors = [rgb_to_hex(color) for color in generate_distinct_colors(number_of_colors)]
        print("Distinct Colors:", colors)
        graph_type=input("Inserire tipo di grafo (G=GRID, R=RANDOM GRAPH): ")
        if (graph_type!='G' and graph_type!='g'):
            print("Graph type: random")
            density = nx.density(G)
            if (density > 0.5):
                G= nx.dense_gnm_random_graph(n=num_nodes, m=num_edges, seed=1)
            else:
                G = nx.gnm_random_graph(n=num_nodes, m=num_edges, seed=1)
            input("Premere invio per continuare ...")
        else:   
            print("Graph type: grid")
            G = generate_grid_graph(num_nodes, num_edges)

        print("Grafo creato! Selezionando source e destination ...")
        
        source, destination = select_source_and_destination(G)
        input("Premere invio per continuare ...")
        print("SOURCE SELECTED: ",source,"\nDESTINATION SELECTED: ",destination)
        print("Is connected:",nx.is_connected(G))
        print("Riducendo il grafo ...")


        #Assegnamento dei pesi
        for (u, v) in G.edges():
            G[u][v]['weight'] = random.randint(min_distance, max_distance)

        #Assegnamento dei colori
        for edge in G.edges():
            G.edges[edge]["color"] = random.choice(colors)

        # Ottenimento dei colori dei nodi
        edge_colors = [G.edges[edge]["color"] for edge in G.edges()]

        initial_solution=initialize_solution(G, source, destination, K, max_attempts=15, reducing=True)
        print("Soluzione iniziale generata, riducendo ")
        G_initial = G.copy()
        G = graph_reduction_algorithm(G, source, destination, initial_solution, K)
        percent = 100 - (100 * G.number_of_nodes()/num_nodes)
        print("Number of nodes: ", num_nodes)
        print("New number of nodes:", G.number_of_nodes())
        print("Grafo ridotto del ", percent, "% rispetto all'originale!")
        input("Premere invio per continuare ...")        
    else:
        exit()
except Exception as e:
        print("ERRORE,",e)

#----------------------------------------------------------------------------------------------------------------------------------------

edge_labels = nx.get_edge_attributes(G, "weight")
edge_colors = list(nx.get_edge_attributes(G, "color").values())

# Plot del grafo se il numero di nodi è inferiore a 100 
if (num_nodes < 100):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_size=400)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges, width=2,edge_color=edge_colors, alpha=1)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)
    plt.axis("off")
    plt.tight_layout()
    plt.show(block=False)

#------------------------------------------------------------------------------------------------------------------------

if (nx.has_path(G, source, destination)==False):
    print("Nessuna soluzione esistente! Non esiste un percorso tra source e destination!")
    exit()

print("Starting search ...")
start_time=time.time()
best_solution=tabu_search_with_restart(G,source, destination,K,max_iterations=max_iterations, initial_tabu_list_size=tabu_list_size)
end_time=time.time()

if (best_solution==None):
    save_graph_json=input("Nessuna soluzione trovata. Salvare il grafo in JSON? (Y/N)")
    
    if (save_graph_json=='Y' or save_graph_json=='y'):
        graph_json_name=input("Inserire il nome del file JSON da salvare: ")
        save_graph_to_json(G_initial, graph_json_name+".json")
print("--------------------------------------------------------------------------------------------------------------")
print("SOURCE:", source, "DESTINATION:",destination)
if (best_solution is not None): print("BEST SOLUTION:", best_solution)
print("PATH COST:",objective_function(G,best_solution, K))
print("Time elapsed:",round(end_time-start_time,4))
plt.show()


#----------------------------------------------------------------------------------------------------------------------------------------

#Check se la soluzione trovata è effettivamente quella ottima tramite confronto con algoritmo di Dijkstra

optimal_solution=check_if_optimal_solution(G_initial,source,destination,K)
if (optimal_solution is not None):
    print("OPTIMAL SOLUTION (DIJKSTRA BASED):",optimal_solution)

    optimal_cost=objective_function(G,optimal_solution,K)
    print("COST OF OPTIMAL SOLUTION:",optimal_cost)
    best_solution_cost=objective_function(G,best_solution, K)
    print("BEST SOLUTION   :",best_solution, "COST:", best_solution_cost)
    if (best_solution_cost <= optimal_cost):
        print("Migliore soluzione trovata!")
    else:
        print("Migliore soluzione non trovata!")
    
        print("OPTIMAL SOLUTION:",optimal_solution)
else:
    print("Impossibile trovare una soluzione ottima che soddisfi i vincoli con Dijkstra")

if (load_graph=='Y' or load_graph=='y'):
    exit()

try:
    save_graph_json=input("Salvare il grafo in JSON? (Y/N)")
    if (save_graph_json=='Y' or save_graph_json=='y'):
       graph_json_name=input("Inserire il nome del file JSON da salvare: ")
       save_graph_to_json(G_initial, graph_json_name+".json")
except Exception as e:
    print("Errore durante il salvataggio del grafo in JSON:", e)
    exit()


print("------------------------------------------------------------------------------------------------------------")

eval=input("Iniziare fase di evaluation? (Y/N)")
if (eval=='N' or eval=='n'):
    exit()
else:
    print("Iniziando fase di evaluation ...\n")

print("+------------------------------------------------------------------------------------------------------------+")
print("|                                                 Evaluation                                                 |")
print("+------------------------------------------------------------------------------------------------------------+")
num_nodes=[10,50,100,500,1000,2000,5000]
min_distance=1
max_distance=100
source=1
number_of_colors=[5,15,20,30,50,100,100]
K=[3,5,5,7,10,20,50]

total_err=[[] for _ in range(len(num_nodes))]
total_time=[[] for _ in range(len(num_nodes))]

for i in range(len(num_nodes)):
    for j in range(1,20):
        print("ITERAZIONE", j)
        destination=random.randrange(2,num_nodes[i])
        print("Generando un grafo con",num_nodes[i],"nodi...")
        print("Numero di colori",K[i])
        
        colors = [rgb_to_hex(color) for color in generate_distinct_colors(number_of_colors[i])]
        #print("Distinct Colors:", colors)
        G = nx.gnp_random_graph(n=num_nodes[i], p=0.5)
        while not nx.is_connected(G):
            G = nx.gnp_random_graph(n=num_nodes[i], p=0.5)

        print("Is connected:",nx.is_connected(G))
        print("Numero di archi",len(G.edges))

        for (u, v) in G.edges():
            G[u][v]['weight'] = random.randint(min_distance, max_distance)

        for edge in G.edges():
            G.edges[edge]["color"] = random.choice(colors)

        edge_colors = [G.edges[edge]["color"] for edge in G.edges()]

        start_time=time.time()
        sol=tabu_search_with_restart(G,source,destination,K[i],max_iterations=max_iterations,initial_tabu_list_size=tabu_list_size)
        end_time=time.time()
        if (sol):
            print("Soluzione trovata per grafo con ",len(G.nodes),"nodi: ", sol,"\nCosto:", calculate_path_cost(G,sol),"Tempo: ",round(end_time-start_time, 4))
            try:
                actual_cost=calculate_path_cost(G,sol)
                optimal_cost=calculate_path_cost(G,check_if_optimal_solution(G,source, destination,K[i]))
                time_elapsed=round(end_time-start_time,4)
                if(optimal_cost==actual_cost):
                    print("Soluzione esatta!")
                    err=0
                    
                else:
                    err=(abs(actual_cost-optimal_cost))/actual_cost
            except:
                print("Skipping")
                continue

        total_err[i].append(err)
        total_time[i].append(time_elapsed)
    input("Premere invio per continuare ...")

for i in range(len(num_nodes)):
    print("Errore totale per grafo con numero di nodi=",num_nodes[i],"e per k=",K[i],":",total_err[i])
    print("Errore medio:", sum(total_err[i])/50)
    print("Tempo di esecuzione totale per grafo con numero di nodi=",num_nodes[i],"e per k=",K[i],":",total_time[i])
    print("Tempo di esecuzione medio: ",sum(total_time[i])/50)
    print("--------------------------------------------------------------------------------------------------------")