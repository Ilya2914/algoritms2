import numpy as np
import math
import random
import time
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

current_instance = None
best_path = None
best_cost = None
fig = None
ax = None
canvas = None


root = None

problem_var = None
algo_var = None
iter_var = None
ants_var = None
t0_var = None
sa_alpha_var = None
vfsa_c_var = None
vfsa_jump_var = None
aco_alpha_var = None
aco_beta_var = None
aco_q_var = None
aco_rho_var = None

results_table = None

sa_params_frame = None
sa_std_params_frame = None
sa_mod_params_frame = None
aco_params_frame = None

def create_tsp_instance(name, nodes, edge_list=None, coords=None):
    if coords is None:
        coords = {}
        
    instance = {
        'name': name,
        'n': nodes,
        'coords': coords
    }
    
    dist_matrix = np.full((nodes, nodes), float('inf'))

    np.fill_diagonal(dist_matrix, 0)
    
    if edge_list:
        for u, v, w in edge_list:
            u_idx, v_idx = u - 1, v - 1
            dist_matrix[u_idx][v_idx] = w
            dist_matrix[v_idx][u_idx] = w 
            
    instance['dist_matrix'] = dist_matrix
    return instance

def get_path_cost(instance, path):
    cost = 0
    matrix = instance['dist_matrix']
    n = len(path)
    for i in range(n):
        u, v = path[i], path[(i + 1) % n]
        cost += matrix[u][v]
    return cost

def read_stp(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    name = "Неизвестно"
    nodes = 0
    edges = []
    coords = {}
    
    section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith("Name"):
            try:
                name = line.split('"')[1]
            except IndexError:
                name = line.split()[1]
            
        if line.startswith("Section Graph"):
            section = "Graph"
            continue
        elif line.startswith("Section Coordinates"):
            section = "Coordinates"
            continue
            
        if line.startswith("End"):
            section = None
            
        if section == "Graph":
            parts = line.split()
            if parts[0] == "Nodes":
                nodes = int(parts[1])
            elif parts[0] == "E":
                u, v, w = int(parts[1]), int(parts[2]), int(parts[3])
                edges.append((u, v, w))
        elif section == "Coordinates":
            parts = line.split()
            if parts[0] == "DD":
                node_id = int(parts[1])
                x, y = float(parts[2]), float(parts[3])
                coords[node_id - 1] = (x, y) # 0-идкексация
                
    return create_tsp_instance(name, nodes, edge_list=edges, coords=coords)

# 1 Модифицированный отжиг

def solve_vfsa(instance, max_iter=10000, t0=5000.0, vfsa_c=0.5, long_jump_prob=0.0):

    n = instance['n']
    # Жадная инициализация для ускорения сходимости на больших графах
    unvisited = set(range(1, n))
    current_path = [0]
    
    current_node = 0
    while unvisited:
        best_dist = float('inf')
        best_node = -1
        for node in unvisited:
            d = instance['dist_matrix'][current_node][node]
            if d < best_dist:
                best_dist = d
                best_node = node
        
        current_path.append(best_node)
        unvisited.remove(best_node)
        current_node = best_node

    current_cost = get_path_cost(instance, current_path)
    
    local_best_path = list(current_path)
    local_best_cost = current_cost
    
    for k in range(1, max_iter + 1):
        T = t0 / (1 + vfsa_c * k)
        
        if T < 1e-6:
            break
        u = random.random()
        if u < long_jump_prob:
            i, j = random.sample(range(n), 2)
            candidate_path = list(current_path)
            candidate_path[i], candidate_path[j] = candidate_path[j], candidate_path[i]
        else: # 80% шанс на локальное улучшение
            i, j = sorted(random.sample(range(n), 2))
            candidate_path = current_path[:i] + current_path[i:j+1][::-1] + current_path[j+1:]
            
        candidate_cost = get_path_cost(instance, candidate_path)
        
        delta = candidate_cost - current_cost
        if delta < 0:
            accept = True
        else:
            try:
                p = math.exp(-delta / T)
            except OverflowError:
                p = 0.0
            accept = random.random() < p
            
        if accept:
            current_path = candidate_path
            current_cost = candidate_cost
            if current_cost < local_best_cost:
                local_best_cost = current_cost
                local_best_path = list(current_path)
        
    return local_best_path, local_best_cost, []

#2 Стандартный алгоритм имитации отжига
def solve_sa_standard(instance, max_iter=10000, t0=5000.0, cooling_alpha=0.99):
    n = instance['n']
    # Жадная инициализация для ускорения сходимости на больших графах
    unvisited = set(range(1, n))
    current_path = [0]
    
    current_node = 0
    while unvisited:
        best_dist = float('inf')
        best_node = -1
        for node in unvisited:
            d = instance['dist_matrix'][current_node][node]
            if d < best_dist:
                best_dist = d
                best_node = node
        
        current_path.append(best_node)
        unvisited.remove(best_node)
        current_node = best_node

    current_cost = get_path_cost(instance, current_path)
    
    local_best_path = list(current_path)
    local_best_cost = current_cost
    
    T = t0
    alpha = cooling_alpha 
    
    for k in range(1, max_iter + 1):
        if T < 1e-6:
            break
        i, j = sorted(random.sample(range(n), 2))
        candidate_path = current_path[:i] + current_path[i:j+1][::-1] + current_path[j+1:]
        
        candidate_cost = get_path_cost(instance, candidate_path)
        
        delta = candidate_cost - current_cost
        if delta < 0:
            accept = True
        else:
            try:
                p = math.exp(-delta / T)
            except OverflowError:
                p = 0.0
            accept = random.random() < p
            
        if accept:
            current_path = candidate_path
            current_cost = candidate_cost
            if current_cost < local_best_cost:
                local_best_cost = current_cost
                local_best_path = list(current_path)
    
        T *= alpha
        
    return local_best_path, local_best_cost, []

# 3 Муравьиные алгоритм с шаблонами
def solve_aco_modified(instance, n_ants=20, n_iter=100, alpha=5.0, beta=10.0, q=1.0, rho=0.3):
    n = instance['n']
    dist_matrix = instance['dist_matrix']
    k_candidates = 20
    candidates = []
    for i in range(n):
        row = dist_matrix[i]
        sorted_indices = np.argsort(row)
        candidates.append(sorted_indices[1:k_candidates+1])

    pheromone = np.ones((n, n)) * 0.1
    local_best_path = None
    local_best_cost = float('inf')
    
    for it in range(n_iter):
        paths = []
        costs = []
        
        for ant in range(n_ants):
            current = random.randint(0, n - 1)
            visited = {current}
            path = [current]
            
            while len(path) < n:
                probs = []
                possible_next = []
                
                candidate_moves = [noc for noc in candidates[current] if noc not in visited]
                
                if candidate_moves:
                    selection_pool = candidate_moves
                else:
                    selection_pool = [node for node in range(n) if node not in visited]
                
                if not selection_pool:
                     break
                
                heuristics = []
                
                for node in selection_pool:
                    dist = dist_matrix[current][node]
                    eta = 1.0 / (dist + 1e-6) 
                    tau = pheromone[current][node]
                    v = (tau ** alpha) * (eta ** beta)
                    heuristics.append(v)
                    possible_next.append(node)
                    
                heuristics = np.array(heuristics)
                total = np.sum(heuristics)
                
                if total == 0:
                    probs = np.ones(len(heuristics)) / len(heuristics)
                else:
                    probs = heuristics / total
                    
                next_node = np.random.choice(possible_next, p=probs)
                path.append(next_node)
                visited.add(next_node)
                current = next_node
                
            cost = get_path_cost(instance, path)
            paths.append(path)
            costs.append(cost)
            
            if cost < local_best_cost:
                local_best_cost = cost
                local_best_path = path
        
        pheromone *= (1 - rho)
        
        iter_best_idx = np.argmin(costs)
        iter_best_path = paths[iter_best_idx]
        iter_best_cost = costs[iter_best_idx]
        
        if iter_best_cost > 0:
            delta_tau = q / iter_best_cost
            for i in range(n):
                u, v = iter_best_path[i], iter_best_path[(i + 1) % n]
                pheromone[u][v] += delta_tau
                pheromone[v][u] += delta_tau
            
    return local_best_path, local_best_cost, []

# 4 Стандартный Муравьиные алгоритм
def solve_aco_standard(instance, n_ants=20, n_iter=100, alpha=5.0, beta=10.0, q=1.0, rho=0.3):
    n = instance['n']
    dist_matrix = instance['dist_matrix']

    pheromone = np.ones((n, n)) * 0.1
    local_best_path = None
    local_best_cost = float('inf')
    
    for it in range(n_iter):
        paths = []
        costs = []
        
        for ant in range(n_ants):
            current = random.randint(0, n - 1)
            visited = {current}
            path = [current]
            
            while len(path) < n:
                selection_pool = [node for node in range(n) if node not in visited]
                
                if not selection_pool: break
                
                possible_next = []
                heuristics = []
                
                for node in selection_pool:
                    dist = dist_matrix[current][node]
                    eta = 1.0 / (dist + 1e-6)
                    tau = pheromone[current][node]
                    v = (tau ** alpha) * (eta ** beta)
                    heuristics.append(v)
                    possible_next.append(node)
                    
                heuristics = np.array(heuristics)
                total = np.sum(heuristics)
                
                if total == 0:
                    probs = np.ones(len(heuristics)) / len(heuristics)
                else:
                    probs = heuristics / total
                    
                next_node = np.random.choice(possible_next, p=probs)
                path.append(next_node)
                visited.add(next_node)
                current = next_node
                
            cost = get_path_cost(instance, path)
            paths.append(path)
            costs.append(cost)
            
            if cost < local_best_cost:
                local_best_cost = cost
                local_best_path = path
        
        pheromone *= (1 - rho)
        
        iter_best_idx = np.argmin(costs)
        iter_best_path = paths[iter_best_idx]
        iter_best_cost = costs[iter_best_idx]
        
        if iter_best_cost > 0:
            delta_tau = q / iter_best_cost
            for i in range(n):
                u, v = iter_best_path[i], iter_best_path[(i + 1) % n]
                pheromone[u][v] += delta_tau
                pheromone[v][u] += delta_tau
            
    return local_best_path, local_best_cost, []

#GUI 

result_message_label = None 

def draw_solution(path=None):
    global ax, canvas, current_instance, best_cost
    
    ax.clear()
    
    if current_instance:
        n = current_instance['n']
        coords = current_instance['coords']
        
        if coords and len(coords) == n:
            try:
                # Отрисовка узлов
                x = [coords[i][0] for i in range(n)]
                y = [coords[i][1] for i in range(n)]
                ax.scatter(x, y, c='blue', s=20, zorder=2)
                
                # Отрисовка маршрута
                if path:
                    path_x = [coords[path[i]][0] for i in range(len(path))]
                    path_y = [coords[path[i]][1] for i in range(len(path))]
                    path_x.append(path_x[0])
                    path_y.append(path_y[0])
                    
                    ax.plot(path_x, path_y, c='red', linewidth=1, zorder=1)
                    ax.set_title(f"Стоимость пути: {best_cost:.2f}")
                else:
                    ax.set_title(f"{current_instance['name']} ({n} вершин)")
            except (KeyError, IndexError):
                 ax.text(0.5, 0.5, "Ошибка данных координат\nНевозможно построить график", 
                         transform=ax.transAxes, ha='center')
        else:
            ax.text(0.5, 0.5, "Нет координат для отображения\nТолько расчет стоимости", 
                         transform=ax.transAxes, ha='center')
    else:
        ax.text(0.5, 0.5, "Задача не загружена", 
                     transform=ax.transAxes, ha='center')

    canvas.draw()

def on_load_click():
    global current_instance, problem_var, result_message_label
    
    name = problem_var.get()
    
    if name == 'Контрольный пример':
        # Генерация случайного графа 5 вершин для теста
        coords = {i: (random.randint(0, 100), random.randint(0, 100)) for i in range(5)}
        control_edges = [
            (1, 2, 10), (1, 3, 15), (1, 4, 20), (1, 5, 10),
            (2, 3, 35), (2, 4, 25), (2, 5, 15),
            (3, 4, 30), (3, 5, 20),
            (4, 5, 25)
        ]
        instance = create_tsp_instance("Контрольный", 5, edge_list=control_edges, coords=coords)
        
        for i in range(5):
            for j in range(i+1, 5):
                if instance['dist_matrix'][i][j] == float('inf'):
                    instance['dist_matrix'][i][j] = 999
                    instance['dist_matrix'][j][i] = 999
        
        current_instance = instance
                    
    else:
        try:
            filename = name
            current_instance = read_stp(filename)
        except FileNotFoundError:
            result_message_label.config(text=f"Ошибка: Файл {name} не найден!")
            return

    draw_solution()
    result_message_label.config(text=f"Загружено: {current_instance['name']} ({current_instance['n']} вершин)")

def on_run_click():

    global current_instance, best_path, best_cost
    global algo_var, iter_var, t0_var, ants_var, result_message_label, root, results_table
    global sa_alpha_var, vfsa_c_var, vfsa_jump_var, aco_alpha_var, aco_beta_var, aco_q_var, aco_rho_var
    
    if not current_instance:
        result_message_label.config(text="Сначала загрузите задачу!")
        return
        
    algo = algo_var.get()
    try:
        iterations = int(iter_var.get())
    except ValueError:
        result_message_label.config(text="Ошибка: Некорректное число итераций")
        return
    
    result_message_label.config(text="Выполняется расчет")
    root.update()
    
    start_time = time.time()
    
    try:
        t0 = float(t0_var.get())
    except ValueError:
        t0 = 5000.0

    try:
        ants = int(ants_var.get())
    except ValueError:
        ants = 20

    try:
        sa_alpha = float(sa_alpha_var.get())
    except ValueError:
        sa_alpha = 0.99

    try:
        vfsa_c = float(vfsa_c_var.get())
    except ValueError:
        vfsa_c = 0.5

    try:
        vfsa_jump = float(vfsa_jump_var.get())
    except ValueError:
        vfsa_jump = 0.0

    try:
        aco_alpha = float(aco_alpha_var.get())
    except ValueError:
        aco_alpha = 5.0

    try:
        aco_beta = float(aco_beta_var.get())
    except ValueError:
        aco_beta = 10.0

    try:
        aco_q = float(aco_q_var.get())
    except ValueError:
        aco_q = 1.0

    try:
        aco_rho = float(aco_rho_var.get())
    except ValueError:
        aco_rho = 0.3

    # Базовая валидация диапазонов
    if not (0.0 < sa_alpha < 1.0):
        result_message_label.config(text="Ошибка: alpha SA должен быть в диапазоне (0, 1)")
        return
    if vfsa_c <= 0:
        result_message_label.config(text="Ошибка: коэффициент VFSA c должен быть > 0")
        return
    if not (0.0 <= vfsa_jump <= 1.0):
        result_message_label.config(text="Ошибка: вероятность прыжка VFSA должна быть в диапазоне [0, 1]")
        return
    if aco_alpha < 0 or aco_beta < 0 or aco_q <= 0 or not (0.0 <= aco_rho < 1.0):
        result_message_label.config(text="Ошибка: проверьте коэффициенты ACO")
        return

    if algo == "SA_STANDARD":
        best_path, best_cost, history = solve_sa_standard(
            current_instance,
            max_iter=iterations,
            t0=t0,
            cooling_alpha=sa_alpha,
        )
        algo_name = "Алгоритм имитации отжига (без мод.)"
        params_str = f"T0={t0}, alpha={sa_alpha}, Iter={iterations}"
    elif algo == "SA_MOD":
        best_path, best_cost, history = solve_vfsa(
            current_instance,
            max_iter=iterations,
            t0=t0,
            vfsa_c=vfsa_c,
            long_jump_prob=vfsa_jump,
        )
        algo_name = "SA (модиф.)"
        params_str = f"T0={t0}, c={vfsa_c}, p_jump={vfsa_jump}, Iter={iterations}"
    elif algo == "ACO_STANDARD":
        best_path, best_cost, history = solve_aco_standard(
            current_instance,
            n_ants=10,
            n_iter=30,
            alpha=aco_alpha,
            beta=aco_beta,
            q=aco_q,
            rho=aco_rho,
        )
        algo_name = "Муравьиные алгоритм (без мод.)"
        params_str = f"m=10, a={aco_alpha}, b={aco_beta}, q={aco_q}, rho={aco_rho}, Iter=30"
    else:
        best_path, best_cost, history = solve_aco_modified(
            current_instance,
            n_ants=10,
            n_iter=30,
            alpha=aco_alpha,
            beta=aco_beta,
            q=aco_q,
            rho=aco_rho,
        )
        algo_name = "Муравьиные алгоритм (модиф.)"
        params_str = f"m=10, a={aco_alpha}, b={aco_beta}, q={aco_q}, rho={aco_rho}, Iter=30"
        
    elapsed = time.time() - start_time
    result_message_label.config(text=f"Готово! Стоимость: {best_cost:.2f}")
    
    if results_table:
        results_table.insert("", "end", values=(
            current_instance['name'],
            algo_name,
            params_str,
            f"{best_cost:.2f}",
            f"{elapsed:.4f}"
        ))

    draw_solution(best_path)

def on_compare_all_click():
    global current_instance, best_path, best_cost
    global iter_var, t0_var, ants_var, result_message_label, root, results_table
    global sa_alpha_var, vfsa_c_var, vfsa_jump_var, aco_alpha_var, aco_beta_var, aco_q_var, aco_rho_var

    if not current_instance:
        result_message_label.config(text="Сначала загрузите задачу!")
        return

    try:
        iterations = int(iter_var.get())
    except ValueError:
        result_message_label.config(text="Ошибка: Некорректное число итераций")
        return

    try:
        t0 = float(t0_var.get())
    except ValueError:
        t0 = 5000.0

    try:
        ants = int(ants_var.get())
    except ValueError:
        ants = 20

    try:
        sa_alpha = float(sa_alpha_var.get())
    except ValueError:
        sa_alpha = 0.99

    try:
        vfsa_c = float(vfsa_c_var.get())
    except ValueError:
        vfsa_c = 0.5

    try:
        vfsa_jump = float(vfsa_jump_var.get())
    except ValueError:
        vfsa_jump = 0.0

    try:
        aco_alpha = float(aco_alpha_var.get())
    except ValueError:
        aco_alpha = 5.0

    try:
        aco_beta = float(aco_beta_var.get())
    except ValueError:
        aco_beta = 10.0

    try:
        aco_q = float(aco_q_var.get())
    except ValueError:
        aco_q = 1.0

    try:
        aco_rho = float(aco_rho_var.get())
    except ValueError:
        aco_rho = 0.3

    result_message_label.config(text="Сравнение 4 алгоритмов")
    root.update()

    runs = [
        (
            "SA (без мод.)",
            lambda: solve_sa_standard(current_instance, max_iter=iterations, t0=t0, cooling_alpha=sa_alpha),
            f"T0={t0}, alpha={sa_alpha}, Iter={iterations}",
        ),
        (
            "SA (модиф.)",
            lambda: solve_vfsa(current_instance, max_iter=iterations, t0=t0, vfsa_c=vfsa_c, long_jump_prob=vfsa_jump),
            f"T0={t0}, c={vfsa_c}, p_jump={vfsa_jump}, Iter={iterations}",
        ),
        (
            "ACO (без мод.)",
            lambda: solve_aco_standard(current_instance, n_ants=10, n_iter=30, alpha=aco_alpha, beta=aco_beta, q=aco_q, rho=aco_rho),
            f"m=10, a={aco_alpha}, b={aco_beta}, q={aco_q}, rho={aco_rho}, Iter=30",
        ),
        (
            "ACO (модиф.)",
            lambda: solve_aco_modified(current_instance, n_ants=10, n_iter=30, alpha=aco_alpha, beta=aco_beta, q=aco_q, rho=aco_rho),
            f"m=10, a={aco_alpha}, b={aco_beta}, q={aco_q}, rho={aco_rho}, Iter=30",
        ),
    ]

    best_overall = None
    best_overall_cost = float('inf')
    best_overall_name = ""

    total_runs = len(runs)
    for idx, (algo_name, runner, params_str) in enumerate(runs, start=1):
        result_message_label.config(text=f"Сравнение: {idx}/{total_runs} -> {algo_name}")
        root.update()
        started = time.time()
        path, cost, _ = runner()
        elapsed = time.time() - started

        if results_table:
            results_table.insert("", "end", values=(
                current_instance['name'],
                algo_name,
                params_str,
                f"{cost:.2f}",
                f"{elapsed:.4f}"
            ))
            # Обновляем интерфейс после каждой добавленной строки.
            root.update_idletasks()
            root.update()

        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall = path
            best_overall_name = algo_name

    best_path = best_overall
    best_cost = best_overall_cost
    draw_solution(best_path)
    result_message_label.config(text=f"Сравнение завершено. Лучший: {best_overall_name}, стоимость: {best_overall_cost:.2f}")

def on_algo_change(*_args):
    global algo_var, sa_params_frame, sa_std_params_frame, sa_mod_params_frame, aco_params_frame

    if algo_var is None:
        return

    current = algo_var.get()

    if sa_params_frame is not None:
        sa_params_frame.pack_forget()
    if sa_std_params_frame is not None:
        sa_std_params_frame.pack_forget()
    if sa_mod_params_frame is not None:
        sa_mod_params_frame.pack_forget()
    if aco_params_frame is not None:
        aco_params_frame.pack_forget()

    if current == "SA_STANDARD":
        sa_params_frame.pack(fill=tk.X, pady=(0, 5))
        sa_std_params_frame.pack(fill=tk.X, pady=(0, 5))
    elif current == "SA_MOD":
        sa_params_frame.pack(fill=tk.X, pady=(0, 5))
        sa_mod_params_frame.pack(fill=tk.X, pady=(0, 5))
    elif current == "ACO_STANDARD" or current == "ACO_MOD":
        aco_params_frame.pack(fill=tk.X, pady=(0, 5))

def main():
    global root, fig, ax, canvas, result_message_label, results_table
    global problem_var, algo_var, iter_var, ants_var, t0_var
    global sa_alpha_var, vfsa_c_var, vfsa_jump_var, aco_alpha_var, aco_beta_var, aco_q_var, aco_rho_var
    global sa_params_frame, sa_std_params_frame, sa_mod_params_frame, aco_params_frame
    
    root = tk.Tk()
    root.title("Решение задачи")
    root.geometry("1100x700")
    
    controls_outer = ttk.Frame(root)
    controls_outer.pack(side=tk.LEFT, fill=tk.Y)

    controls_canvas = tk.Canvas(controls_outer, width=390, highlightthickness=0)
    controls_scrollbar = ttk.Scrollbar(controls_outer, orient="vertical", command=controls_canvas.yview)
    controls_canvas.configure(yscrollcommand=controls_scrollbar.set)

    controls_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    controls_canvas.pack(side=tk.LEFT, fill=tk.Y)

    controls_frame = ttk.Frame(controls_canvas, padding="15")
    controls_window = controls_canvas.create_window((0, 0), window=controls_frame, anchor="nw")

    def _on_controls_configure(event):
        controls_canvas.configure(scrollregion=controls_canvas.bbox("all"))
        controls_canvas.itemconfig(controls_window, width=event.width)

    controls_canvas.bind("<Configure>", _on_controls_configure)
    controls_frame.bind("<Configure>", lambda _e: controls_canvas.configure(scrollregion=controls_canvas.bbox("all")))
    
    ttk.Label(controls_frame, text="Элементы управления", font=("Arial", 14, "bold")).pack(pady=(0, 15))
    
    prob_frame = ttk.LabelFrame(controls_frame, text="Выбор задачи", padding="10")
    prob_frame.pack(fill=tk.X, pady=5)
    
    problem_var = tk.StringVar(value="berlin52.stp")
    problem_combo = ttk.Combobox(prob_frame, textvariable=problem_var)
    problem_combo['values'] = ('Контрольный пример', 'berlin52.stp', 'world666.stp')
    problem_combo.pack(pady=5, fill=tk.X)
    ttk.Button(prob_frame, text="Загрузить задачу", command=on_load_click).pack(pady=5, fill=tk.X)
    
    algo_frame = ttk.LabelFrame(controls_frame, text="Выбор алгоритма", padding="10")
    algo_frame.pack(fill=tk.X, pady=10)
    
    algo_var = tk.StringVar(value="SA_MOD")
    ttk.Radiobutton(algo_frame, text="Отжиг без модификации ", variable=algo_var, value="SA_STANDARD").pack(anchor="w", pady=2)
    ttk.Radiobutton(algo_frame, text="Отжиг с модификацией ", variable=algo_var, value="SA_MOD").pack(anchor="w", pady=2)
    ttk.Radiobutton(algo_frame, text="Муравьиный без модификации ", variable=algo_var, value="ACO_STANDARD").pack(anchor="w", pady=2)
    ttk.Radiobutton(algo_frame, text="Муравьиный с модификацией ", variable=algo_var, value="ACO_MOD").pack(anchor="w", pady=2)
    
    param_frame = ttk.LabelFrame(controls_frame, text="Параметры алгоритма", padding="10")
    param_frame.pack(fill=tk.X, pady=5)
    
    ttk.Label(param_frame, text="Кол-во итераций:").pack(anchor="w")
    iter_var = tk.StringVar(value="1000")
    ttk.Entry(param_frame, textvariable=iter_var).pack(fill=tk.X, pady=(0, 5))

    sa_params_frame = ttk.LabelFrame(param_frame, text="Параметры отжига", padding="8")
    ttk.Label(sa_params_frame, text="Нач. температура T0:").pack(anchor="w")
    t0_var = tk.StringVar(value="5000.0")
    ttk.Entry(sa_params_frame, textvariable=t0_var).pack(fill=tk.X, pady=(0, 5))

    sa_std_params_frame = ttk.LabelFrame(param_frame, text="SA без модификации", padding="8")
    ttk.Label(sa_std_params_frame, text="alpha охлаждения:").pack(anchor="w")
    sa_alpha_var = tk.StringVar(value="0.99")
    ttk.Entry(sa_std_params_frame, textvariable=sa_alpha_var).pack(fill=tk.X, pady=(0, 5))

    sa_mod_params_frame = ttk.LabelFrame(param_frame, text="SA с модификацией (VFSA)", padding="8")
    ttk.Label(sa_mod_params_frame, text="c охлаждения:").pack(anchor="w")
    vfsa_c_var = tk.StringVar(value="0.5")
    ttk.Entry(sa_mod_params_frame, textvariable=vfsa_c_var).pack(fill=tk.X, pady=(0, 5))
    ttk.Label(sa_mod_params_frame, text="p длинного прыжка:").pack(anchor="w")
    vfsa_jump_var = tk.StringVar(value="0.0")
    ttk.Entry(sa_mod_params_frame, textvariable=vfsa_jump_var).pack(fill=tk.X, pady=(0, 5))

    aco_params_frame = ttk.LabelFrame(param_frame, text="Параметры муравьиного алгоритма", padding="8")
    ttk.Label(aco_params_frame, text="Кол-во муравьев:").pack(anchor="w")
    ants_var = tk.StringVar(value="20")
    ttk.Entry(aco_params_frame, textvariable=ants_var).pack(fill=tk.X, pady=(0, 5))
    ttk.Label(aco_params_frame, text="Коэфф. значимости феромона alpha:").pack(anchor="w")
    aco_alpha_var = tk.StringVar(value="5.0")
    ttk.Entry(aco_params_frame, textvariable=aco_alpha_var).pack(fill=tk.X, pady=(0, 5))
    ttk.Label(aco_params_frame, text="Коэфф. значимости длины beta:").pack(anchor="w")
    aco_beta_var = tk.StringVar(value="10.0")
    ttk.Entry(aco_params_frame, textvariable=aco_beta_var).pack(fill=tk.X, pady=(0, 5))
    ttk.Label(aco_params_frame, text="Коэфф. добавляемого феромона q:").pack(anchor="w")
    aco_q_var = tk.StringVar(value="1.0")
    ttk.Entry(aco_params_frame, textvariable=aco_q_var).pack(fill=tk.X, pady=(0, 5))
    ttk.Label(aco_params_frame, text="Интенсивность испарения rho:").pack(anchor="w")
    aco_rho_var = tk.StringVar(value="0.3")
    ttk.Entry(aco_params_frame, textvariable=aco_rho_var).pack(fill=tk.X, pady=(0, 5))
    

    ttk.Button(controls_frame, text="ЗАПУСТИТЬ ВЫБРАННЫЙ", command=on_run_click).pack(pady=(20, 8), fill=tk.X, ipady=5)
    ttk.Button(controls_frame, text="СРАВНИТЬ ВСЕ 4 АЛГОРИТМА", command=on_compare_all_click).pack(pady=(0, 10), fill=tk.X, ipady=5)
    
    result_message_label = ttk.Label(controls_frame, text="Готов к работе. Выберите задачу.", font=("Arial", 11), wraplength=320, justify="center")
    result_message_label.pack(pady=10)

    algo_var.trace_add("write", on_algo_change)
    on_algo_change()
    
    plot_frame = ttk.Frame(root)
    plot_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)
    columns = ("problem", "algo", "params", "cost", "time")
    results_table = ttk.Treeview(plot_frame, columns=columns, show="headings", height=5)
    results_table.heading("problem", text="Задача")
    results_table.heading("algo", text="Алгоритм")
    results_table.heading("params", text="Параметры")
    results_table.heading("cost", text="Стоимость")
    results_table.heading("time", text="Время (с)")
    
    results_table.column("problem", width=100)
    results_table.column("algo", width=80)
    results_table.column("params", width=380)
    results_table.column("cost", width=100)
    results_table.column("time", width=80)
    
    results_table.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))

    fig, ax = plt.subplots(figsize=(6, 6))
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    

    root.mainloop()

if __name__ == "__main__":
    main()
