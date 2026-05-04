import matplotlib.pyplot as plt
import numpy as np
import random
import math
from tsp_lab import create_tsp_instance, solve_sa_standard, solve_vfsa, solve_aco_standard, read_stp

def generate_test_instance(n=30):
    coords = {i: (random.randint(0, 100), random.randint(0, 100)) for i in range(n)}
    instance = create_tsp_instance("Test", n, coords=coords)
    
    # Заполнение матрицы расстояний евклидовыми расстояниями
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                instance['dist_matrix'][i][j] = np.sqrt(dx**2 + dy**2)
            else:
                instance['dist_matrix'][i][j] = 0
    return instance

def plot_parameter(param_name, values, results, title, xscale='linear'):
    plt.figure(figsize=(8, 5))
    plt.plot(values, results, marker='o', linestyle='-')
    if xscale != 'linear':
        plt.xscale(xscale)
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel('Средняя стоимость пути')
    plt.grid(True)
    plt.show()

def solve_sa_standard_random_start(instance, max_iter=10000, t0=5000.0, cooling_alpha=0.99):
    n = instance['n']
    current_path = list(range(n))
    random.shuffle(current_path)
    current_cost = 0
    matrix = instance['dist_matrix']
    for i in range(n):
        u, v = current_path[i], current_path[(i + 1) % n]
        current_cost += matrix[u][v]

    best_path = list(current_path)
    best_cost = current_cost

    T = t0
    for _ in range(1, max_iter + 1):
        if T < 1e-6:
            break

        i, j = sorted(random.sample(range(n), 2))
        candidate_path = current_path[:i] + current_path[i:j+1][::-1] + current_path[j+1:]

        candidate_cost = 0
        for idx in range(n):
            u, v = candidate_path[idx], candidate_path[(idx + 1) % n]
            candidate_cost += matrix[u][v]

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
            if current_cost < best_cost:
                best_cost = current_cost
                best_path = list(current_path)

        T *= cooling_alpha

    return best_path, best_cost, []

def run_experiment(instance, algo, param_name, param_values, default_kwargs, num_runs=5):
    avg_results = []
    print(f"Запуск экспериментов для параметра {param_name}...")
    for value_index, val in enumerate(param_values):
        kwargs = default_kwargs.copy()
        kwargs[param_name] = val
        
        costs = []
        for run_index in range(num_runs):
            seed = 42 + value_index * 1000 + run_index
            random.seed(seed)
            np.random.seed(seed)
            _, cost, _ = algo(instance, **kwargs)
            costs.append(cost)
        avg_cost = np.mean(costs)
        avg_results.append(avg_cost)
        print(f" {param_name} = {val:.4f} -> Средняя стоимость: {avg_cost:.2f}")
        
    return avg_results

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    

    filename = 'berlin52.stp'
    print(f"Загрузка файла {filename}...")
    inst = read_stp(filename) 
    print(f"Загружена задача: {inst['name']} ({inst['n']} вершин)")
    
    # 1. ИО - Начальная температура (t0)
    t0_vals = [10, 100, 500, 1000, 5000]
    res_t0 = run_experiment(inst, solve_sa_standard_random_start, 't0', t0_vals, {'max_iter': 2000, 'cooling_alpha': 0.99})
    plot_parameter("T0 (Начальная температура)", t0_vals, res_t0, "Имитация отжига: Влияние T0", xscale='log')
    
    # 2. ИО - Коэффициент охлаждения (cooling_alpha)
    alpha_vals = [0.8, 0.9, 0.95, 0.99, 0.999]
    res_alpha = run_experiment(inst, solve_sa_standard, 'cooling_alpha', alpha_vals, {'max_iter': 2000, 't0': 5000})
    plot_parameter("alpha (Коэффициент охлаждения)", alpha_vals, res_alpha, "Имитация отжига: Влияние alpha")

    # 3. ИО - Вероятность длинного прыжка (long_jump_prob)
    jump_vals = [0.0, 0.2, 0.5, 0.8, 1.0]
    res_jump = run_experiment(inst, solve_vfsa, 'long_jump_prob', jump_vals, {'max_iter': 2000, 't0': 5000, 'vfsa_c': 0.5})
    plot_parameter("long_jump_prob", jump_vals, res_jump, "Модиф. ИО: Влияние вероятности прыжка")

    # 4. МА - Влияние феромона (alpha)
    aco_alpha_vals = [0.0, 0.5, 1.0, 2.0, 5.0]
    res_aco_alpha = run_experiment(inst, solve_aco_standard, 'alpha', aco_alpha_vals, {'n_iter': 30, 'n_ants': 10, 'beta': 10.0, 'rho': 0.3})
    plot_parameter("alpha (Вес феромона)", aco_alpha_vals, res_aco_alpha, "Муравьиный алгоритм: Влияние alpha")

    # 5. МА - Влияние эвристики (beta)
    aco_beta_vals = [0.0, 1.0, 2.0, 5.0, 10.0]
    res_aco_beta = run_experiment(inst, solve_aco_standard, 'beta', aco_beta_vals, {'n_iter': 30, 'n_ants': 10, 'alpha': 5.0, 'rho': 0.3})
    plot_parameter("beta (Вес эвристики / расст.)", aco_beta_vals, res_aco_beta, "Муравьиный алгоритм: Влияние beta")

    # 6. МА - Коэффициент испарения (rho)
    aco_rho_vals = [0.01, 0.1, 0.3, 0.6, 0.9]
    res_aco_rho = run_experiment(inst, solve_aco_standard, 'rho', aco_rho_vals, {'n_iter': 30, 'n_ants': 10, 'alpha': 5.0, 'beta': 10.0})
    plot_parameter("rho (Испарение феромона)", aco_rho_vals, res_aco_rho, "Муравьиный алгоритм: Влияние rho")
