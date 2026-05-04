import math
import random
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.figure

SEARCH_SPACE = [-5.12, 5.12]
DIMENSIONS = 2
BITS_PER_DIM = 16

def rastrigin(x, A=10):
    n = len(x)
    return A * n + sum([(xi**2 - A * math.cos(2 * math.pi * xi)) for xi in x])

def run_real_ga(func, bounds, dim, pop_size=50, generations=100, 
                mutation_rate=0.01, crossover_rate=0.8, 
                selection_method='tournament', tournament_size=3, callback=None):
    
    pop = [[random.uniform(bounds[0], bounds[1]) for _ in range(dim)] for _ in range(pop_size)]
    history = []
    
    for gen in range(generations):
        scores = [func(ind) for ind in pop]
        best_idx = np.argmin(scores)
        best_score = scores[best_idx]
        history.append(best_score)
        best_sol = pop[best_idx]

        if callback:
            callback(gen, pop, best_sol, best_score)

        if selection_method == 'roulette':
            max_f = max(scores)
            weights = [(max_f - s + 1e-10) for s in scores]
            total = sum(weights)
            probs = [w/total for w in weights]
            select = lambda: pop[np.random.choice(len(pop), p=probs)]
        else:
            select = lambda: min(random.sample(pop, tournament_size), key=func)

        new_pop = [best_sol] 
        
        while len(new_pop) < pop_size:
            p1, p2 = select(), select()

            if random.random() < crossover_rate:
                alpha = random.random()
                c1 = [alpha * x + (1 - alpha) * y for x, y in zip(p1, p2)]
                c2 = [(1 - alpha) * x + alpha * y for x, y in zip(p1, p2)]
            else:
                c1, c2 = p1[:], p2[:]

            for child in [c1, c2]:
                if len(new_pop) < pop_size:
                    mutated = []
                    for xi in child:
                        if random.random() < mutation_rate:
                            xi += random.gauss(0, (bounds[1] - bounds[0]) * 0.1)
                            xi = max(min(xi, bounds[1]), bounds[0])
                        mutated.append(xi)
                    new_pop.append(mutated)
        pop = new_pop

    return best_sol, history[-1], history

def run_binary_ga(func, bounds, dim, pop_size=50, generations=100,
                  mutation_rate=0.01, crossover_rate=0.8,
                  selection_method='tournament', tournament_size=3, bits=BITS_PER_DIM, callback=None):
    
    gene_len = dim * bits
    pop = [[random.randint(0, 1) for _ in range(gene_len)] for _ in range(pop_size)]
    history = []

    def decode(ind):
        res = []
        for i in range(dim):
            segment = ind[i*bits : (i+1)*bits]
            val = int("".join(map(str, segment)), 2)
            norm = val / ((1 << bits) - 1)
            res.append(bounds[0] + norm * (bounds[1] - bounds[0]))
        return res

    for gen in range(generations):
        decoded_pop = [decode(ind) for ind in pop]
        scores = [func(real) for real in decoded_pop]
        
        best_idx = np.argmin(scores)
        best_score = scores[best_idx]
        history.append(best_score)
        best_real_sol = decoded_pop[best_idx]

        if callback:
            callback(gen, decoded_pop, best_real_sol, best_score)

        if selection_method == 'roulette':
            max_f = max(scores)
            weights = [(max_f - s + 1e-10) for s in scores]
            total = sum(weights)
            probs = [w/total for w in weights]
            select = lambda: pop[np.random.choice(len(pop), p=probs)]
        else:
            def select_tourn():
                idxs = random.sample(range(pop_size), tournament_size)
                best_i = min(idxs, key=lambda i: scores[i]) 
                return pop[best_i]
            select = select_tourn

        new_pop = [pop[best_idx]]
        
        while len(new_pop) < pop_size:
            p1, p2 = select(), select()
            
            if random.random() < crossover_rate:
                pt = random.randint(1, gene_len - 1)
                c1 = p1[:pt] + p2[pt:]
                c2 = p2[:pt] + p1[pt:]
            else:
                c1, c2 = p1[:], p2[:]

            for child in [c1, c2]:
                if len(new_pop) < pop_size:
                    mutated = [(1-b if random.random() < mutation_rate else b) for b in child]
                    new_pop.append(mutated)
        pop = new_pop

    return best_real_sol, history[-1], history

def run_pso(func, bounds, dim, particles=30, iterations=100, w=0.729, c1=1.49, c2=1.49, inertia=True, callback=None):
    pop = [[random.uniform(bounds[0], bounds[1]) for _ in range(dim)] for _ in range(particles)]
    vel = [[random.uniform(-1, 1) for _ in range(dim)] for _ in range(particles)]
    
    pbest = [p[:] for p in pop]
    pbest_val = [func(p) for p in pop]
    
    gbest_idx = np.argmin(pbest_val)
    gbest = pbest[gbest_idx][:]
    gbest_val = pbest_val[gbest_idx]
    
    history = []
    
    for gen in range(iterations):
        if callback:
             callback(gen, [p[:] for p in pop], gbest, gbest_val)

        current_w = w if inertia else 1.0
        for i in range(particles):
            new_v = []
            for d in range(dim):
                r1, r2 = random.random(), random.random()
                v_next = (current_w * vel[i][d] + 
                          c1 * r1 * (pbest[i][d] - pop[i][d]) + 
                          c2 * r2 * (gbest[d] - pop[i][d]))
                new_v.append(v_next)
                
                pop[i][d] = max(min(pop[i][d] + v_next, bounds[1]), bounds[0])
            vel[i] = new_v
            
            val = func(pop[i])
            if val < pbest_val[i]:
                pbest_val[i] = val
                pbest[i] = pop[i][:]
                if val < gbest_val:
                    gbest_val = val
                    gbest = pop[i][:]
        
        history.append(gbest_val)

    return gbest, gbest_val, history


class AlgorithmGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Сравнение Алгоритмов ")
        self.root.geometry("1100x600")
        
        self.paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        self.left_panel = ttk.Frame(self.paned)
        self.paned.add(self.left_panel, weight=1)
        
        self.settings_frame = ttk.LabelFrame(self.left_panel, text="Параметры")
        self.settings_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self.notebook = ttk.Notebook(self.settings_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.ga_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.ga_frame, text="Генетический Алгоритм")
        self.setup_ga_ui()
        
        self.pso_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.pso_frame, text="Роевой Алгоритм ")
        self.setup_pso_ui()
        
        self.btn_frame = ttk.Frame(self.settings_frame)
        self.btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.run_btn = ttk.Button(self.btn_frame, text="Запустить", command=self.run_algorithm)
        self.run_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        self.table_frame = ttk.LabelFrame(self.left_panel, text="Лог Выполнения")
        self.table_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        columns = ("iter", "fitness", "x", "y")
        self.tree = ttk.Treeview(self.table_frame, columns=columns, show="headings", height=15)
        
        self.tree.heading("iter", text="Итерация")
        self.tree.column("iter", width=60, anchor=tk.CENTER)
        
        self.tree.heading("fitness", text="Лучший Fitness")
        self.tree.column("fitness", width=100, anchor=tk.W)

        self.tree.heading("x", text="Лучший X")
        self.tree.column("x", width=80, anchor=tk.W)

        self.tree.heading("y", text="Лучший Y")
        self.tree.column("y", width=80, anchor=tk.W)
        
        scrollbar = ttk.Scrollbar(self.table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.right_panel = ttk.Frame(self.paned)
        self.paned.add(self.right_panel, weight=2)
        
        self.plot_frame = ttk.LabelFrame(self.right_panel, text="Визуализация Популяции ")
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig = matplotlib.figure.Figure(figsize=(5, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.init_plot()

    def init_plot(self):
        self.ax.clear()
        self.ax.set_title("Пространство Поиска")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_xlim(SEARCH_SPACE[0], SEARCH_SPACE[1])
        self.ax.set_ylim(SEARCH_SPACE[0], SEARCH_SPACE[1])
        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        self.scat_pop = self.ax.scatter([], [], c='blue', s=20, alpha=0.6, label='Частицы')
        self.scat_best = self.ax.scatter([], [], c='red', marker='*', s=150, label='Лучшее')
        self.ax.legend(loc='upper right')
        self.canvas.draw()

    def setup_ga_ui(self):
        self.ga_type_var = tk.StringVar(value="real")
        ttk.Label(self.ga_frame, text="Кодировка:").pack(anchor=tk.W)
        ttk.Radiobutton(self.ga_frame, text="Вещественная", variable=self.ga_type_var, value="real").pack(anchor=tk.W)
        ttk.Radiobutton(self.ga_frame, text="Бинарная", variable=self.ga_type_var, value="binary").pack(anchor=tk.W)
        
        ttk.Label(self.ga_frame, text="Метод селекции:").pack(anchor=tk.W, pady=(5,0))
        self.ga_sel_var = tk.StringVar(value="tournament")
        ttk.Radiobutton(self.ga_frame, text="Турнирный (Модифицированный)", variable=self.ga_sel_var, value="tournament").pack(anchor=tk.W)
        ttk.Radiobutton(self.ga_frame, text="Рулетка (Стандартный)", variable=self.ga_sel_var, value="roulette").pack(anchor=tk.W)

        self.ga_params = {}
        def add(label, default, key):
            f = ttk.Frame(self.ga_frame)
            f.pack(fill=tk.X, pady=2)
            ttk.Label(f, text=label, width=20).pack(side=tk.LEFT)
            var = tk.StringVar(value=str(default))
            ttk.Entry(f, textvariable=var).pack(side=tk.RIGHT, expand=True, fill=tk.X)
            self.ga_params[key] = var
            
        add("Размер популяции:", 50, "pop_size")
        add("Поколений:", 100, "generations")
        add("Вероятность мутации:", 0.05, "mutation_rate")
        
    def setup_pso_ui(self):
        self.pso_inertia_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.pso_frame, text="Использовать инерцию (Модифицированный)", variable=self.pso_inertia_var).pack(anchor=tk.W, pady=(0, 5))

        self.pso_params = {}
        def add(label, default, key):
            f = ttk.Frame(self.pso_frame)
            f.pack(fill=tk.X, pady=2)
            ttk.Label(f, text=label, width=20).pack(side=tk.LEFT)
            var = tk.StringVar(value=str(default))
            ttk.Entry(f, textvariable=var).pack(side=tk.RIGHT, expand=True, fill=tk.X)
            self.pso_params[key] = var

        add("Частиц:", 30, "particles")
        add("Итераций:", 85, "iterations")
        add("Инерция (w):", 0.729, "w")
        add("Коэф. c1:", 1.49, "c1")
        add("Коэф. c2:", 1.49, "c2")

    def run_algorithm(self):
        self.init_plot()
        for i in self.tree.get_children():
            self.tree.delete(i)
        
        tab_index = self.notebook.index(self.notebook.select())
        is_pso = (tab_index == 1)

        def callback(gen, population, best_sol, best_fit):
            self.tree.insert("", 0, values=(gen, f"{best_fit:.5f}", f"{best_sol[0]:.4f}", f"{best_sol[1]:.4f}"))
            
            if is_pso:
                try:
                    pop_arr = np.array(population)
                    self.scat_pop.set_offsets(pop_arr)
                    self.scat_best.set_offsets(np.array([best_sol]))
                    self.canvas.draw()
                except Exception as e:
                    print(f"Ошибка графика: {e}")
            
            self.root.update()

        try:
            if not is_pso: 
                ga_type = self.ga_type_var.get()
                kwargs = {
                    'pop_size': int(self.ga_params['pop_size'].get()),
                    'generations': int(self.ga_params['generations'].get()),
                    'mutation_rate': float(self.ga_params['mutation_rate'].get()),
                    'selection_method': self.ga_sel_var.get(),
                    'callback': callback
                }
                if ga_type == 'real':
                    run_real_ga(rastrigin, SEARCH_SPACE, DIMENSIONS, **kwargs)
                else:
                    run_binary_ga(rastrigin, SEARCH_SPACE, DIMENSIONS, **kwargs)
            else: 
                kwargs = {
                    'particles': int(self.pso_params['particles'].get()),
                    'iterations': int(self.pso_params['iterations'].get()),
                    'w': float(self.pso_params['w'].get()),
                    'c1': float(self.pso_params['c1'].get()),
                    'c2': float(self.pso_params['c2'].get()),
                    'inertia': self.pso_inertia_var.get(),
                    'callback': callback
                }
                run_pso(rastrigin, SEARCH_SPACE, DIMENSIONS, **kwargs)
                
        except ValueError:
            print("Ошибка в параметрах")

if __name__ == "__main__":
    root = tk.Tk()
    app = AlgorithmGUI(root)
    root.mainloop()
