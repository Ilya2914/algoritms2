import matplotlib.pyplot as plt
from main import solve_grid_path, measured_run, random_solvable_state
from main import greedy_manhattan_15, bfs_15, ida_star_15, backjumping_15

ALGORITHMS_GRID = ["base", "warnsdorff", "connectivity", "backjumping"]
ALGORITHMS_PUZZLE = {
    "greedy": greedy_manhattan_15,
    "bfs": bfs_15,
    "ida": ida_star_15,
    "backjumping": backjumping_15,
}

def test_grid(n, m):
    results = {}
    for algo in ALGORITHMS_GRID:
        res = measured_run(
            solve_grid_path,
            n, m,
            set(),
            (0, 0),
            (n-1, m-1),
            algo
        )
        results[algo] = res
    return results


def plot_grid_5x5():
    res = test_grid(5, 5)

    names = list(res.keys())
    time_vals = [res[a]["elapsed_ms"] for a in names]
    mem_vals = [res[a]["peak_kb"] for a in names]

    plt.figure()
    plt.bar(names, time_vals)
    plt.title("Сравнение времени (сетка 5x5)")
    plt.xlabel("Алгоритмы")
    plt.ylabel("Время (мс)")
    plt.xticks(rotation=30)
    plt.show()

    plt.figure()
    plt.bar(names, mem_vals)
    plt.title("Сравнение памяти (сетка 5x5)")
    plt.xlabel("Алгоритмы")
    plt.ylabel("Память (КБ)")
    plt.xticks(rotation=30)
    plt.show()


def plot_grid_7x7():
    res = test_grid(7, 7)

    names = list(res.keys())
    paths = [res[a]["all_paths"] for a in names]
    mem_vals = [res[a]["peak_kb"] for a in names]

    plt.figure()
    plt.bar(names, paths)
    plt.title("Количество путей (7x7, лимит 1 мин)")
    plt.xlabel("Алгоритмы")
    plt.ylabel("Количество путей")
    plt.xticks(rotation=30)
    plt.show()

    plt.figure()
    plt.bar(names, mem_vals)
    plt.title("Память (7x7, лимит 1 мин)")
    plt.xlabel("Алгоритмы")
    plt.ylabel("Память (КБ)")
    plt.xticks(rotation=30)
    plt.show()

def test_puzzle(shuffles):
    results_time = {name: [] for name in ALGORITHMS_PUZZLE}
    results_mem = {name: [] for name in ALGORITHMS_PUZZLE}

    for _ in range(5): 
        state = random_solvable_state(shuffles)

        for name, func in ALGORITHMS_PUZZLE.items():
            res = measured_run(func, state)
            results_time[name].append(res["elapsed_ms"])
            results_mem[name].append(res["peak_kb"])

    avg_time = {k: sum(v)/len(v) for k, v in results_time.items()}
    avg_mem = {k: sum(v)/len(v) for k, v in results_mem.items()}

    return avg_time, avg_mem


def plot_puzzle(shuffles):
    time_res, mem_res = test_puzzle(shuffles)

    names = list(time_res.keys())

    plt.figure()
    plt.bar(names, [time_res[n] for n in names])
    plt.title(f"Время (перемешивания: {shuffles})")
    plt.xlabel("Алгоритмы")
    plt.yscale("log")
    plt.ylabel("Время (мс)")
    plt.xticks(rotation=30)
    plt.show()

    plt.figure()
    plt.bar(names, [mem_res[n] for n in names])
    plt.title(f"Память (перемешивания: {shuffles})")
    plt.xlabel("Алгоритмы")
    plt.yscale("log")
    plt.ylabel("Память (КБ)")
    plt.xticks(rotation=30)
    plt.show()


def run_all():
    print("Сетка 5x5...")
    plot_grid_5x5()

    print("Сетка 7x7...")
    plot_grid_7x7()

    print("Пятнашки 30...")
    plot_puzzle(30)

    print("Пятнашки 60...")
    plot_puzzle(60)


if __name__ == "__main__":
    run_all()