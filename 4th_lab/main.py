import random
import threading
import time
import tracemalloc
from collections import deque
import tkinter as tk
from tkinter import messagebox, ttk


GOAL_15 = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    0,
)
GOAL_POS_15 = {value: idx for idx, value in enumerate(GOAL_15)}


def measured_run(func, *args, **kwargs):
    tracemalloc.start()
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if not isinstance(result, dict):
        result = {
            "solved": False,
            "path": [],
            "expanded": 0,
            "details": "Unexpected result type",
        }

    result["elapsed_ms"] = elapsed_ms
    result["peak_kb"] = peak / 1024.0
    return result

def build_grid_graph(n, m, blocked):
    cells = []
    cell_to_idx = {}
    for r in range(n):
        for c in range(m):
            if (r, c) not in blocked:
                idx = len(cells)
                cells.append((r, c))
                cell_to_idx[(r, c)] = idx

    neigh = [[] for _ in range(len(cells))]
    for i, (r, c) in enumerate(cells):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (nr, nc) in cell_to_idx:
                neigh[i].append(cell_to_idx[(nr, nc)])

    return cells, cell_to_idx, neigh


def connectivity_ok(total_free, neigh, visited_mask):
    unvisited = [i for i in range(total_free) if (visited_mask & (1 << i)) == 0]
    if not unvisited:
        return True

    q = deque([unvisited[0]])
    seen = {unvisited[0]}
    while q:
        v = q.popleft()
        for u in neigh[v]:
            if (visited_mask & (1 << u)) == 0 and u not in seen:
                seen.add(u)
                q.append(u)

    return len(seen) == len(unvisited)


def ordered_next(pos, visited_mask, neigh, use_warnsdorff):
    candidates = [u for u in neigh[pos] if (visited_mask & (1 << u)) == 0]
    if not use_warnsdorff:
        return candidates

    def onward_degree(v):
        return sum(1 for x in neigh[v] if (visited_mask & (1 << x)) == 0)

    candidates.sort(key=onward_degree)
    return candidates


def count_all_paths_dfs(
    cells,
    cell_to_idx,
    neigh,
    start,
    finish,
    use_warnsdorff=False,
    use_connectivity=False,
):
    """Count successful Hamiltonian paths from start to finish.
    Respects Warnsdorff ordering and connectivity pruning when requested.
    Returns (count, exceeded).
    """
    start_idx = cell_to_idx.get(start)
    finish_idx = cell_to_idx.get(finish)
    total_free = len(cells)

    if start_idx is None or finish_idx is None:
        return 0, False

    all_paths_count = [0]
    t0 = time.perf_counter()
    time_limit = 60.0
    exceeded = [False]

    def dfs(pos, visited_mask):
        if time.perf_counter() - t0 > time_limit:
            exceeded[0] = True
            return

        visited_len = visited_mask.bit_count()
        if visited_len == total_free:
            if pos == finish_idx:
                all_paths_count[0] += 1
            return

        remaining = total_free - visited_len
        next_nodes = ordered_next(pos, visited_mask, neigh, use_warnsdorff)
        # avoid visiting finish prematurely (only allow it as last step)
        if remaining > 1:
            next_nodes = [u for u in next_nodes if u != finish_idx]

        for nxt in next_nodes:
            if (visited_mask & (1 << nxt)) == 0:
                new_mask = visited_mask | (1 << nxt)
                if use_connectivity and not connectivity_ok(total_free, neigh, new_mask):
                    continue
                dfs(nxt, new_mask)

    dfs(start_idx, 1 << start_idx)
    return all_paths_count[0], exceeded[0]


def solve_grid_path(n, m, blocked, start, finish, mode="base"):
    cells, cell_to_idx, neigh = build_grid_graph(n, m, blocked)
    total_free = len(cells)

    start_idx = cell_to_idx.get(start)
    finish_idx = cell_to_idx.get(finish)
    if start_idx is None or finish_idx is None:
        return {
            "solved": False,
            "path": [],
            "expanded": 0,
            "all_paths": 0,
            "details": "Start/finish blocked or outside grid",
        }

    use_warnsdorff = mode in {"warnsdorff", "backjumping"}
    use_connectivity = mode in {"connectivity", "backjumping"}
    use_backjumping = mode == "backjumping"

    expanded = 0
    dead_cache = set()
    path = [start_idx]
    start_mask = 1 << start_idx
    t0 = time.perf_counter()
    time_limit = 30.0  # 30 seconds

    def dfs(pos, visited_mask):
        nonlocal expanded

        if time.perf_counter() - t0 > time_limit:
            return False, "Time limit exceeded"

        expanded += 1

        if use_backjumping:
            state = (pos, visited_mask)
            if state in dead_cache:
                return False, "cached dead-end"

        if len(path) == total_free:
            return pos == finish_idx, "done"

        remaining = total_free - len(path)
        next_nodes = ordered_next(pos, visited_mask, neigh, use_warnsdorff)
        if remaining > 1:
            next_nodes = [u for u in next_nodes if u != finish_idx]

        for nxt in next_nodes:
            path.append(nxt)
            new_mask = visited_mask | (1 << nxt)

            if use_connectivity and not connectivity_ok(total_free, neigh, new_mask):
                path.pop()
                continue

            ok, reason = dfs(nxt, new_mask)
            if ok:
                return True, reason

            path.pop()

        if use_backjumping:
            dead_cache.add((pos, visited_mask))
        return False, "dead-end"

    solved, details = dfs(start_idx, start_mask)
    
    # Count successful paths (respecting mode flags)
    all_paths_count, paths_exceeded = count_all_paths_dfs(
        cells,
        cell_to_idx,
        neigh,
        start,
        finish,
        use_warnsdorff=use_warnsdorff,
        use_connectivity=use_connectivity,
    )
    
    return {
        "solved": solved,
        "path": [cells[i] for i in path] if solved else [],
        "expanded": expanded,
        "all_paths": all_paths_count,
        "all_paths_exceeded": paths_exceeded,
        "details": "Solved" if solved else details,
    }

def manhattan_15(state):
    dist = 0
    for i, value in enumerate(state):
        if value == 0:
            continue
        gi = GOAL_POS_15[value]
        r1, c1 = divmod(i, 4)
        r2, c2 = divmod(gi, 4)
        dist += abs(r1 - r2) + abs(c1 - c2)
    return dist


def neighbors_15(state):
    z = state.index(0)
    r, c = divmod(z, 4)
    result = []
    for dr, dc, move in [(-1, 0, "U"), (1, 0, "D"), (0, -1, "L"), (0, 1, "R")]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 4 and 0 <= nc < 4:
            nz = nr * 4 + nc
            arr = list(state)
            arr[z], arr[nz] = arr[nz], arr[z]
            result.append((tuple(arr), move))
    return result


def is_solvable_15(state):
    arr = [x for x in state if x != 0]
    inversions = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] > arr[j]:
                inversions += 1

    row_from_bottom = 4 - (state.index(0) // 4)
    return (inversions + row_from_bottom) % 2 == 1


def greedy_manhattan_15(start_state, max_steps=400):
    expanded = 0
    current = tuple(start_state)
    visited = {current}
    path = [current]
    t0 = time.perf_counter()
    time_limit = 5.0

    for _ in range(max_steps):
        if time.perf_counter() - t0 > time_limit:
            return {
                "solved": False,
                "path": path,
                "expanded": expanded,
                "details": "Time limit exceeded",
            }
        expanded += 1
        if current == GOAL_15:
            return {"solved": True, "path": path, "expanded": expanded, "details": "Solved"}

        candidates = [(manhattan_15(ns), ns) for ns, _ in neighbors_15(current) if ns not in visited]
        if not candidates:
            return {
                "solved": False,
                "path": path,
                "expanded": expanded,
                "details": "Local minimum / dead-end",
            }

        candidates.sort(key=lambda x: x[0])
        current = candidates[0][1]
        visited.add(current)
        path.append(current)

    return {
        "solved": False,
        "path": path,
        "expanded": expanded,
        "details": "Step limit exceeded",
    }


def bfs_15(start_state, max_nodes=250000):
    start_state = tuple(start_state)
    expanded = 0
    q = deque([start_state])
    prev = {start_state: None}
    t0 = time.perf_counter()
    time_limit = 5.0

    while q:
        if time.perf_counter() - t0 > time_limit:
            return {"solved": False, "path": [], "expanded": expanded, "details": "Time limit exceeded"}
        s = q.popleft()
        expanded += 1

        if expanded > max_nodes:
            return {"solved": False, "path": [], "expanded": expanded, "details": "Node limit exceeded"}

        if s == GOAL_15:
            path = []
            cur = s
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return {"solved": True, "path": path, "expanded": expanded, "details": "Solved"}

        for ns, _ in neighbors_15(s):
            if ns not in prev:
                prev[ns] = s
                q.append(ns)

    return {"solved": False, "path": [], "expanded": expanded, "details": "No solution"}


def ida_star_15(start_state, max_bound=120):
    start_state = tuple(start_state)
    if not is_solvable_15(start_state):
        return {"solved": False, "path": [], "expanded": 0, "details": "Unsolvable"}

    expanded = 0
    path = [start_state]
    visited = {start_state}
    t0 = time.perf_counter()
    time_limit = 5.0

    def search(g, bound):
        nonlocal expanded
        if time.perf_counter() - t0 > time_limit:
            return "timeout"
        s = path[-1]
        f = g + manhattan_15(s)

        if f > bound:
            return f

        expanded += 1
        if s == GOAL_15:
            return True

        min_next = float("inf")
        for ns, _ in neighbors_15(s):
            if ns in visited:
                continue

            visited.add(ns)
            path.append(ns)
            t = search(g + 1, bound)
            if t == "timeout":
                path.pop()
                visited.remove(ns)
                return "timeout"
            if t is True:
                return True
            if t < min_next:
                min_next = t
            path.pop()
            visited.remove(ns)

        return min_next

    bound = manhattan_15(start_state)
    while bound <= max_bound:
        t = search(0, bound)
        if t == "timeout":
            return {"solved": False, "path": [], "expanded": expanded, "details": "Time limit exceeded"}
        if t is True:
            return {"solved": True, "path": path.copy(), "expanded": expanded, "details": "Solved"}
        if t == float("inf"):
            break
        bound = t

    return {"solved": False, "path": [], "expanded": expanded, "details": "Bound limit exceeded"}


def backjumping_15(start_state, max_depth=80):
    start_state = tuple(start_state)
    if not is_solvable_15(start_state):
        return {"solved": False, "path": [], "expanded": 0, "details": "Unsolvable"}

    expanded = 0
    t0 = time.perf_counter()
    time_limit = 5.0

    for depth_limit in range(manhattan_15(start_state), max_depth + 1):
        if time.perf_counter() - t0 > time_limit:
            return {"solved": False, "path": [], "expanded": expanded, "details": "Time limit exceeded"}
        path = [start_state]
        visited = {start_state}
        best_depth_seen = {start_state: 0}

        def dfs(depth):
            nonlocal expanded
            if time.perf_counter() - t0 > time_limit:
                return False
            s = path[-1]
            expanded += 1

            if s == GOAL_15:
                return True
            if depth == depth_limit:
                return False
            if depth + manhattan_15(s) > depth_limit:
                return False

            for ns, _ in neighbors_15(s):
                nd = depth + 1
                prev_best = best_depth_seen.get(ns)
                if prev_best is not None and prev_best <= nd:
                    continue
                if ns in visited:
                    continue

                best_depth_seen[ns] = nd
                visited.add(ns)
                path.append(ns)
                ok = dfs(nd)
                if ok:
                    return True
                path.pop()
                visited.remove(ns)

            return False

        if dfs(0):
            return {
                "solved": True,
                "path": path,
                "expanded": expanded,
                "details": f"Solved at depth {depth_limit}",
            }

    return {"solved": False, "path": [], "expanded": expanded, "details": "Depth limit exceeded"}

def random_solvable_state(steps=80):
    state = list(GOAL_15)
    prev = None
    for _ in range(steps):
        cur = tuple(state)
        candidates = [ns for ns, _ in neighbors_15(cur) if ns != prev]
        nxt = random.choice(candidates)
        prev = cur
        state = list(nxt)
    return tuple(state)

def run_gui():
    if tk is None or ttk is None:
        print("GUI недоступен: в текущем Python нет поддержки tkinter.")
        print("Запустите программу интерпретатором Python с поддержкой tkinter.")
        return

    root = tk.Tk()
    root.title("Лабораторная 4")
    root.geometry("1040x760")

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    def run_async(task_func, on_done, on_error=None):
        box = {"done": False, "result": None, "error": None}

        def worker():
            try:
                box["result"] = task_func()
            except Exception as exc:
                box["error"] = exc
            finally:
                box["done"] = True

        threading.Thread(target=worker, daemon=True).start()

        def poll():
            if not box["done"]:
                root.after(80, poll)
                return
            if box["error"] is not None:
                if on_error is not None:
                    on_error(box["error"])
                else:
                    messagebox.showerror("Ошибка", str(box["error"]))
            else:
                on_done(box["result"])

        root.after(80, poll)

    def set_widgets_enabled(widget_specs, enabled):
        for widget, on_state in widget_specs:
            widget.configure(state=on_state if enabled else "disabled")

    # ----- Tab 1: Grid path -----
    tab_grid = ttk.Frame(notebook)
    notebook.add(tab_grid, text="Задача 1: Путь по сетке")

    grid_state = {
        "n": 7,
        "m": 7,
        "start": (0, 0),
        "finish": (6, 6),
        "blocked": set(),
        "solution": [],
    }

    n_var = tk.IntVar(value=7)
    m_var = tk.IntVar(value=7)
    mode_click_var = tk.StringVar(value="start")
    algo_grid_var = tk.StringVar(value="base")
    status_grid_var = tk.StringVar(value="Готово")

    top_grid = ttk.Frame(tab_grid)
    top_grid.pack(fill="x", padx=8, pady=8)

    ttk.Label(top_grid, text="N:").grid(row=0, column=0, sticky="w")
    ttk.Entry(top_grid, textvariable=n_var, width=5).grid(row=0, column=1, padx=4)
    ttk.Label(top_grid, text="M:").grid(row=0, column=2, sticky="w")
    ttk.Entry(top_grid, textvariable=m_var, width=5).grid(row=0, column=3, padx=4)

    ttk.Label(top_grid, text="Алгоритм:").grid(row=0, column=4, sticky="e", padx=(10, 0))
    ttk.Combobox(
        top_grid,
        textvariable=algo_grid_var,
        values=["base", "warnsdorff", "connectivity", "backjumping"],
        state="readonly",
        width=14,
    ).grid(row=0, column=5, padx=4)

    grid_params = ttk.Frame(tab_grid)
    grid_params.pack(fill="x", padx=8)

    canvas = tk.Canvas(tab_grid, bg="white", highlightthickness=1, highlightbackground="#bdbdbd")
    canvas.pack(fill="both", expand=True, padx=8, pady=8)

    click_mode = ttk.Frame(tab_grid)
    click_mode.pack(fill="x", padx=8)
    ttk.Label(click_mode, text="Режим клика:").pack(side="left")
    ttk.Radiobutton(click_mode, text="Старт", variable=mode_click_var, value="start").pack(side="left", padx=4)
    ttk.Radiobutton(click_mode, text="Финиш", variable=mode_click_var, value="finish").pack(side="left", padx=4)
    ttk.Radiobutton(click_mode, text="Препятствие", variable=mode_click_var, value="block").pack(side="left", padx=4)

    status_grid = ttk.Frame(tab_grid)
    status_grid.pack(fill="x", padx=8, pady=(4, 8))
    ttk.Label(status_grid, textvariable=status_grid_var).pack(side="left")

    grid_controls = []

    def cell_from_xy(x, y):
        w = max(canvas.winfo_width(), 1)
        h = max(canvas.winfo_height(), 1)
        cw = w / grid_state["m"]
        ch = h / grid_state["n"]
        c = int(x // cw)
        r = int(y // ch)
        if 0 <= r < grid_state["n"] and 0 <= c < grid_state["m"]:
            return (r, c)
        return None

    def draw_grid():
        canvas.delete("all")
        w = max(canvas.winfo_width(), 560)
        h = max(canvas.winfo_height(), 560)
        n = grid_state["n"]
        m = grid_state["m"]
        cw = w / m
        ch = h / n

        idx_map = {cell: i for i, cell in enumerate(grid_state["solution"])}
        for r in range(n):
            for c in range(m):
                x1, y1 = c * cw, r * ch
                x2, y2 = x1 + cw, y1 + ch
                cell = (r, c)

                fill = "#f5f5f5"
                if cell in grid_state["blocked"]:
                    fill = "#303030"
                elif cell in idx_map:
                    fill = "#8ecae6"
                if cell == grid_state["start"]:
                    fill = "#34a853"
                if cell == grid_state["finish"]:
                    fill = "#ea4335"

                canvas.create_rectangle(x1, y1, x2, y2, fill=fill, outline="#909090")
                if cell in idx_map:
                    canvas.create_text(
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                        text=str(idx_map[cell] + 1),
                        font=("Helvetica", 10, "bold"),
                    )

    def reset_grid():
        grid_state["n"] = max(2, min(9, int(n_var.get() or 7)))
        grid_state["m"] = max(2, min(9, int(m_var.get() or 7)))
        grid_state["start"] = (0, 0)
        grid_state["finish"] = (grid_state["n"] - 1, grid_state["m"] - 1)
        grid_state["blocked"] = set()
        grid_state["solution"] = []
        draw_grid()
        status_grid_var.set("Сетка создана")

    def on_grid_click(event):
        cell = cell_from_xy(event.x, event.y)
        if not cell:
            return
        mode = mode_click_var.get()
        if mode == "start":
            if cell != grid_state["finish"]:
                grid_state["start"] = cell
                grid_state["blocked"].discard(cell)
        elif mode == "finish":
            if cell != grid_state["start"]:
                grid_state["finish"] = cell
                grid_state["blocked"].discard(cell)
        else:
            if cell not in (grid_state["start"], grid_state["finish"]):
                if cell in grid_state["blocked"]:
                    grid_state["blocked"].remove(cell)
                else:
                    grid_state["blocked"].add(cell)
        grid_state["solution"] = []
        draw_grid()

    def solve_grid_gui():
        set_widgets_enabled(grid_controls, False)
        status_grid_var.set("Поиск выполняется...")

        def task():
            return measured_run(
                solve_grid_path,
                grid_state["n"],
                grid_state["m"],
                grid_state["blocked"],
                grid_state["start"],
                grid_state["finish"],
                algo_grid_var.get(),
            )

        def done(result):
            set_widgets_enabled(grid_controls, True)
            if result["solved"]:
                grid_state["solution"] = result["path"]
                paths_str = f">{result['all_paths']}" if result['all_paths_exceeded'] else str(result['all_paths'])
                status_grid_var.set(
                    f"Решено. Узлы={result['expanded']}, Пути={paths_str}, Время={result['elapsed_ms']:.1f} мс, Память={result['peak_kb']:.1f} KB"
                )
                messagebox.showinfo("Задача 1", "Путь построен успешно.")
            else:
                grid_state["solution"] = []
                status_grid_var.set(
                    f"Решения нет ({result['details']}). Узлы={result['expanded']}, Время={result['elapsed_ms']:.1f} мс, Память={result['peak_kb']:.1f} KB"
                )
                messagebox.showwarning("Задача 1", "Решение не найдено. Попробуйте другой алгоритм или сетку.")
            draw_grid()

        def fail(exc):
            set_widgets_enabled(grid_controls, True)
            status_grid_var.set(f"Ошибка: {exc}")
            messagebox.showerror("Задача 1", str(exc))

        run_async(task, done, fail)

    def compare_grid_gui():
        set_widgets_enabled(grid_controls, False)
        status_grid_var.set("Сравнение алгоритмов выполняется...")

        def task():
            lines = []
            for mode in ["base", "warnsdorff", "connectivity", "backjumping"]:
                result = measured_run(
                    solve_grid_path,
                    grid_state["n"],
                    grid_state["m"],
                    grid_state["blocked"],
                    grid_state["start"],
                    grid_state["finish"],
                    mode,
                )
                moves = len(result.get("path", [])) - 1 if result.get("path") else -1
                paths_str = f">{result['all_paths']}" if result.get('all_paths_exceeded', False) else f"{result['all_paths']:7d}"
                lines.append(
                    f"{mode:12s} | решено={str(result['solved']):5s} | шаги={moves:4d} | "
                    f"пути={paths_str:>7} | узлы={result['expanded']:7d} | время={result['elapsed_ms']:8.1f} мс | "
                    f"память={result['peak_kb']:8.1f} KB"
                )
            return lines

        def done(lines):
            set_widgets_enabled(grid_controls, True)
            status_grid_var.set("Сравнение завершено")
            messagebox.showinfo("Сравнение: задача 1", "\n".join(lines))

        def fail(exc):
            set_widgets_enabled(grid_controls, True)
            status_grid_var.set(f"Ошибка: {exc}")
            messagebox.showerror("Задача 1", str(exc))

        run_async(task, done, fail)

    btn_grid_create = ttk.Button(top_grid, text="Создать сетку", command=reset_grid)
    btn_grid_solve = ttk.Button(top_grid, text="Решить", command=solve_grid_gui)
    btn_grid_compare = ttk.Button(top_grid, text="Сравнить все", command=compare_grid_gui)
    btn_grid_create.grid(row=0, column=6, padx=8)
    btn_grid_solve.grid(row=0, column=7, padx=4)
    btn_grid_compare.grid(row=0, column=8, padx=4)

    grid_controls = [
        (btn_grid_create, "normal"),
        (btn_grid_solve, "normal"),
        (btn_grid_compare, "normal"),
    ]

    canvas.bind("<Button-1>", on_grid_click)
    canvas.bind("<Configure>", lambda _: draw_grid())

    tab_puzzle = ttk.Frame(notebook)
    notebook.add(tab_puzzle, text="Задача 2: Пятнашки")

    puzzle_state = {
        "board": list(GOAL_15),
    }

    algo_puzzle_var = tk.StringVar(value="ida")
    status_puzzle_var = tk.StringVar(value="Готово")
    random_steps_var = tk.IntVar(value=35)
    tile_buttons = []

    top_puzzle = ttk.Frame(tab_puzzle)
    top_puzzle.pack(fill="x", padx=8, pady=8)
    ttk.Label(top_puzzle, text="Алгоритм: IDA*").pack(side="left")

    puzzle_params = ttk.Frame(tab_puzzle)
    puzzle_params.pack(fill="x", padx=8)
    ttk.Label(puzzle_params, text="Случайное перемешивание").grid(row=0, column=0, sticky="w")
    ttk.Entry(puzzle_params, textvariable=random_steps_var, width=6).grid(row=0, column=1, padx=4)

    grid_puzzle = ttk.Frame(tab_puzzle)
    grid_puzzle.pack(padx=8, pady=8)

    status_puzzle = ttk.Frame(tab_puzzle)
    status_puzzle.pack(fill="x", padx=8, pady=(0, 8))
    ttk.Label(status_puzzle, textvariable=status_puzzle_var).pack(side="left")

    puzzle_controls = []

    def sync_puzzle_buttons():
        for i, value in enumerate(puzzle_state["board"]):
            r, c = divmod(i, 4)
            btn = tile_buttons[r][c]
            if value == 0:
                btn.configure(text="", bg="#222", fg="white")
            else:
                btn.configure(text=str(value), bg="#f3f6ff", fg="#111")

    def click_tile(index):
        z = puzzle_state["board"].index(0)
        rz, cz = divmod(z, 4)
        r, c = divmod(index, 4)
        if abs(r - rz) + abs(c - cz) == 1:
            puzzle_state["board"][z], puzzle_state["board"][index] = (
                puzzle_state["board"][index],
                puzzle_state["board"][z],
            )
            sync_puzzle_buttons()

    def reset_puzzle_goal():
        puzzle_state["board"] = list(GOAL_15)
        sync_puzzle_buttons()
        status_puzzle_var.set("Эталонное состояние")

    def random_puzzle():
        steps = max(5, int(random_steps_var.get() or 35))
        puzzle_state["board"] = list(random_solvable_state(steps))
        sync_puzzle_buttons()
        status_puzzle_var.set("Случайное решаемое состояние")

    def animate_puzzle(path, idx=0):
        if idx >= len(path):
            return
        puzzle_state["board"] = list(path[idx])
        sync_puzzle_buttons()
        root.after(120, lambda: animate_puzzle(path, idx + 1))

    def build_puzzle_mapping():
        # One shared profile so "Решить" and "Сравнить все" are directly comparable.
        greedy_steps = 600
        bfs_nodes = 250000
        ida_bound = 120
        backjump_depth = 80
        return {
            "greedy": lambda s: greedy_manhattan_15(s, max_steps=greedy_steps),
            "bfs": lambda s: bfs_15(s, max_nodes=bfs_nodes),
            "ida": lambda s: ida_star_15(s, max_bound=ida_bound),
            "backjumping": lambda s: backjumping_15(s, max_depth=backjump_depth),
        }

    def solve_puzzle_gui():
        set_widgets_enabled(puzzle_controls, False)
        status_puzzle_var.set("Поиск решения выполняется...")
        state = tuple(puzzle_state["board"])
        mapping = build_puzzle_mapping()

        def task():
            return measured_run(mapping[algo_puzzle_var.get()], state)

        def done(result):
            set_widgets_enabled(puzzle_controls, True)
            if result["solved"]:
                status_puzzle_var.set(
                    f"Решено за {len(result['path']) - 1} ходов. Узлы={result['expanded']}, "
                    f"Время={result['elapsed_ms']:.1f} мс, Память={result['peak_kb']:.1f} KB"
                )
                messagebox.showinfo("Задача 2", "Решение построено. Показываю шаги на поле.")
                animate_puzzle(result["path"])
            else:
                status_puzzle_var.set(
                    f"Решения нет ({result['details']}). Узлы={result['expanded']}, "
                    f"Время={result['elapsed_ms']:.1f} мс, Память={result['peak_kb']:.1f} KB"
                )
                messagebox.showinfo(
                    "Задача 2",
                    "Решение не найдено в текущих лимитах поиска. Попробуйте другую стартовую позицию или алгоритм.",
                )

        def fail(exc):
            set_widgets_enabled(puzzle_controls, True)
            status_puzzle_var.set(f"Ошибка: {exc}")
            messagebox.showerror("Задача 2", str(exc))

        run_async(task, done, fail)

    def compare_puzzle_gui():
        set_widgets_enabled(puzzle_controls, False)
        status_puzzle_var.set("Сравнение алгоритмов выполняется...")
        state = tuple(puzzle_state["board"])
        mapping = build_puzzle_mapping()

        def task():
            lines = []
            for name in ["greedy", "bfs", "ida", "backjumping"]:
                result = measured_run(mapping[name], state)
                moves = len(result.get("path", [])) - 1 if result.get("path") else -1
                lines.append(
                    f"{name:12s} | решено={str(result['solved']):5s} | ходы={moves:4d} | "
                    f"узлы={result['expanded']:7d} | время={result['elapsed_ms']:8.1f} мс | "
                    f"память={result['peak_kb']:8.1f} KB"
                )
            return lines

        def done(lines):
            set_widgets_enabled(puzzle_controls, True)
            status_puzzle_var.set("Сравнение завершено")
            messagebox.showinfo("Сравнение: задача 2", "\n".join(lines))

        def fail(exc):
            set_widgets_enabled(puzzle_controls, True)
            status_puzzle_var.set(f"Ошибка: {exc}")
            messagebox.showerror("Задача 2", str(exc))

        run_async(task, done, fail)

    btn_puzzle_solve = ttk.Button(top_puzzle, text="Решить", command=solve_puzzle_gui)
    btn_puzzle_compare = ttk.Button(top_puzzle, text="Сравнить все", command=compare_puzzle_gui)
    btn_puzzle_random = ttk.Button(top_puzzle, text="Случайная", command=random_puzzle)
    btn_puzzle_reset = ttk.Button(top_puzzle, text="Сброс к эталону", command=reset_puzzle_goal)
    btn_puzzle_solve.pack(side="left", padx=4)
    btn_puzzle_compare.pack(side="left", padx=4)
    btn_puzzle_random.pack(side="left", padx=4)
    btn_puzzle_reset.pack(side="left", padx=4)

    puzzle_controls = [
        (btn_puzzle_solve, "normal"),
        (btn_puzzle_compare, "normal"),
        (btn_puzzle_random, "normal"),
        (btn_puzzle_reset, "normal"),
    ]

    for r in range(4):
        row_buttons = []
        for c in range(4):
            idx = r * 4 + c
            btn = tk.Button(
                grid_puzzle,
                text="",
                width=6,
                height=3,
                font=("Helvetica", 16, "bold"),
                command=lambda i=idx: click_tile(i),
            )
            btn.grid(row=r, column=c, padx=2, pady=2)
            row_buttons.append(btn)
        tile_buttons.append(row_buttons)

    reset_grid()
    random_puzzle()
    root.mainloop()


def main():
    run_gui()


if __name__ == "__main__":
    main()
