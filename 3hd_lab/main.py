import random
import statistics
import time
from collections import deque
import tkinter as tk
from tkinter import messagebox
from tkinter.scrolledtext import ScrolledText
import os
import subprocess

def make_list_node(value, next_node=None):
    return {"value": value, "next": next_node}


def build_sorted_linked_list(values):
    if not values:
        return None
    head = make_list_node(values[0])
    cur = head
    for value in values[1:]:
        cur["next"] = make_list_node(value)
        cur = cur["next"]
    return head


def linked_list_to_array(head):
    arr = []
    cur = head
    while cur is not None:
        arr.append(cur["value"])
        cur = cur["next"]
    return arr

def make_tree_node(value):
    return {"value": value, "left": None, "right": None, "idx": -1}


def find_middle(head):
    prev = None
    slow = head
    fast = head

    while fast is not None and fast["next"] is not None:
        prev = slow
        slow = slow["next"]
        fast = fast["next"]["next"]

    left_head = head
    right_head = slow["next"]

    if prev is None:
        left_head = None
    else:
        prev["next"] = None

    slow["next"] = None
    return left_head, slow, right_head


def list_to_bst_recursive_middle(head):
    if head is None:
        return None
    if head["next"] is None:
        return make_tree_node(head["value"])

    left_head, mid, right_head = find_middle(head)
    root = make_tree_node(mid["value"])
    root["left"] = list_to_bst_recursive_middle(left_head)
    root["right"] = list_to_bst_recursive_middle(right_head)
    return root

def make_avl_node(value):
    return {"value": value, "left": None, "right": None, "height": 1}


def avl_height(node):
    return node["height"] if node else 0


def avl_update_height(node):
    node["height"] = 1 + max(avl_height(node["left"]), avl_height(node["right"]))


def avl_balance_factor(node):
    return avl_height(node["left"]) - avl_height(node["right"])


def avl_rotate_right(y):
    x = y["left"]
    t2 = x["right"]

    x["right"] = y
    y["left"] = t2

    avl_update_height(y)
    avl_update_height(x)
    return x


def avl_rotate_left(x):
    y = x["right"]
    t2 = y["left"]

    y["left"] = x
    x["right"] = t2

    avl_update_height(x)
    avl_update_height(y)
    return y


def avl_insert(root, key):
    if root is None:
        return make_avl_node(key)

    if key < root["value"]:
        root["left"] = avl_insert(root["left"], key)
    else:

        root["right"] = avl_insert(root["right"], key)

    avl_update_height(root)
    balance = avl_balance_factor(root)

    if balance > 1 and root["left"] and key < root["left"]["value"]:
        return avl_rotate_right(root)

    if balance < -1 and root["right"] and key >= root["right"]["value"]:
        return avl_rotate_left(root)

    if balance > 1 and root["left"] and key >= root["left"]["value"]:
        root["left"] = avl_rotate_left(root["left"])
        return avl_rotate_right(root)

    if balance < -1 and root["right"] and key < root["right"]["value"]:
        root["right"] = avl_rotate_right(root["right"])
        return avl_rotate_left(root)

    return root


def build_avl_from_sorted_list(head):
    root = None
    cur = head
    while cur:
        root = avl_insert(root, cur["value"])
        cur = cur["next"]
    return root


def build_balanced_bst_from_array(arr, left, right):
    if left > right:
        return None
    mid = (left + right) // 2
    node = make_tree_node(arr[mid])
    node["left"] = build_balanced_bst_from_array(arr, left, mid - 1)
    node["right"] = build_balanced_bst_from_array(arr, mid + 1, right)
    return node


def assign_indices_and_collect(root):
    if root is None:
        return []

    nodes = []
    q = deque([root])

    idx = 0
    while q:
        node = q.popleft()
        node["idx"] = idx
        idx += 1
        nodes.append(node)

        if node["left"]:
            q.append(node["left"])
        if node["right"]:
            q.append(node["right"])

    return nodes


def preprocess_lca_binary_lifting(root):
    if root is None:
        return [], [], {}

    nodes = assign_indices_and_collect(root)
    n = len(nodes)
    log = max(1, n.bit_length())

    up = [[-1] * n for _ in range(log)]
    depth = [0] * n
    by_value = {}

    q = deque([(root, -1)])
    while q:
        node, parent_idx = q.popleft()
        i = node["idx"]
        up[0][i] = parent_idx
        by_value.setdefault(node["value"], []).append(i)

        if node["left"]:
            depth[node["left"]["idx"]] = depth[i] + 1
            q.append((node["left"], i))
        if node["right"]:
            depth[node["right"]["idx"]] = depth[i] + 1
            q.append((node["right"], i))

    for k in range(1, log):
        for v in range(n):
            p = up[k - 1][v]
            up[k][v] = -1 if p == -1 else up[k - 1][p]

    return up, depth, by_value

def build_bst_with_lca_precompute(head):
    arr = linked_list_to_array(head)
    root = build_balanced_bst_from_array(arr, 0, len(arr) - 1)
    up, depth, by_value = preprocess_lca_binary_lifting(root)
    nodes = assign_indices_and_collect(root)
    return root, up, depth, by_value, nodes


def make_rb_node(value, color):
    return {"value": value, "color": color, "left": None, "right": None, "parent": None}


def make_rb_tree():
    nil = make_rb_node(0, "B")
    nil["left"] = nil
    nil["right"] = nil
    nil["parent"] = nil
    return {"nil": nil, "root": nil}


def rb_left_rotate(tree, x):
    nil = tree["nil"]
    y = x["right"]

    x["right"] = y["left"]
    if y["left"] is not nil:
        y["left"]["parent"] = x

    y["parent"] = x["parent"]
    if x["parent"] is nil:
        tree["root"] = y
    elif x is x["parent"]["left"]:
        x["parent"]["left"] = y
    else:
        x["parent"]["right"] = y

    y["left"] = x
    x["parent"] = y


def rb_right_rotate(tree, y):
    nil = tree["nil"]
    x = y["left"]

    y["left"] = x["right"]
    if x["right"] is not nil:
        x["right"]["parent"] = y

    x["parent"] = y["parent"]
    if y["parent"] is nil:
        tree["root"] = x
    elif y is y["parent"]["right"]:
        y["parent"]["right"] = x
    else:
        y["parent"]["left"] = x

    x["right"] = y
    y["parent"] = x


def rb_insert_fixup(tree, z):
    nil = tree["nil"]

    while z["parent"]["color"] == "R":
        if z["parent"] is z["parent"]["parent"]["left"]:
            uncle = z["parent"]["parent"]["right"]
            if uncle["color"] == "R":
                z["parent"]["color"] = "B"
                uncle["color"] = "B"
                z["parent"]["parent"]["color"] = "R"
                z = z["parent"]["parent"]
            else:
                if z is z["parent"]["right"]:
                    z = z["parent"]
                    rb_left_rotate(tree, z)
                z["parent"]["color"] = "B"
                z["parent"]["parent"]["color"] = "R"
                rb_right_rotate(tree, z["parent"]["parent"])
        else:
            uncle = z["parent"]["parent"]["left"]
            if uncle["color"] == "R":
                z["parent"]["color"] = "B"
                uncle["color"] = "B"
                z["parent"]["parent"]["color"] = "R"
                z = z["parent"]["parent"]
            else:
                if z is z["parent"]["left"]:
                    z = z["parent"]
                    rb_right_rotate(tree, z)
                z["parent"]["color"] = "B"
                z["parent"]["parent"]["color"] = "R"
                rb_left_rotate(tree, z["parent"]["parent"])

    tree["root"]["color"] = "B"


def rb_insert(tree, key):
    nil = tree["nil"]
    z = make_rb_node(key, "R")
    z["left"] = nil
    z["right"] = nil

    y = nil
    x = tree["root"]

    while x is not nil:
        y = x
        if z["value"] < x["value"]:
            x = x["left"]
        else:
            x = x["right"]

    z["parent"] = y
    if y is nil:
        tree["root"] = z
    elif z["value"] < y["value"]:
        y["left"] = z
    else:
        y["right"] = z

    rb_insert_fixup(tree, z)


def build_rb_tree_from_sorted_list(head):
    arr = linked_list_to_array(head)
    if not arr:
        tree = make_rb_tree()
        return tree
    
    tree = make_rb_tree()
    nil = tree["nil"]
    
    for val in arr:
        rb_insert(tree, val)
    
    def add_nil_children_to_red_leaves(node):
        if node is nil:
            return
        
        is_leaf = node["left"] is nil and node["right"] is nil
        if is_leaf and node["color"] == "R":
            node["left"] = make_rb_node(0, "B")
            node["left"]["parent"] = node
            node["right"] = make_rb_node(0, "B")
            node["right"]["parent"] = node
            node["left"]["left"] = nil
            node["left"]["right"] = nil
            node["right"]["left"] = nil
            node["right"]["right"] = nil
        
        if node["left"] and node["left"] is not nil:
            add_nil_children_to_red_leaves(node["left"])
        if node["right"] and node["right"] is not nil:
            add_nil_children_to_red_leaves(node["right"])
    
    if tree["root"] is not nil:
        add_nil_children_to_red_leaves(tree["root"])
    
    return tree


def validate_rb_tree(tree):
    nil = tree["nil"]
    root = tree["root"]
    if root is nil:
        return True, "OK"
    if root["color"] != "B":
        return False, "Корень не черный"

    def dfs(node):
        if node is nil:
            return 1, True, ""
        
        is_fake_nil = node["value"] == 0
        if is_fake_nil:
            return 1, True, ""
        
        if node["color"] == "R":
            if node["left"]["color"] == "R" or node["right"]["color"] == "R":
                return 0, False, "У красного узла есть красный потомок"
        left_bh, left_ok, left_err = dfs(node["left"])
        if not left_ok:
            return 0, False, left_err
        right_bh, right_ok, right_err = dfs(node["right"])
        if not right_ok:
            return 0, False, right_err
        if left_bh != right_bh:
            return 0, False, "Нарушена черная высота"
        return left_bh + (1 if node["color"] == "B" else 0), True, ""

    _, ok, err = dfs(root)
    return ok, "OK" if ok else err


def height_tree(node):
    if node is None:
        return 0
    return 1 + max(height_tree(node["left"]), height_tree(node["right"]))


def height_avl(node):
    if node is None:
        return 0
    return 1 + max(height_avl(node["left"]), height_avl(node["right"]))


def height_rb_node(node, nil):
    if node is nil:
        return 0
    return 1 + max(height_rb_node(node["left"], nil), height_rb_node(node["right"], nil))


def inorder_tree(node, out):
    if node is None:
        return
    inorder_tree(node["left"], out)
    out.append(node["value"])
    inorder_tree(node["right"], out)


def inorder_avl(node, out):
    if node is None:
        return
    inorder_avl(node["left"], out)
    out.append(node["value"])
    inorder_avl(node["right"], out)


def inorder_rb(node, nil, out):
    if node is nil:
        return
    inorder_rb(node["left"], nil, out)
    out.append(node["value"])
    inorder_rb(node["right"], nil, out)


def tree_levels_lines(root):
    if root is None:
        return ["(пусто)"]

    lines = []
    q = deque([(root, 0)])
    current_level = 0
    current_values = []

    while q:
        node, level = q.popleft()
        if level != current_level:
            lines.append(f"L{current_level}: " + " ".join(current_values))
            current_values = []
            current_level = level

        current_values.append(str(node["value"]))

        if node["left"] is not None:
            q.append((node["left"], level + 1))
        if node["right"] is not None:
            q.append((node["right"], level + 1))

    lines.append(f"L{current_level}: " + " ".join(current_values))
    return lines


def rb_tree_levels_lines(tree):
    nil = tree["nil"]
    root = tree["root"]
    if root is nil:
        return ["(пусто)"]

    lines = []
    q = deque([(root, 0)])
    current_level = 0
    current_values = []

    while q:
        node, level = q.popleft()
        if level != current_level:
            lines.append(f"L{current_level}: " + " ".join(current_values))
            current_values = []
            current_level = level

        current_values.append(f"{node['value']}({node['color']})")

        if node["left"] is not nil:
            q.append((node["left"], level + 1))
        if node["right"] is not nil:
            q.append((node["right"], level + 1))

    lines.append(f"L{current_level}: " + " ".join(current_values))
    return lines


def tree_to_graphviz_dot(root, label_fn, color_fn=None, node_id_prefix=""):
    """Конвертирует дерево в DOT формат для graphviz"""
    if root is None:
        return "digraph { empty [label=\"(пусто)\"] }"

    dot_lines = ["digraph {"]
    dot_lines.append('  rankdir=TB;')
    dot_lines.append('  node [shape=circle, style=filled, fontname="Arial", fontsize=14];')

    node_counter = [0]

    def get_node_id():
        node_counter[0] += 1
        return f"{node_id_prefix}_{node_counter[0]}"

    def walk(node):
        if node is None:
            return None

        node_id = get_node_id()
        label = label_fn(node)


        if color_fn:
            fill_color = color_fn(node)
            text_color = "white" if fill_color in ["red", "darkred", "#CC0000"] else "black"
        else:
            fill_color = "lightblue"
            text_color = "black"

        dot_lines.append(f'  {node_id} [label="{label}", fillcolor="{fill_color}", fontcolor="{text_color}"];')


        left_child = node.get("left")
        right_child = node.get("right")

        if left_child is not None:
            child_id = walk(left_child)
            dot_lines.append(f'  {node_id} -> {child_id};')

        if right_child is not None:
            child_id = walk(right_child)
            dot_lines.append(f'  {node_id} -> {child_id};')

        return node_id

    walk(root)
    dot_lines.append("}")
    return "\n".join(dot_lines)


def rb_tree_to_graphviz_dot(tree, node_id_prefix=""):
    """Конвертирует красно-черное дерево в DOT формат"""
    nil = tree["nil"]
    root = tree["root"]

    if root is nil:
        return "digraph { empty [label=\"(пусто)\"] }"

    dot_lines = ["digraph {"]
    dot_lines.append('  rankdir=TB;')
    dot_lines.append('  node [shape=circle, style=filled, fontname="Arial", fontsize=14];')

    node_counter = [0]

    def get_node_id():
        node_counter[0] += 1
        return f"{node_id_prefix}_{node_counter[0]}"

    def walk(node):
        if node is nil:
            return None

        node_id = get_node_id()
        label = str(node["value"])


        fill_color = "#FF6B6B" if node["color"] == "R" else "#2C3E50"
        text_color = "white"

        dot_lines.append(f'  {node_id} [label="{label}", fillcolor="{fill_color}", fontcolor="{text_color}"];')

        if node["left"] is not nil:
            child_id = walk(node["left"])
            dot_lines.append(f'  {node_id} -> {child_id};')

        if node["right"] is not nil:
            child_id = walk(node["right"])
            dot_lines.append(f'  {node_id} -> {child_id};')

        return node_id

    walk(root)
    dot_lines.append("}")
    return "\n".join(dot_lines)


def render_tree_to_image(dot_content, output_path):
    """Рендерит DOT в PNG с помощью graphviz"""
    try:

        temp_dot = output_path.replace('.png', '.dot')
        with open(temp_dot, 'w') as f:
            f.write(dot_content)

        subprocess.run(['dot', '-Tpng', temp_dot, '-o', output_path],
                      check=True, capture_output=True)

        if os.path.exists(temp_dot):
            os.remove(temp_dot)

        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def pretty_print_tree(root, label_fn, colorize_fn=None, max_width=100):
    """Функция для красивого вывода дерева в текст с ASCII art и Unicode"""
    if root is None:
        return ["┌─ (пусто)"]

    lines = []

    def walk(node, prefix="", is_last=True, depth=0):
        if node is None:
            return

        label = label_fn(node)
        if colorize_fn:
            label = colorize_fn(node, label)
        else:
            label = f"【{label}】"


        connector = "└─ " if is_last else "├─ "
        if depth == 0:
            connector = ""

        lines.append(prefix + connector + label)


        children = []
        if node.get("left") is not None:
            children.append(node["left"])
        if node.get("right") is not None:
            children.append(node["right"])


        for i, child in enumerate(children):
            is_last_child = i == len(children) - 1
            if depth == 0:
                new_prefix = ""
            else:
                new_prefix = prefix + ("   " if is_last else "│  ")
            walk(child, new_prefix, is_last_child, depth + 1)

    walk(root)
    return lines if lines else ["(пусто)"]


def pretty_print_rb_tree(tree):
    """Красивый вывод красно-черного дерева с цветами"""
    nil = tree["nil"]
    root = tree["root"]

    if root is nil:
        return ["(пусто)"]

    lines = []

    def colorize_rb(node, label):

        if node["color"] == "R":
            return f"[{label}]"
        else:
            return f"[{label}]"

    def walk(node, prefix="", is_last=True, depth=0):
        if node is nil:
            return

        label = str(node["value"])
        label = colorize_rb(node, label)

        connector = "└─ " if is_last else "├─ "
        if depth == 0:
            connector = ""

        lines.append(prefix + connector + label)


        children = []
        if node["left"] is not nil:
            children.append(node["left"])
        if node["right"] is not nil:
            children.append(node["right"])

        for i, child in enumerate(children):
            is_last_child = i == len(children) - 1
            if depth == 0:
                new_prefix = ""
            else:
                new_prefix = prefix + ("   " if is_last else "│  ")
            walk(child, new_prefix, is_last_child, depth + 1)

    walk(root)
    return lines if lines else ["(пусто)"]


def generate_sorted_data(size, distribution):
    if distribution == "uniform_unique":
        arr = random.sample(range(size * 20), size)
    elif distribution == "with_duplicates":
        arr = [random.randint(0, max(1, size // 3)) for _ in range(size)]
    elif distribution == "clustered":
        arr = []
        for _ in range(size):
            if random.random() < 0.8:
                arr.append(random.randint(0, max(1, size // 10)))
            else:
                arr.append(random.randint(size * 2, size * 3))
    else:
        raise ValueError(f"Неизвестное распределение: {distribution}")

    arr.sort()
    return arr


def benchmark_once(values):
    head = build_sorted_linked_list(values)

    t0 = time.perf_counter()
    base_root = list_to_bst_recursive_middle(head)
    t1 = time.perf_counter()

    head2 = build_sorted_linked_list(values)
    t2 = time.perf_counter()
    avl_root = build_avl_from_sorted_list(head2)
    t3 = time.perf_counter()

    head3 = build_sorted_linked_list(values)
    t4 = time.perf_counter()
    lca_root, up, depth, _, _ = build_bst_with_lca_precompute(head3)
    t5 = time.perf_counter()

    head4 = build_sorted_linked_list(values)
    t6 = time.perf_counter()
    rb_tree = build_rb_tree_from_sorted_list(head4)
    t7 = time.perf_counter()

    return {
        "base_recursive_ms": (t1 - t0) * 1000,
        "avl_ms": (t3 - t2) * 1000,
        "lca_precompute_ms": (t5 - t4) * 1000,
        "rb_tree_ms": (t7 - t6) * 1000,
        "base_h": float(height_tree(base_root)),
        "avl_h": float(height_avl(avl_root)),
        "lca_h": float(height_tree(lca_root)),
        "rb_h": float(height_rb_node(rb_tree["root"], rb_tree["nil"])),
        "lca_table_rows": float(len(up)),
        "lca_nodes": float(len(depth)),
    }


def build_benchmark_report_lines(show_visualization=True, output_dir="tree_visualizations"):
    random.seed(42)

    sizes = [10, 100, 1000]
    distributions = ["uniform_unique", "with_duplicates", "clustered"]
    repeats = 7
    distribution_names = {
        "uniform_unique": "Равномерное, уникальные значения",
        "with_duplicates": "С повторами",
        "clustered": "Кластеризованное",
    }

    if show_visualization:
        os.makedirs(output_dir, exist_ok=True)

    lines = []
    lines.append("Задача: отсортированный односвязный список -> сбалансированное дерево поиска")
    lines.append("Алгоритмы: базовый (рекурсивный с поиском середины), AVL, LCA с предподсчетом, красно-черное дерево")
    lines.append("")

    for distribution in distributions:
        lines.append("=" * 70)
        lines.append(f" Распределение: {distribution_names[distribution]}")
        lines.append("=" * 70)

        for size in sizes:
            visualization_values = None
            all_runs = []
            for run_idx in range(repeats):
                values = generate_sorted_data(size, distribution)
                if run_idx == 0:
                    visualization_values = values[:]
                all_runs.append(benchmark_once(values))

            def avg(metric):
                return statistics.mean(run[metric] for run in all_runs)

            lines.append("")
            lines.append(f"Размер: n = {size}")
            lines.append("-" * 70)
            lines.append(
                f"Время:     базовый={avg('base_recursive_ms'):7.3f}мс | "
                f"avl={avg('avl_ms'):7.3f}мс | "
                f"lca={avg('lca_precompute_ms'):7.3f}мс | "
                f"кчд={avg('rb_tree_ms'):7.3f}мс"
            )
            lines.append(
                f"Высоты:    базовый={avg('base_h'):5.1f} | "
                f"avl={avg('avl_h'):5.1f} | "
                f"lca={avg('lca_h'):5.1f} | "
                f"кчд={avg('rb_h'):5.1f}"
            )

            if show_visualization and visualization_values is not None:
                lines.append("")
                lines.append(f"ДЕРЕВЬЯ для n={size} ({distribution_names[distribution]}):")
                lines.append("-" * 70)

                lines.append("  Базовое (середина):")
                base_root = list_to_bst_recursive_middle(build_sorted_linked_list(visualization_values))
                for tree_line in pretty_print_tree(base_root, lambda n: str(n["value"]), max_width=60):
                    lines.append(f"    {tree_line}")

                lines.append("")
                lines.append("  AVL (сбалансированное):")
                avl_root = build_avl_from_sorted_list(build_sorted_linked_list(visualization_values))
                for tree_line in pretty_print_tree(avl_root, lambda n: str(n["value"]), max_width=60):
                    lines.append(f"    {tree_line}")

                lines.append("")
                lines.append("  LCA (предподсчет):")
                lca_root, _, _, _, _ = build_bst_with_lca_precompute(build_sorted_linked_list(visualization_values))
                for tree_line in pretty_print_tree(lca_root, lambda n: str(n["value"]), max_width=60):
                    lines.append(f"    {tree_line}")

                lines.append("")
                lines.append("  Красно-черное (красные и черные):")
                rb_tree = build_rb_tree_from_sorted_list(build_sorted_linked_list(visualization_values))
                ok, msg = validate_rb_tree(rb_tree)
                lines.append(f"  Проверка КЧД: {msg}")
                for tree_line in pretty_print_rb_tree(rb_tree):
                    lines.append(f"    {tree_line}")

    return lines


def run_benchmark():
    print()
    for line in build_benchmark_report_lines():
        print(line)

def make_inorder_summary_line(values, algo_name):
    show = values[:20]
    suffix = " ..." if len(values) > 20 else ""
    return f"{algo_name} (симметричный обход, первые 20): {show}{suffix}"


def build_demo_report_lines(values, show_visualization=True, output_dir="tree_visualizations"):
    values = sorted(values)
    lines = []

    if show_visualization:
        os.makedirs(output_dir, exist_ok=True)

    head = build_sorted_linked_list(values)
    base_root = list_to_bst_recursive_middle(head)
    out_base = []
    inorder_tree(base_root, out_base)
    lines.append(make_inorder_summary_line(out_base, "Базовый"))
    lines.append(f"Высота (базовый): {height_tree(base_root)}")

    head2 = build_sorted_linked_list(values)
    avl_root = build_avl_from_sorted_list(head2)
    out_avl = []
    inorder_avl(avl_root, out_avl)
    lines.append(make_inorder_summary_line(out_avl, "AVL"))
    lines.append(f"Высота (AVL): {height_avl(avl_root)}")

    head3 = build_sorted_linked_list(values)
    lca_root, up, depth, by_value, nodes = build_bst_with_lca_precompute(head3)
    out_lca = []
    inorder_tree(lca_root, out_lca)
    lines.append(make_inorder_summary_line(out_lca, "LCA с предподсчетом"))
    lines.append(f"Высота (LCA): {height_tree(lca_root)}")

    head4 = build_sorted_linked_list(values)
    rb_tree = build_rb_tree_from_sorted_list(head4)
    out_rb = []
    inorder_rb(rb_tree["root"], rb_tree["nil"], out_rb)
    lines.append(make_inorder_summary_line(out_rb, "Красно-черное дерево"))
    lines.append(f"Высота (красно-черное): {height_rb_node(rb_tree['root'], rb_tree['nil'])}")
    ok, msg = validate_rb_tree(rb_tree)
    lines.append(f"Проверка КЧД: {msg}")

    if show_visualization:
        lines.append("")
        lines.append("═" * 60)
        lines.append("ВИЗУАЛИЗАЦИЯ ДЕРЕВЬЕВ")
        lines.append("═" * 60)


        lines.append("")
        lines.append(" Базовое дерево (бинарный поиск по середине):")
        lines.append("─" * 50)
        for tree_line in pretty_print_tree(base_root, lambda n: str(n["value"])):
            lines.append("  " + tree_line)


        lines.append("")
        lines.append("AVL дерево (сбалансированное):")
        lines.append("─" * 50)
        for tree_line in pretty_print_tree(avl_root, lambda n: str(n["value"])):
            lines.append("  " + tree_line)


        lines.append("")
        lines.append("LCA дерево (с предподсчетом):")
        lines.append("─" * 50)
        for tree_line in pretty_print_tree(lca_root, lambda n: str(n["value"])):
            lines.append("  " + tree_line)


        lines.append("")
        lines.append("  Красно-черное дерево (красные/черные узлы):")
        lines.append("─" * 50)
        for tree_line in pretty_print_rb_tree(rb_tree):
            lines.append("  " + tree_line)

        lines.append("═" * 60)


    if up and nodes:
        lines.append("")

    return lines, by_value, nodes, up, depth


def tree_compute_layout(root, nil=None):
    """Вычисляет координаты узлов дерева для рисования с лучшим алгоритмом"""
    if root is None or (nil is not None and root is nil):
        return {}, {}

    positions = {}
    counter = [0]

    def get_height(node):
        """Вычисляет высоту поддерева"""
        if node is None or (nil is not None and node is nil):
            return 0
        return 1 + max(get_height(node.get("left")), get_height(node.get("right")))

    def compute_positions(node, x, y, offset):
        """Вычисляет позиции с учетом высоты поддеревьев"""
        if node is None or (nil is not None and node is nil):
            return

        positions[id(node)] = (x, y)

        left = node.get("left")
        right = node.get("right")

        if left or right:

            left_height = get_height(left) if left else 0
            right_height = get_height(right) if right else 0


            new_y = y + 70
            next_offset = max(offset // 1.8, 25)

            if left:
                left_x = x - offset
                compute_positions(left, left_x, new_y, next_offset)

            if right:
                right_x = x + offset
                compute_positions(right, right_x, new_y, next_offset)

    height = get_height(root)
    initial_offset = max(50, 80 - height * 8)

    compute_positions(root, 0, 20, initial_offset)
    return positions, {}


def draw_tree_on_canvas(canvas, root, nil=None, node_color=None, draw_nil=False, nil_color="#000000", nil_radius=5):
    canvas.delete("all")

    if root is None or (nil is not None and root is nil):
        canvas.create_text(canvas.winfo_width()//2, canvas.winfo_height()//2,
                          text="(пусто)", font=("Arial", 12), fill="gray")
        return

    canvas_width = max(canvas.winfo_width(), 250)
    canvas_height = max(canvas.winfo_height(), 350)

    positions, _ = tree_compute_layout(root, nil)

    if not positions:
        canvas.create_text(canvas.winfo_width()//2, canvas.winfo_height()//2,
                          text="Ошибка при расчете позиций", font=("Arial", 10), fill="red")
        return


    xs = [pos[0] for pos in positions.values()]
    ys = [pos[1] for pos in positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)


    width_range = max_x - min_x if max_x > min_x else 1
    height_range = max_y - min_y if max_y > min_y else 1

    padding_x = 40
    padding_y = 30

    scale_x = (canvas_width - 2 * padding_x) / width_range if width_range > 0 else 1
    scale_y = (canvas_height - 2 * padding_y) / height_range if height_range > 0 else 1
    scale = min(scale_x, scale_y, 2.5)

    def normalize_pos(pos):
        x, y = pos
        norm_x = (x - min_x) * scale + padding_x
        norm_y = (y - min_y) * scale + padding_y
        return norm_x, norm_y


    def draw_edges(node):
        if node is None or (nil is not None and node is nil):
            return

        if id(node) not in positions:
            return

        node_pos = normalize_pos(positions[id(node)])

        left_child = node.get("left")
        right_child = node.get("right")

        for child in [left_child, right_child]:
            if child is not None and (nil is None or child is not nil):
                if id(child) in positions:
                    child_pos = normalize_pos(positions[id(child)])
                    canvas.create_line(node_pos[0], node_pos[1], child_pos[0], child_pos[1],
                                     width=2, fill="#666666")
                draw_edges(child)

    draw_edges(root)


    def draw_nodes(node):
        if node is None or (nil is not None and node is nil):
            return

        if id(node) not in positions:
            return

        x, y = normalize_pos(positions[id(node)])
        radius = 14


        if node_color is not None:
            try:
                c = node_color(node)
            except:
                c = "#87CEEB"
        else:
            c = "#87CEEB"

        if isinstance(c, tuple):
            color, text_color = c[0], c[1]
        else:
            color = c

            try:
                hexc = color.lstrip('#')
                r = int(hexc[0:2], 16)
                g = int(hexc[2:4], 16)
                b = int(hexc[4:6], 16)
                luminance = (0.299 * r + 0.587 * g + 0.114 * b)
                text_color = '#000000' if luminance > 140 else '#FFFFFF'
            except Exception:
                text_color = '#000000'


        outline_color = "#333333"
        outline_width = 2
        node_radius = radius

        if isinstance(c, tuple) and c[0] == "#FF4D4D":
            outline_color = "#FF0000"
            outline_width = 3
            node_radius = radius + 2


        text = str(node["value"])

        font_size = 9 if (isinstance(c, tuple) and c[0] == "#FF4D4D") else 8
        
        is_nil_visual = node["value"] == 0
        
        if not is_nil_visual:
            canvas.create_oval(x - node_radius - 1, y - node_radius - 1, x + node_radius + 1, y + node_radius + 1,
                              fill="#DDDDDD", outline="")
            canvas.create_oval(x - node_radius, y - node_radius, x + node_radius, y + node_radius,
                              fill=color, outline=outline_color, width=outline_width)
            canvas.create_text(x, y, text=text, font=("Arial", font_size, "bold"), fill=text_color)
        else:
            canvas.create_oval(x - 4, y - 4, x + 4, y + 4,
                              fill=color, outline=outline_color, width=1)


        for child in [node.get("left"), node.get("right")]:
            if child is not None and (nil is None or child is not nil):
                draw_nodes(child)


    draw_nodes(root)


def draw_rb_tree_on_canvas(canvas, tree):
    """Рисует красно-черное дерево на Canvas с цветными кружочками"""
    nil = tree["nil"]
    root = tree["root"]

    def color_fn(node):
        if node["color"] == "R":
            return ("#FF4D4D", "#FFFFFF")
        else:
            return ("#1E1E1E", "#FFFFFF")

    draw_tree_on_canvas(canvas, root, nil, color_fn)


def draw_default_trees(canvases):
    """Рисует деревья по умолчанию с примером"""
    values = [1, 2, 3, 4, 5, 6, 7]

    head1 = build_sorted_linked_list(values)
    base_root = list_to_bst_recursive_middle(head1)
    draw_tree_on_canvas(canvases[0], base_root, nil=None, node_color=lambda n: "#87CEEB")

    head2 = build_sorted_linked_list(values)
    avl_root = build_avl_from_sorted_list(head2)
    draw_tree_on_canvas(canvases[1], avl_root, nil=None, node_color=lambda n: "#90EE90")

    head3 = build_sorted_linked_list(values)
    lca_root, _, _, _, _ = build_bst_with_lca_precompute(head3)
    draw_tree_on_canvas(canvases[2], lca_root, nil=None, node_color=lambda n: "#FFE4B5")

    head4 = build_sorted_linked_list(values)
    rb_tree = build_rb_tree_from_sorted_list(head4)
    draw_rb_tree_on_canvas(canvases[3], rb_tree)


def run_gui():
    root = tk.Tk()
    root.title("Лабораторная 5: с список -> дерево")
    root.geometry("1400x800")


    top = tk.Frame(root)
    top.pack(fill="x", padx=10, pady=10)

    tk.Label(top, text="Числа через пробел:").pack(side="left")
    entry = tk.Entry(top, width=40)
    entry.pack(side="left", fill="x", expand=True, padx=8)
    entry.insert(0, "1 2 3 4 5 6 7 8 9")

    show_vis_var = tk.BooleanVar(value=False)
    


    try:
        from tkinter import ttk
        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)


        text_frame = tk.Frame(notebook)
        notebook.add(text_frame, text="Отчет")

        output = ScrolledText(text_frame, wrap="word", font=("Menlo", 10))
        output.pack(fill="both", expand=True)


        tree_frame = tk.Frame(notebook)
        notebook.add(tree_frame, text="Деревья")

        canvas_frame_1 = tk.Frame(tree_frame)
        canvas_frame_1.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        canvas_frame_2 = tk.Frame(tree_frame)
        canvas_frame_2.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        canvas_frame_3 = tk.Frame(tree_frame)
        canvas_frame_3.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        canvas_frame_4 = tk.Frame(tree_frame)
        canvas_frame_4.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        tk.Label(canvas_frame_1, text="Базовое", font=("Arial", 10, "bold")).pack()
        canvas1 = tk.Canvas(canvas_frame_1, bg="white", width=250, height=350)
        canvas1.pack(fill="both", expand=True)
        canvas1.create_text(125, 175, text=" Загрузка...", font=("Arial", 14), fill="gray")

        tk.Label(canvas_frame_2, text="AVL", font=("Arial", 10, "bold")).pack()
        canvas2 = tk.Canvas(canvas_frame_2, bg="white", width=250, height=350)
        canvas2.pack(fill="both", expand=True)
        canvas2.create_text(125, 175, text=" Загрузка...", font=("Arial", 14), fill="gray")

        tk.Label(canvas_frame_3, text="LCA", font=("Arial", 10, "bold")).pack()
        canvas3 = tk.Canvas(canvas_frame_3, bg="white", width=250, height=350)
        canvas3.pack(fill="both", expand=True)
        canvas3.create_text(125, 175, text=" Загрузка...", font=("Arial", 14), fill="gray")

        tk.Label(canvas_frame_4, text="Красно-черное", font=("Arial", 10, "bold")).pack()
        canvas4 = tk.Canvas(canvas_frame_4, bg="white", width=250, height=350)
        canvas4.pack(fill="both", expand=True)
        canvas4.create_text(125, 175, text=" Загрузка...", font=("Arial", 14), fill="gray")

        canvases = [canvas1, canvas2, canvas3, canvas4]
    except ImportError:

        output = ScrolledText(root, wrap="word", font=("Menlo", 10))
        output.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        canvases = []

    def set_output(lines):
        output.delete("1.0", tk.END)
        output.insert(tk.END, "\n".join(lines) + "\n")

    def clear_canvases():
        for canvas in canvases:
            canvas.delete("all")

    def on_run_demo():
        raw = entry.get().strip()
        if not raw:
            messagebox.showwarning("Ввод", "Введите хотя бы одно число.")
            return
        try:
            values = [int(x) for x in raw.split()]
        except ValueError:
            messagebox.showerror("Ошибка", "Вводите только целые числа.")
            return

        lines, _, _, _, _ = build_demo_report_lines(values, show_visualization=show_vis_var.get())
        set_output(lines)

        if canvases:
            clear_canvases()
            root.update_idletasks()
            values = sorted(values)

            head1 = build_sorted_linked_list(values)
            base_root = list_to_bst_recursive_middle(head1)
            draw_tree_on_canvas(canvases[0], base_root, nil=None, node_color=lambda n: "#87CEEB")

            head2 = build_sorted_linked_list(values)
            avl_root = build_avl_from_sorted_list(head2)
            draw_tree_on_canvas(canvases[1], avl_root, nil=None, node_color=lambda n: "#90EE90")

            head3 = build_sorted_linked_list(values)
            lca_root, _, _, _, _ = build_bst_with_lca_precompute(head3)
            draw_tree_on_canvas(canvases[2], lca_root, nil=None, node_color=lambda n: "#FFE4B5")

            head4 = build_sorted_linked_list(values)
            rb_tree = build_rb_tree_from_sorted_list(head4)
            draw_rb_tree_on_canvas(canvases[3], rb_tree)

    def on_run_benchmark():
        root.update_idletasks()
        lines = build_benchmark_report_lines(show_visualization=show_vis_var.get())
        set_output(lines)

        if canvases:
            clear_canvases()
            root.update_idletasks()

            import random as _random
            _random.seed(42)
            values = _random.sample(range(10 * 20), 10)
            values = sorted(values)

            head1 = build_sorted_linked_list(values)
            base_root = list_to_bst_recursive_middle(head1)
            draw_tree_on_canvas(canvases[0], base_root, nil=None, node_color=lambda n: "#87CEEB")

            head2 = build_sorted_linked_list(values)
            avl_root = build_avl_from_sorted_list(head2)
            draw_tree_on_canvas(canvases[1], avl_root, nil=None, node_color=lambda n: "#90EE90")

            head3 = build_sorted_linked_list(values)
            lca_root, _, _, _, _ = build_bst_with_lca_precompute(head3)
            draw_tree_on_canvas(canvases[2], lca_root, nil=None, node_color=lambda n: "#FFE4B5")

            head4 = build_sorted_linked_list(values)
            rb_tree = build_rb_tree_from_sorted_list(head4)
            draw_rb_tree_on_canvas(canvases[3], rb_tree)


    buttons = tk.Frame(root)
    buttons.pack(fill="x", padx=10, pady=(0, 8))
    tk.Button(buttons, text="Построить и сравнить", command=on_run_demo, bg="#87CEEB", font=("Arial", 10)).pack(side="left", padx=5)
    tk.Button(buttons, text="Запустить бенчмарк", command=on_run_benchmark, bg="#FFB6C1", font=("Arial", 10)).pack(side="left", padx=5)

    set_output([

        " Заполните числа выше и нажимайте кнопки ",
        "'Построить и сравнить' для демонстрации ",
        "  Или 'Запустить бенчмарк' для сравнения.  "

    ])


    if canvases:
        root.after(500, lambda: draw_default_trees(canvases))

    root.mainloop()

if __name__ == "__main__":
    run_gui()
