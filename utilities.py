import re
from os import PathLike, system
from sys import platform
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from plotly import graph_objects as go


def read_arff(file_name: str | PathLike) -> tuple[pd.DataFrame, str]:
    """
    this function reads an arff file and returns a pandas dataframe
    the 'arff' file must be in the following format:
    ```
    @relation <relation_name>

    @attribute <attribute_name1> {<value1>, <value2>, ...}
    @attribute <attribute_name2> {<value1>, <value2>, ...}
    ...
    @attribute <attribute_nameN> {<value1>, <value2>, ...}

    @data
    <value1>, <value2>, ..., <valueN>
    <value1>, <value2>, ..., <valueN>
    ...
    ```

    to do so, I used regular expressions to find the attributes and the relation name,
    and then I used pd.read_csv to read the data
    :param file_name:
    :return:
    """
    attribute_exp = re.compile("^@attribute ([A-Za-z][A-Za-z0-9]+) {[0-9,]+}")
    relation_exp = re.compile("^@relation ([a-zA-Z0-9.]+)")
    data_exp = re.compile("^@data")
    col_names = []
    relation_name = None
    file = open(file_name)
    line_num = 0
    for line_num, line in enumerate(file):
        if t := relation_exp.match(line):
            relation_name = t.groups()[0]
        elif t := attribute_exp.match(line):
            col_names.append(t.groups()[0])
        elif data_exp.match(line):
            break
    file.close()
    return pd.read_csv(file_name, skiprows=line_num + 1, names=col_names), relation_name


class Node:
    def __init__(self, data, cost_func: Callable[[Iterable], float] = None, class_col="class", parent=None):
        """
        this class represents a node in the decision tree

        :param data: the data that this node represents
        :param cost_func: is a function that takes a list of values and returns a number
        :param class_col:
        :param parent:
        """
        if cost_func is None:
            cost_func = self.entropy
        self.children: dict[int, Node] = {}  # {value: Node}
        self.data = data
        self.class_col = class_col
        if parent:
            self.depth = parent.depth + 1
        else:
            self.depth = 0
            self.position = 0  # this will be used to draw the tree

        if len(data.columns) == 1:
            self.is_leaf = True
            self.output = np.argmax(np.bincount(data[class_col]))
            return
        if np.unique(data[class_col]).shape[0] == 1:
            self.is_leaf = True
            self.output = np.argmax(np.bincount(data[class_col]))
            return

        self.is_leaf = False
        # error = cost_func(data[class_col])

        all_costs = {}
        for col in [i for i in data.columns if i != class_col]:
            class_prob = np.bincount(data[col], ) / len(data)
            current_cost = sum([class_prob[i] * cost_func(data.loc[data[col] == i, :][class_col])
                                for i in range(len(class_prob))])
            all_costs[col] = current_cost

        self.all_costs = all_costs
        self.split_feature = min(all_costs, key=lambda x: all_costs[x])
        self.children = {
            i: Node(data.loc[data[self.split_feature] == i, :].drop(columns=[self.split_feature]), cost_func,
                    parent=self) for i in np.unique(data[self.split_feature])}

    def to_tikz(self):
        return (r"\documentclass[convert]{standalone}" "\n"
                r"\usepackage{tree-dvips}" "\n"
                r"\usepackage{qtree}" "\n"
                r"\begin{document}" "\n"
                r"\Tree"
                f"{self.to_tikz_()}\n"
                r"\end{document}")

    def to_tikz_(self):
        if self.is_leaf:
            return str(self.output)
        return f"[.{self.split_feature} {' '.join(child.to_tikz_() for _, child in self.children.items())} ]"

    def predict_one_datapoint(self, x):
        if self.is_leaf:
            return self.output
        #         return self.childs[x[self.split_feature]].predict_one_dataPoint(x)
        try:
            return self[x[self.split_feature]].predict_one_datapoint(x)
        except KeyError:
            return np.argmax(np.bincount(self.data[self.class_col]))

    # def predict(self, x):
    #     x_clone = x.copy()
    #     x_clone['class'] = None
    #     for i in x_clone.index:
    #         x_clone.loc[i, 'class'] = self.predict_one_datapoint(x.loc[i, :])
    #     return x_clone

    def predict(self, x) -> list:
        classes = []
        for i in x.index:
            classes.append(self.predict_one_datapoint(x.loc[i, :]))
        return classes

    def accuracy(self, x):
        return (x['class'] == self.predict(x)).astype(int).sum() / len(x)

    def __getitem__(self, key):
        if key not in self.children:
            raise KeyError(f"split_feature:{self.split_feature}\n"
                           f"and available keys:{list(self.children.keys())}\n"
                           f"requested key is {key}")
        return self.children[key]

    def __repr__(self):
        return f"<{self.split_feature}:({self.position}, {self.depth})>"

    @staticmethod
    def entropy(data, base=2):
        """
        this function calculates the entropy of a given data,
        the data must be one dimensional and categorical
        the formula is:
        H(X) = -sum(p(x)log(p(x)))
        p(x) is the probability of x

        :param data:
        :param base:
        :return:
        """
        if base <= 1:
            raise ValueError("base must be greater than 1")
        t = np.bincount(data) / len(data)
        t = t[t > 0]
        if base == 2:
            return -np.sum(t * np.log2(t))
        return -np.sum(t * np.log(t) / np.log(base))


def draw(root, title=""):
    """
    this function uses breadth-first traverse to find all nodes, And uses plotly to draw the tree.

    :param root:
    :param title:
    :return:
    """
    my_queue: list[Node] = [root]
    x_lines = []
    y_lines = []
    nodes_pos: dict[(int, int), Node] = {(0, 0): my_queue[0]}
    max_depth = 0
    all_x = []
    all_y = []
    while my_queue:
        current_node = my_queue.pop(0)
        max_depth += current_node.depth
        tmp_mean = (len(current_node.children) - 1) / 2
        alter_pos = [i - tmp_mean for i in range(len(current_node.children))][::-1]
        for index, child_node in current_node.children.items():
            my_queue.append(child_node)
            child_node.position = current_node.position + alter_pos.pop(-1) / (0.1 * child_node.depth ** 2)
            nodes_pos.update({(child_node.position, -child_node.depth): child_node})
            all_x.append(child_node.position)
            all_y.append(-child_node.depth)
            x_lines += [current_node.position, child_node.position, None]
            y_lines += [-current_node.depth, -child_node.depth, None]
    fig = go.Figure()
    # mean = lambda x: sum(x) / len(x)
    x_mean = np.mean(all_x)
    y_mean = np.mean(all_y)
    #     R = 0.1*(x_mean+y_mean)
    #     R = 2/len(nodes_pos)
    R = len(nodes_pos) / max_depth
    fig.add_trace(go.Scatter(x=x_lines, y=y_lines,
                             mode='lines', hoverinfo='none',
                             line=dict(color='rgb(100,100,100)', width=2)))

    for (x, y), node in nodes_pos.items():
        fig.add_shape(type="circle", x0=x - R, y0=y - R, x1=x + R, y1=y + R,
                      line_color="#f0057a", fillcolor="#f595c5", layer='below')
        if node.is_leaf:
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                text=[node.output],
                mode="text",
                hoverinfo="name",
                name="",
                textfont=dict(
                    color="black",
                    size=18,
                    family="Arail")))
        else:
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                text=[nodes_pos[(x, y)].split_feature],
                mode="text",
                hovertemplate=
                f'<i>Split feature</i>: {node.split_feature}' +
                f"<br>possible classes{list(node.children.keys())}<br>"
                f'cost Values:<br>' +
                "<br>".join(f"<b>{i}</b> :{j:4.4f}" for i, j in node.all_costs.items()) + "" +
                f"<br>depth: {node.depth}",
                name="",
                textfont=dict(
                    color="black",
                    size=18,
                    family="Arail")))
    axis = {'showline': False, 'zeroline': False, 'showgrid': False, 'showticklabels': False}

    fig.update_layout(title=title,
                      font_size=12,
                      showlegend=False,
                      xaxis=axis,
                      yaxis=axis,
                      margin=dict(l=40, r=40, b=85, t=100),
                      hovermode='closest',
                      plot_bgcolor='#eeeeee'
                      )
    # R = max(abs(min(all_x) - max(all_x)), abs(min(all_y) - max(all_y))) / 2 * 1.5

    # fig.update_xaxes(range=[x_mean - R, x_mean + R], zeroline=False)
    # fig.update_yaxes(range=[y_mean - R, y_mean + R])

    c = 0.5 * 1.5
    width = max(all_x) - min(all_x)
    height = max(all_y) - min(all_y)
    width = int(max(width * c, 10))
    height = int(max(height * c, 10))
    margin = 2
    fig.update_xaxes(range=[min(all_x) - margin, max(all_x) + margin], scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[min(all_y) - margin, max(all_y) + margin], scaleanchor="x", scaleratio=1)

    fig.update_layout(
        hoverlabel=dict(
            bgcolor="#b567c9",
            font_size=16,
            font_family="Rockwell"
        )
    )
    fig.show()


def create_pdf(node: Node, filename):
    if platform != "linux":
        raise OSError("this function is not supported in this machine :(")
    with open(f"{filename}.tex", 'w') as f:
        f.write(node.to_tikz())
    system("ls qtree.sty || wget https://www.ling.upenn.edu/advice/latex/qtree/qtree.sty")
    system(
        "ls tree-dvips.sty || "
        "wget http://ctan.math.washington.edu/tex-archive/macros/latex209/contrib/trees/tree-dvips/tree-dvips.sty")
    # note: we used || because if the file exists, it will not download it again
    # to install pdflatex: sudo apt-get install texlive-latex-base

    system(f"pdflatex {filename}.tex")

    print(f"output is in {filename}.pdf")
    system(f"xdg-open {filename}.pdf")
    t = ["aux", "bbl", "blg", "idx", "ind", "lof",
         "lot", "out", "toc", "acn", "acr", "alg",
         "glg", "glo", "gls", "ist", "fls", "log",
         "fdb_latexmk", "tex", "gz"]
    for i in t:
        try:
            system(f"rm *.{i}")
        except Exception as e:
            print("there was an error in deleting files {i}, error is\n{e}")


class NodePruned:
    to_tikz = Node.to_tikz
    to_tikz_ = Node.to_tikz_
    predict = Node.predict
    accuracy = Node.accuracy
    __getitem__ = Node.__getitem__
    __repr__ = Node.__repr__
    # NOTE: we cant use inheritance because we don't want to run the Node.__init__ function

    children: dict[int, 'NodePruned']

    def __init__(self, data, cost_func=None, class_col="class", parent=None, max_cost=None, min_depth=None):
        if cost_func is None:
            cost_func = Node.entropy
        self.output = None
        self.is_leaf = False
        self.children = {}  # {value: Node}
        self.data = data
        self.class_col = class_col
        if parent:
            self.depth = parent.depth + 1
        #             print(self.depth)
        else:
            self.depth = 0
            self.position = 0  # this will be used to draw the tree

        if len(data.columns) == 1:
            self.is_leaf = True
            self.output = np.argmax(np.bincount(data[class_col]))
            return
        if np.unique(data[class_col]).shape[0] == 1:
            self.is_leaf = True
            self.output = np.argmax(np.bincount(data[class_col]))
            return
        if max_cost:
            max_cost *= np.log2(np.unique(data[class_col]).shape[0])

        all_costs = {}
        for col in [i for i in data.columns if i != class_col]:
            if np.unique(data[col]).shape == (1,):
                continue
            class_prob = np.bincount(data[col], ) / len(data)
            current_cost = sum([class_prob[i] * cost_func(data.loc[data[col] == i, :][class_col]) \
                                for i in range(len(class_prob))])
            if max_cost and min_depth:
                if self.depth > min_depth and current_cost > max_cost:
                    continue
            all_costs[col] = current_cost

        if len(all_costs) == 0:
            self.is_leaf = True
            self.output = np.argmax(np.bincount(data[class_col]))
            return

        self.all_costs = all_costs
        self.split_feature = min(all_costs, key=lambda x: all_costs[x])
        if max_cost and min_depth:
            self.children = {
                i: NodePruned(data.loc[data[self.split_feature] == i, :].drop(columns=[self.split_feature]),
                              cost_func,
                              parent=self,
                              max_cost=max_cost,
                              min_depth=min_depth) for i in np.unique(data[self.split_feature])}
        else:
            for i in np.unique(data[self.split_feature]):
                new_data = data.loc[data[self.split_feature] == i, :].drop(columns=[self.split_feature])
                self.children[i] = NodePruned(new_data, cost_func, parent=self)

    def predict_one_datapoint(self, x):
        if self.is_leaf:
            if self.output is None:
                self.output = np.argmax(np.bincount(self.data[self.class_col]))
            return self.output
        #         return self.children[x[self.split_feature]].predict_one_dataPoint(x)
        try:
            return self[x[self.split_feature]].predict_one_datapoint(x)
        except KeyError:
            return np.argmax(np.bincount(self.data[self.class_col]))


def dfs(root: NodePruned | Node):
    ans = []
    for node in root.children.values():
        if not node.is_leaf:
            ans += dfs(node)

    return ans + [root]


def post_prune_tree(root: Node, validation_data: pd.DataFrame, verbose=False):
    best_acc = root.accuracy(validation_data)
    dfs_order = dfs(root)
    number_of_pruned_nodes = 0
    number_of_rollback = 0
    for index, node in enumerate(dfs_order):
        dfs_order[index].is_leaf = True
        if root.accuracy(validation_data) > best_acc:
            # we find better acc
            bestAcc = root.accuracy
            number_of_pruned_nodes += 1
        else:
            # we messed up, lets rollback
            dfs_order[index].is_leaf = False
            number_of_rollback += 1
    if verbose:
        print(f"number of pruned nodes: {number_of_pruned_nodes}\n"
              f"number of rollbacks: {number_of_rollback}\n")
    return root
