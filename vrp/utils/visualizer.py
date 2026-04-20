from __future__ import annotations

import matplotlib.pyplot as plt

from vrp.models.gnn_input import GNNInput
from vrp.models.gnn_output import GNNOutput


def plot_graph_pair(graph_input: GNNInput, graph_output: GNNOutput) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    input_ax, output_ax = axes

    draw_graph_input(input_ax, graph_input)
    input_ax.set_title("Graph input")

    draw_graph_input(output_ax, graph_input)
    draw_graph_output(output_ax, graph_input, graph_output)
    output_ax.set_title("Graph output")

    finalize_plot(fig)


def draw_graph_input(ax: plt.Axes, graph_input: GNNInput) -> None:
    depot_x: list[float] = []
    depot_y: list[float] = []
    client_x: list[float] = []
    client_y: list[float] = []

    for node_idx in range(graph_input.get_node_count()):
        x, y = graph_input.get_coordinates(node_idx)
        if graph_input.is_depot(node_idx):
            depot_x.append(x)
            depot_y.append(y)
        else:
            client_x.append(x)
            client_y.append(y)

        annotate_node(ax, graph_input, node_idx)

    if depot_x:
        ax.scatter(
            depot_x,
            depot_y,
            c="#d62728",
            marker="s",
            s=120,
            label="Depots",
            edgecolors="black",
        )

    if client_x:
        ax.scatter(
            client_x,
            client_y,
            c="#1f77b4",
            marker="o",
            s=70,
            label="Clients",
            edgecolors="black",
            linewidths=0.5,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.25)
    ax.legend()


def draw_graph_output(
    ax: plt.Axes,
    graph_input: GNNInput,
    graph_output: GNNOutput,
) -> None:
    active_edges: list[tuple[int, int]] = []

    for edge_idx, (from_idx, to_idx) in enumerate(graph_input.get_edge_pairs()):
        if graph_output.edge_scores[edge_idx] > 0.5:
            active_edges.append((from_idx, to_idx))

    for from_idx, to_idx in active_edges:
        from_x, from_y = graph_input.get_coordinates(from_idx)
        to_x, to_y = graph_input.get_coordinates(to_idx)
        ax.plot(
            [from_x, to_x],
            [from_y, to_y],
            color="#000000",
            linewidth=2,
            alpha=0.85,
        )


def annotate_node(ax: plt.Axes, graph_input: GNNInput, node_idx: int) -> None:
    x, y = graph_input.get_coordinates(node_idx)
    ax.annotate(
        str(node_idx),
        (x, y),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=8,
    )


def finalize_plot(fig: plt.Figure) -> None:
    fig.tight_layout()
    plt.show()
