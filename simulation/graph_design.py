"""Explicit modular DAGs used by every CGLCP simulation experiment.

The graph family separates directed causal reachability from undirected
locality.  In the reference DAG, ancestor interventions reach a common target
through a module hub and, for remote modules, a depth-controlled convergent
pathway.  In the query module, safe interventions are partitioned into
``S_in``, ``S_out``, and ``S_base`` with mutually exclusive edge rules, while
only a controlled fraction ``pi_ua`` of upstream ancestors receives a direct
``u -> query`` edge.  These motifs preserve acyclicity and create distinct
distance shells for true-safe and false-safe interventions.
A collider bridge keeps each ancestor close to its hub after its
ancestor-to-hub edge is deleted in the working graph, without restoring a
directed path to the target.

All simulated local pools are derived from graph labels and graph distances;
the requested local unsafe count is used only to choose where working-graph
edge deletions occur.  The constructor verifies the resulting count before it
returns.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from math import isclose
from random import Random
from typing import Iterable


Edge = tuple[str, str]


def _directed_adjacency(nodes: Iterable[str], edges: Iterable[Edge]) -> dict[str, set[str]]:
    adjacency = {node: set() for node in nodes}
    for source, target in edges:
        adjacency[source].add(target)
    return adjacency


def _undirected_adjacency(nodes: Iterable[str], edges: Iterable[Edge]) -> dict[str, set[str]]:
    adjacency = {node: set() for node in nodes}
    for source, target in edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    return adjacency


def _can_reach(adjacency: dict[str, set[str]], source: str, target: str) -> bool:
    if source == target:
        return True
    visited = {source}
    frontier = [source]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency[current]:
            if neighbor == target:
                return True
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    return False


def _distances(adjacency: dict[str, set[str]], source: str) -> dict[str, int]:
    distance = {source: 0}
    queue = deque([source])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in distance:
                distance[neighbor] = distance[current] + 1
                queue.append(neighbor)
    return distance


def _is_acyclic(nodes: Iterable[str], edges: Iterable[Edge]) -> bool:
    """Return whether the directed graph is acyclic (Kahn's algorithm)."""

    node_list = list(nodes)
    adjacency = _directed_adjacency(node_list, edges)
    indegree = {node: 0 for node in node_list}
    for source in node_list:
        for target in adjacency[source]:
            indegree[target] += 1
    queue = deque(node for node, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        source = queue.popleft()
        visited += 1
        for target in adjacency[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return visited == len(node_list)


def _balanced_counts(total: int, groups: int) -> list[int]:
    base, remainder = divmod(total, groups)
    return [base + int(index < remainder) for index in range(groups)]


@dataclass(frozen=True)
class ModularGraphScenario:
    """One reference/working graph pair and its graph-derived local pool."""

    name: str
    modules: int
    target: str
    query: str
    nodes: tuple[str, ...]
    reference_edges: tuple[Edge, ...]
    working_edges: tuple[Edge, ...]
    deleted_edges: tuple[Edge, ...]
    calibration_interventions: tuple[str, ...]
    predicted_safe_interventions: tuple[str, ...]
    local_pool: tuple[str, ...]
    local_unsafe_labels: tuple[int, ...]
    local_distances: tuple[int, ...]
    topology: str = "layered"
    pathway_mediators: tuple[str, ...] = ()
    query_local_regulatory_edges: tuple[Edge, ...] = ()
    query_safe_in: tuple[str, ...] = ()
    query_safe_out: tuple[str, ...] = ()
    query_safe_base: tuple[str, ...] = ()
    query_near_ancestors: tuple[str, ...] = ()
    pi_ua: float = 0.0
    added_edges: tuple[Edge, ...] = ()

    @property
    def pool_size(self) -> int:
        return len(self.local_pool)

    @property
    def true_local_unsafe_count(self) -> int:
        return sum(self.local_unsafe_labels)

    @property
    def global_false_safe_count(self) -> int:
        reference = _directed_adjacency(self.nodes, self.reference_edges)
        return sum(
            _can_reach(reference, node, self.target)
            for node in self.predicted_safe_interventions
        )

    @property
    def global_contamination_rate(self) -> float:
        return self.global_false_safe_count / len(self.predicted_safe_interventions)

    def metadata(self) -> dict[str, int | float | str]:
        return {
            "graph_scenario": self.name,
            "graph_topology": self.topology,
            "graph_modules": self.modules,
            "graph_nodes": len(self.nodes),
            "reference_edges": len(self.reference_edges),
            "working_edges": len(self.working_edges),
            "deleted_edges": len(self.deleted_edges),
            "added_edges": len(self.added_edges),
            "global_predicted_safe_count": len(self.predicted_safe_interventions),
            "global_false_safe_count": self.global_false_safe_count,
            "global_contamination": self.global_contamination_rate,
            "pool_size": self.pool_size,
            "true_local_unsafe_count": self.true_local_unsafe_count,
            "local_contamination": self.true_local_unsafe_count / self.pool_size,
            "pathway_mediator_nodes": len(self.pathway_mediators),
            "query_local_regulatory_edges": len(self.query_local_regulatory_edges),
            "query_safe_in": len(self.query_safe_in),
            "query_safe_out": len(self.query_safe_out),
            "query_safe_base": len(self.query_safe_base),
            "query_near_ancestors": len(self.query_near_ancestors),
            "pi_ua": self.pi_ua,
        }


def _convergent_pathway_edges(
    hubs: list[str],
    target: str,
) -> tuple[list[str], list[Edge]]:
    """Create depth-controlled convergent pathways for remote module pairs.

    M1 and M2 point directly to the target.  M3--M4 share one mediator,
    M5--M6 share a two-mediator chain, and M7--M8 share a three-mediator
    chain.  Additional module pairs, if requested, continue this depth pattern.
    Direct hub-to-target edges are replaced, rather than retained, so the
    intended undirected distance shells are effective.
    """

    mediators: list[str] = []
    edges: list[Edge] = []
    for module, hub in enumerate(hubs):
        if module < 2:
            edges.append((hub, target))
    for group_start in range(2, len(hubs), 2):
        group_end = min(group_start + 1, len(hubs) - 1)
        depth = 1 + (group_start - 2) // 2
        group_name = f"m{group_start + 1}_m{group_end + 1}"
        chain = [
            f"pathway_{group_name}_layer{layer}"
            for layer in range(1, depth + 1)
        ]
        mediators.extend(chain)
        for module in range(group_start, group_end + 1):
            edges.append((hubs[module], chain[0]))
        edges.extend(zip(chain, chain[1:]))
        edges.append((chain[-1], target))
    return mediators, edges


def build_modular_graph_scenario(
    *,
    name: str,
    pool_size: int,
    max_local_unsafe: int,
    local_unsafe: int,
    modules: int = 8,
    total_deletions: int | None = None,
    global_contamination: float = 0.10,
    pi_ua: float = 0.25,
    topology_seed: int | None = None,
    deletion_seed: int | None = None,
) -> ModularGraphScenario:
    """Build a graph pair whose nearest pool has the requested unsafe count.

    ``max_local_unsafe`` fixes the reference graph and query-module safe count
    across an ablation.  ``local_unsafe`` changes only the spatial allocation
    of a fixed number of working-graph edge deletions.  The remaining deletions
    are distributed across modules other than the query module.

    The current construction requires ``global_contamination=0.10`` so that a
    fixed number ``E`` of deleted ancestor edges is paired with ``9E`` truly
    safe calibration leaves.  Consequently, the global predicted-safe pool
    has size ``10E`` and exactly ``E`` false-safe members.
    """

    if modules < 2:
        raise ValueError("modules must be at least 2")
    if pool_size < 1:
        raise ValueError("pool_size must be positive")
    if not 0 <= local_unsafe <= max_local_unsafe <= pool_size:
        raise ValueError(
            "Require 0 <= local_unsafe <= max_local_unsafe <= pool_size"
        )
    if not isclose(global_contamination, 0.10):
        raise ValueError("The explicit graph family currently fixes global contamination at 0.10")
    if not 0.0 <= pi_ua <= 1.0:
        raise ValueError("pi_ua must lie in [0, 1]")

    deletions = max_local_unsafe if total_deletions is None else total_deletions
    if deletions < max_local_unsafe:
        raise ValueError("total_deletions must be at least max_local_unsafe")
    query_safe_count = pool_size - max_local_unsafe
    safe_total = 9 * deletions
    if safe_total < query_safe_count:
        raise ValueError(
            "Not enough global safe leaves for the requested query-module pool; "
            "increase total_deletions"
        )

    target = "target"
    query = "query_m0"
    hubs = [f"hub_m{module}" for module in range(modules)]
    safe_counts = [query_safe_count] + _balanced_counts(
        safe_total - query_safe_count,
        modules - 1,
    )

    safe_nodes: list[list[str]] = []
    ancestor_nodes: list[list[str]] = []
    bridge_nodes: list[list[str]] = []
    for module in range(modules):
        safe_nodes.append(
            [f"safe_m{module}_{index}" for index in range(safe_counts[module])]
        )
        ancestor_nodes.append(
            [f"ancestor_m{module}_{index}" for index in range(deletions)]
        )
        bridge_nodes.append(
            [f"bridge_m{module}_{index}" for index in range(deletions)]
        )

    # The query-module safe set is a genuine disjoint partition.  Only S_base
    # shares the hub; S_in and S_out connect directly to the query in opposite
    # directions.  Balanced counts keep the three groups comparable.
    safe_in_count, safe_out_count, _ = _balanced_counts(query_safe_count, 3)
    query_safe_order = list(safe_nodes[0])
    if topology_seed is not None:
        Random(topology_seed).shuffle(query_safe_order)
    query_safe_in = query_safe_order[:safe_in_count]
    query_safe_out = query_safe_order[safe_in_count:safe_in_count + safe_out_count]
    query_safe_base = query_safe_order[safe_in_count + safe_out_count:]

    # Choose exactly round(pi_ua * E) query-module ancestors and distribute
    # them evenly through the fixed ancestor ordering.  Consequently, every
    # prefix of deleted M1 ancestors contains approximately the same pi_ua
    # fraction, while all scenarios retain one common reference graph.
    near_count = round(pi_ua * deletions)
    if topology_seed is None:
        near_indices = {
            index
            for index in range(deletions)
            if ((index + 1) * near_count) // deletions
            > (index * near_count) // deletions
        }
    else:
        near_rng = Random(topology_seed + 1)
        near_indices = set(near_rng.sample(range(deletions), near_count))
    query_near_ancestors = [ancestor_nodes[0][index] for index in sorted(near_indices)]

    pathway_mediators, pathway_edges = _convergent_pathway_edges(hubs, target)
    nodes = [target, query, *hubs, *pathway_mediators]
    for module in range(modules):
        nodes.extend(safe_nodes[module])
        nodes.extend(ancestor_nodes[module])
        nodes.extend(bridge_nodes[module])

    reference_edges: list[Edge] = list(pathway_edges)
    for module, hub in enumerate(hubs):
        if module == 0:
            reference_edges.extend((hub, safe) for safe in query_safe_base)
        else:
            reference_edges.extend((hub, safe) for safe in safe_nodes[module])
        for ancestor, bridge in zip(ancestor_nodes[module], bridge_nodes[module]):
            reference_edges.extend(
                [
                    (ancestor, hub),
                    (ancestor, bridge),
                    (hub, bridge),
                ]
            )
    reference_edges.append((hubs[0], query))

    # Gene-like local regulation in M1.  S_in -> a, a -> S_out, and
    # h1 -> S_base are mutually exclusive.  Only the pi_ua subset of ancestors
    # points directly to a; the remaining deleted ancestors stay at distance
    # three through their within-module collider bridge.
    query_local_regulatory_edges: list[Edge] = [
        *((safe, query) for safe in query_safe_in),
        *((query, safe) for safe in query_safe_out),
        *((ancestor, query) for ancestor in query_near_ancestors),
    ]
    reference_edges.extend(query_local_regulatory_edges)

    query_deletion_order = list(range(deletions))
    remote_deletion_candidates = [
        (module, index)
        for module in range(1, modules)
        for index in range(deletions)
    ]
    if deletion_seed is not None:
        deletion_rng = Random(deletion_seed)
        deletion_rng.shuffle(query_deletion_order)
        deletion_rng.shuffle(remote_deletion_candidates)
    deleted_edges: list[Edge] = [
        (ancestor_nodes[0][index], hubs[0])
        for index in query_deletion_order[:local_unsafe]
    ]
    remaining = deletions - local_unsafe
    for index in range(remaining):
        if deletion_seed is None:
            module = 1 + (index % (modules - 1))
            within_module = index // (modules - 1)
        else:
            module, within_module = remote_deletion_candidates[index]
        deleted_edges.append(
            (ancestor_nodes[module][within_module], hubs[module])
        )

    deleted_set = set(deleted_edges)
    working_edges = [edge for edge in reference_edges if edge not in deleted_set]
    calibration = [
        node
        for module in range(modules)
        for node in (*safe_nodes[module], *ancestor_nodes[module])
    ]

    reference_adjacency = _directed_adjacency(nodes, reference_edges)
    working_adjacency = _directed_adjacency(nodes, working_edges)
    undirected_working = _undirected_adjacency(nodes, working_edges)
    distance = _distances(undirected_working, query)

    predicted_safe = [
        node
        for node in calibration
        if not _can_reach(working_adjacency, node, target)
    ]
    predicted_safe.sort(key=lambda node: (distance[node], node))
    local_pool = predicted_safe[:pool_size]
    local_labels = [
        int(_can_reach(reference_adjacency, node, target))
        for node in local_pool
    ]

    scenario = ModularGraphScenario(
        name=name,
        modules=modules,
        target=target,
        query=query,
        nodes=tuple(nodes),
        reference_edges=tuple(reference_edges),
        working_edges=tuple(working_edges),
        deleted_edges=tuple(deleted_edges),
        calibration_interventions=tuple(calibration),
        predicted_safe_interventions=tuple(predicted_safe),
        local_pool=tuple(local_pool),
        local_unsafe_labels=tuple(local_labels),
        local_distances=tuple(distance[node] for node in local_pool),
        topology="gene_regulatory_convergent",
        pathway_mediators=tuple(pathway_mediators),
        query_local_regulatory_edges=tuple(query_local_regulatory_edges),
        query_safe_in=tuple(query_safe_in),
        query_safe_out=tuple(query_safe_out),
        query_safe_base=tuple(query_safe_base),
        query_near_ancestors=tuple(query_near_ancestors),
        pi_ua=pi_ua,
    )

    if scenario.true_local_unsafe_count != local_unsafe:
        raise AssertionError(
            f"Graph-derived local B={scenario.true_local_unsafe_count}, "
            f"expected {local_unsafe}"
        )
    if len(scenario.predicted_safe_interventions) != 10 * deletions:
        raise AssertionError("Unexpected global predicted-safe pool size")
    if scenario.global_false_safe_count != deletions:
        raise AssertionError("Unexpected global false-safe count")
    if not isclose(scenario.global_contamination_rate, global_contamination):
        raise AssertionError("Unexpected global contamination rate")
    if len(scenario.deleted_edges) != deletions:
        raise AssertionError("Unexpected number of working-graph deletions")
    if len(scenario.working_edges) != len(scenario.reference_edges) - deletions:
        raise AssertionError("Working edge count does not match deletions")
    if not _is_acyclic(nodes, reference_edges):
        raise AssertionError("The reference graph must remain a DAG")
    if not _is_acyclic(nodes, working_edges):
        raise AssertionError("The working graph must remain a DAG")
    if _can_reach(reference_adjacency, query, target):
        raise AssertionError("The held-out query must be truly safe")
    if _can_reach(working_adjacency, query, target):
        raise AssertionError("The held-out query must be predicted safe")
    return scenario


def add_uniform_false_exclusion_edges(
    scenario: ModularGraphScenario,
    *,
    addition_count: int,
    seed: int,
    pool_size: int | None = None,
    name: str | None = None,
    nested: bool = False,
) -> ModularGraphScenario:
    """Add uniformly selected false-positive working-graph edges.

    Each added edge has the form ``safe intervention -> target`` and is present
    only in the working graph.  It therefore makes a genuinely safe
    intervention appear unsafe and removes it from the predicted-safe
    population.  Candidates that are descendants of the held-out query are
    excluded so the query itself remains predicted safe.

    To keep the global predicted-safe population size, false-safe count, and
    contamination rate exactly unchanged, the constructor also appends one
    remote true-safe buffer intervention for each added edge.  These buffer
    nodes are attached as ``last module hub -> buffer`` in both graphs.  They
    are ordinary calibration interventions; their only purpose is to prevent
    the denominator change caused by false exclusion from being confounded
    with the edge-addition experiment.

    Conditional on ``addition_count``, the affected true-safe interventions
    are sampled uniformly without replacement using ``seed``.  The same seed
    can therefore be reused across local-deletion scenarios to hold the added
    edge set fixed while the deleted-edge locations change.
    """

    if addition_count < 0:
        raise ValueError("addition_count must be non-negative")
    requested_pool_size = scenario.pool_size if pool_size is None else pool_size
    if requested_pool_size < 1:
        raise ValueError("pool_size must be positive")
    if addition_count == 0:
        return replace(scenario, name=name or scenario.name)

    reference_adjacency = _directed_adjacency(
        scenario.nodes,
        scenario.reference_edges,
    )
    working_adjacency = _directed_adjacency(
        scenario.nodes,
        scenario.working_edges,
    )
    candidates = sorted(
        node
        for node in scenario.predicted_safe_interventions
        if not _can_reach(reference_adjacency, node, scenario.target)
        and not _can_reach(working_adjacency, scenario.query, node)
    )
    if addition_count > len(candidates):
        raise ValueError(
            f"Requested {addition_count} additions but only "
            f"{len(candidates)} eligible true-safe interventions exist"
        )
    if nested:
        ordered_candidates = list(candidates)
        Random(seed).shuffle(ordered_candidates)
        selected = tuple(sorted(ordered_candidates[:addition_count]))
    else:
        selected = tuple(sorted(Random(seed).sample(candidates, addition_count)))
    added_edges = tuple((node, scenario.target) for node in selected)

    named_remote_hub = f"hub_m{scenario.modules - 1}"
    if named_remote_hub in scenario.nodes:
        remote_hub = named_remote_hub
    else:
        target_parents = sorted(
            source
            for source, target in scenario.reference_edges
            if target == scenario.target
        )
        if not target_parents:
            raise ValueError("No reference-graph parent of the target is available")
        remote_hub = target_parents[-1]
    buffer_nodes = tuple(
        f"safe_addition_buffer_m{scenario.modules - 1}_{index}"
        for index in range(addition_count)
    )
    if set(buffer_nodes) & set(scenario.nodes):
        raise ValueError("Generated buffer-node names collide with existing nodes")
    buffer_edges = tuple((remote_hub, node) for node in buffer_nodes)

    nodes = (*scenario.nodes, *buffer_nodes)
    reference_edges = (*scenario.reference_edges, *buffer_edges)
    working_edges = (*scenario.working_edges, *buffer_edges, *added_edges)
    calibration = (*scenario.calibration_interventions, *buffer_nodes)

    reference_adjacency = _directed_adjacency(nodes, reference_edges)
    working_adjacency = _directed_adjacency(nodes, working_edges)
    undirected_working = _undirected_adjacency(nodes, working_edges)
    distances = _distances(undirected_working, scenario.query)
    predicted_safe = [
        node
        for node in calibration
        if not _can_reach(working_adjacency, node, scenario.target)
    ]
    predicted_safe.sort(key=lambda node: (distances[node], node))
    local_pool = predicted_safe[:requested_pool_size]
    local_labels = tuple(
        int(_can_reach(reference_adjacency, node, scenario.target))
        for node in local_pool
    )

    result = replace(
        scenario,
        name=name or f"{scenario.name}_add{addition_count}",
        nodes=tuple(nodes),
        reference_edges=tuple(reference_edges),
        working_edges=tuple(working_edges),
        added_edges=added_edges,
        calibration_interventions=tuple(calibration),
        predicted_safe_interventions=tuple(predicted_safe),
        local_pool=tuple(local_pool),
        local_unsafe_labels=local_labels,
        local_distances=tuple(distances[node] for node in local_pool),
        topology=f"{scenario.topology}_with_false_exclusion_additions",
    )

    if not _is_acyclic(nodes, reference_edges):
        raise AssertionError("The augmented reference graph must remain a DAG")
    if not _is_acyclic(nodes, working_edges):
        raise AssertionError("The augmented working graph must remain a DAG")
    if _can_reach(reference_adjacency, scenario.query, scenario.target):
        raise AssertionError("The held-out query must remain truly safe")
    if _can_reach(working_adjacency, scenario.query, scenario.target):
        raise AssertionError("Added edges made the held-out query predicted unsafe")
    if len(result.predicted_safe_interventions) != len(
        scenario.predicted_safe_interventions
    ):
        raise AssertionError("Global predicted-safe population size changed")
    if result.global_false_safe_count != scenario.global_false_safe_count:
        raise AssertionError("Global false-safe count changed")
    if not isclose(
        result.global_contamination_rate,
        scenario.global_contamination_rate,
    ):
        raise AssertionError("Global contamination rate changed")
    if len(result.working_edges) != (
        len(scenario.working_edges) + len(buffer_edges) + len(added_edges)
    ):
        raise AssertionError("Unexpected augmented working-edge count")
    return result


def build_layered_locality_scenario(
    *,
    name: str = "s7_layered_locality",
) -> ModularGraphScenario:
    """Build one DAG with five nested, distance-defined calibration pools.

    The working-predicted-safe population has 300 calibration interventions
    arranged in five distance shells of sizes ``25, 25, 50, 100, 100``.
    The shells contain ``10, 5, 5, 5, 5`` deleted true ancestors.  Therefore
    the radius pools for ``tau=2, 3, 4, 5, 6`` have respectively
    ``(n, B)=(25,10), (50,15), (100,20), (200,25), (300,30)``.  The full
    population has global contamination 0.10.  Every count is a consequence
    of directed reachability and undirected working-graph distance.
    """

    shell_sizes = [25, 25, 50, 100, 100]
    shell_unsafe = [10, 5, 5, 5, 5]
    target = "target"
    query = "query"
    hub = "causal_hub"
    anchors = [query, *[f"anchor_{index}" for index in range(1, 6)]]

    nodes = [target, hub, *anchors]
    reference_edges: list[Edge] = [(hub, target)]
    reference_edges.extend(
        (anchors[index], anchors[index + 1])
        for index in range(len(anchors) - 1)
    )
    deleted_edges: list[Edge] = []
    calibration: list[str] = []

    for shell, (size, unsafe_count) in enumerate(
        zip(shell_sizes, shell_unsafe)
    ):
        safe_count = size - unsafe_count
        for index in range(safe_count):
            safe = f"shell{shell}_safe_{index:03d}"
            nodes.append(safe)
            calibration.append(safe)
            # Safe and deleted-ancestor members of a shell have equal
            # undirected distance from the query.
            reference_edges.append((anchors[shell + 1], safe))
        for index in range(unsafe_count):
            ancestor = f"shell{shell}_ancestor_{index:03d}"
            bridge = f"shell{shell}_bridge_{index:03d}"
            nodes.extend([ancestor, bridge])
            calibration.append(ancestor)
            reference_edges.extend(
                [
                    (ancestor, hub),
                    (ancestor, bridge),
                    (anchors[shell], bridge),
                ]
            )
            deleted_edges.append((ancestor, hub))

    deleted_set = set(deleted_edges)
    working_edges = [edge for edge in reference_edges if edge not in deleted_set]
    reference_adjacency = _directed_adjacency(nodes, reference_edges)
    working_adjacency = _directed_adjacency(nodes, working_edges)
    undirected_working = _undirected_adjacency(nodes, working_edges)
    distance = _distances(undirected_working, query)

    predicted_safe = [
        node
        for node in calibration
        if not _can_reach(working_adjacency, node, target)
    ]
    predicted_safe.sort(key=lambda node: (distance[node], node))
    local_labels = [
        int(_can_reach(reference_adjacency, node, target))
        for node in predicted_safe
    ]
    scenario = ModularGraphScenario(
        name=name,
        modules=len(shell_sizes),
        target=target,
        query=query,
        nodes=tuple(nodes),
        reference_edges=tuple(reference_edges),
        working_edges=tuple(working_edges),
        deleted_edges=tuple(deleted_edges),
        calibration_interventions=tuple(calibration),
        predicted_safe_interventions=tuple(predicted_safe),
        local_pool=tuple(predicted_safe),
        local_unsafe_labels=tuple(local_labels),
        local_distances=tuple(distance[node] for node in predicted_safe),
        topology="nested_distance_shells",
    )

    expected = {
        2: (25, 10),
        3: (50, 15),
        4: (100, 20),
        5: (200, 25),
        6: (300, 30),
    }
    for tau, (pool_size, unsafe_count) in expected.items():
        selected = [
            index
            for index, value in enumerate(scenario.local_distances)
            if value <= tau
        ]
        if len(selected) != pool_size:
            raise AssertionError(
                f"Radius tau={tau} does not have n_tau={pool_size}"
            )
        if sum(scenario.local_unsafe_labels[index] for index in selected) != unsafe_count:
            raise AssertionError(
                f"Radius tau={tau} does not have B_tau={unsafe_count}"
            )
    if len(scenario.predicted_safe_interventions) != 300:
        raise AssertionError("Unexpected layered predicted-safe population")
    if scenario.global_false_safe_count != 30:
        raise AssertionError("Unexpected layered false-safe count")
    if not isclose(scenario.global_contamination_rate, 0.10):
        raise AssertionError("Unexpected layered global contamination")
    if _can_reach(reference_adjacency, query, target):
        raise AssertionError("The layered query must be reference-safe")
    return scenario


def take_local_pool_prefix(
    scenario: ModularGraphScenario,
    pool_size: int,
    *,
    name: str,
) -> ModularGraphScenario:
    """Return a nearest-neighbor prefix while preserving the same graph pair."""

    if not 1 <= pool_size <= len(scenario.local_pool):
        raise ValueError("pool_size must index a nonempty local-pool prefix")
    return replace(
        scenario,
        name=name,
        local_pool=scenario.local_pool[:pool_size],
        local_unsafe_labels=scenario.local_unsafe_labels[:pool_size],
        local_distances=scenario.local_distances[:pool_size],
    )


def take_local_pool_radius(
    scenario: ModularGraphScenario,
    radius: int,
    *,
    name: str,
) -> ModularGraphScenario:
    """Return the complete working-distance ball with radius ``radius``."""

    selected = [
        index
        for index, distance in enumerate(scenario.local_distances)
        if distance <= radius
    ]
    if not selected:
        raise ValueError("radius must define a nonempty local pool")
    return replace(
        scenario,
        name=name,
        local_pool=tuple(scenario.local_pool[index] for index in selected),
        local_unsafe_labels=tuple(
            scenario.local_unsafe_labels[index] for index in selected
        ),
        local_distances=tuple(
            scenario.local_distances[index] for index in selected
        ),
    )


__all__ = [
    "ModularGraphScenario",
    "build_layered_locality_scenario",
    "build_modular_graph_scenario",
    "take_local_pool_prefix",
    "take_local_pool_radius",
]
