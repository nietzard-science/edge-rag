"""
Knowledge Graph Visualisation — PNG and Interactive HTML

Generates a publication-quality figure of the RELATED_TO subgraph for the
paper artifact. Hub entities (pronouns, generic terms) are filtered to keep
the visualisation readable. Node size encodes degree centrality; colour encodes
entity type.

Architectural position: Development utility (not part of the production pipeline).
Consumed by: Paper author for figure generation (Chapter 4 evaluation).

Output files (written to docs/figures/):
    graph_preview.png  — Static PNG for the paper (use dpi=300 for print quality)
    graph_preview.html — Interactive pyvis graph for exploration (optional)

Usage:
    python test_system/graph_3d.py
    python test_system/graph_3d.py --top 100        # show more nodes
    python test_system/graph_3d.py --no-html        # PNG only
    python test_system/graph_3d.py --dataset 2wiki  # different dataset
    python test_system/graph_3d.py --no-show        # headless / CI mode

Last reviewed: 2026-05-30 (audit pass, project version 5.4).
"""

import sys
import argparse
import logging
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import kuzu
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

PROJECT_ROOT = Path(__file__).parent.parent
# Paper figures are written to a single canonical location (docs/figures/)
# so that every generated artefact lives outside the test tree and beside the
# rest of the paper documentation.
OUTPUT_DIR   = PROJECT_ROOT / "docs" / "figures"


# ---------------------------------------------------------------------------
# Entity type → display colour
# Types must match what GLiNER stores in the Entity.type field.
# After NER refactoring (v3.5.0) types are stored UPPERCASE by the
# entity_extraction pipeline (spaCy-style: PERSON, GPE, WORK_OF_ART, …).
# ---------------------------------------------------------------------------
TYPE_COLORS: dict[str, str] = {
    "PERSON":        "#e74c3c",   # Red
    "ORGANIZATION":  "#3498db",   # Blue
    "GPE":           "#2ecc71",   # Green  (geopolitical entity)
    "WORK_OF_ART":   "#f39c12",   # Orange
    "EVENT":         "#9b59b6",   # Purple
    "LOC":           "#1abc9c",   # Teal
    "FAC":           "#e67e22",   # Dark orange (facility/building)
    "LOCATION":      "#1abc9c",   # Teal (alias used by some GLiNER configs)
}
DEFAULT_COLOR = "#7f8c8d"

LEGEND_PATCHES = [
    mpatches.Patch(color="#e74c3c", label="Person"),
    mpatches.Patch(color="#3498db", label="Organization"),
    mpatches.Patch(color="#2ecc71", label="Location (GPE)"),
    mpatches.Patch(color="#f39c12", label="Work of Art"),
    mpatches.Patch(color="#9b59b6", label="Event"),
    mpatches.Patch(color="#1abc9c", label="Location (LOC/FAC)"),
    mpatches.Patch(color="#7f8c8d", label="Other"),
]

# Generic terms and pronouns that create misleading hub nodes.
HUB_WORDS: frozenset[str] = frozenset({
    "i", "he", "she", "it", "they", "we", "you",
    "his", "her", "their", "him", "them", "who", "that",
    "this", "these", "those", "its", "one",
    "american", "united states", "us", "uk", "british",
    "country", "film", "movie", "people", "world", "city",
    "government", "man", "woman", "year", "time", "place",
    "football", "song", "album", "book", "company", "university",
    "music", "television", "television series", "series", "new",
})


def _is_hub(name: str) -> bool:
    """Return True if the entity name is a generic hub word to be filtered."""
    if not name or len(name.strip()) <= 2:
        return True
    return name.strip().lower() in HUB_WORDS


def _resolve_db_path(dataset: str) -> Path:
    """
    Derive the KuzuDB path from the dataset name.

    Path convention: data/{dataset}/graph/graph_KuzuDB

    Args:
        dataset: Dataset identifier (e.g. "hotpotqa").

    Returns:
        Absolute path to the KuzuDB directory.

    Raises:
        FileNotFoundError: If the database directory does not exist.
    """
    path = PROJECT_ROOT / "data" / dataset / "graph" / "graph_KuzuDB"
    if not path.exists():
        raise FileNotFoundError(
            f"KuzuDB not found at: {path}\n"
            f"  Run ingestion first: "
            f"python local_importingestion.py --dataset {dataset}"
        )
    return path


def _load_edges(conn: kuzu.Connection, limit: int = 2000) -> list[tuple]:
    """
    Load RELATED_TO edges from KuzuDB, filtered to named-entity types.

    Entity types are UPPERCASE (PERSON, ORGANIZATION, WORK_OF_ART, GPE,
    EVENT, LOC, FAC, LOCATION) as stored by the entity_extraction pipeline.

    Args:
        conn:  Open KuzuDB connection.
        limit: Maximum number of raw edges to fetch before hub filtering.

    Returns:
        List of (e1_name, e1_type, relation_type, e2_name, e2_type) tuples
        with hub entities removed.
    """
    named_entity_types = (
        "['PERSON', 'ORGANIZATION', 'WORK_OF_ART', 'GPE', "
        "'EVENT', 'LOC', 'FAC', 'LOCATION']"
    )
    query = f"""
        MATCH (e1:Entity)-[rel:RELATED_TO]->(e2:Entity)
        WHERE e1.type IN {named_entity_types}
          AND e2.type IN {named_entity_types}
        RETURN e1.name, e1.type, rel.relation_type, e2.name, e2.type
        LIMIT {limit}
    """
    result = conn.execute(query)
    edges = []
    while result.has_next():
        row = result.get_next()
        e1_name, e1_type, rel_type, e2_name, e2_type = row
        if not _is_hub(e1_name) and not _is_hub(e2_name):
            edges.append((e1_name, e1_type, rel_type, e2_name, e2_type))
    return edges


def _read_graph_stats(conn: kuzu.Connection) -> dict[str, int]:
    """
    Read entity, relation, and mention counts directly from KuzuDB.

    Returns:
        Dict with keys: entities, relations, mentions.
    """
    stats = {}
    for key, query in (
        ("entities",  "MATCH (e:Entity) RETURN COUNT(e)"),
        ("relations", "MATCH ()-[r:RELATED_TO]->() RETURN COUNT(r)"),
        ("mentions",  "MATCH ()-[r:MENTIONS]->() RETURN COUNT(r)"),
    ):
        r = conn.execute(query)
        stats[key] = r.get_next()[0]
    return stats


def _build_networkx_graph(
    edges: list[tuple],
    top_n: int,
) -> tuple[nx.DiGraph, dict[str, str]]:
    """
    Build a NetworkX DiGraph from the top-N entities by degree.

    Isolated nodes (no edges to other top-N entities) are excluded to keep
    the figure legible.

    Args:
        edges: Filtered (e1_name, e1_type, rel_type, e2_name, e2_type) list.
        top_n: Maximum number of entities to include.

    Returns:
        (G, entity_types) where G is the DiGraph and entity_types maps
        node name → entity type string.
    """
    degree: Counter = Counter()
    for e1, _, _, e2, _ in edges:
        degree[e1] += 1
        degree[e2] += 1

    top_entities = {e for e, _ in degree.most_common(top_n)}
    min_degree   = min((degree[e] for e in top_entities), default=0)
    logger.info(
        "Top-%d entities selected (min degree: %d)", top_n, min_degree
    )

    G: nx.DiGraph = nx.DiGraph()
    entity_types: dict[str, str] = {}

    for e1, t1, rel_type, e2, t2 in edges:
        if e1 in top_entities and e2 in top_entities:
            entity_types[e1] = t1
            entity_types[e2] = t2
            G.add_edge(e1, e2, relation=rel_type)

    logger.info(
        "Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges()
    )
    return G, entity_types


def _draw(
    G: nx.DiGraph,
    entity_types: dict[str, str],
    top_n: int,
    stats: dict[str, int],
) -> plt.Figure:
    """
    Render the knowledge graph figure.

    Args:
        G:            NetworkX DiGraph.
        entity_types: Node name → entity type mapping for colour assignment.
        top_n:        Label shown in title.
        stats:        Live counts from KuzuDB for the stats annotation box.

    Returns:
        matplotlib Figure object.
    """
    logger.info("Computing spring layout ...")
    pos = nx.spring_layout(G, k=2.0, iterations=80, seed=42)

    node_colors = [
        TYPE_COLORS.get(entity_types.get(n, ""), DEFAULT_COLOR)
        for n in G.nodes()
    ]
    # Node size proportional to degree; minimum 150 for visibility.
    node_sizes = [max(150, 100 + 80 * G.degree(n)) for n in G.nodes()]

    logger.info("Rendering figure ...")
    fig, ax = plt.subplots(figsize=(18, 13), facecolor="#16213e")
    ax.set_facecolor("#16213e")

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#5a6694",
        arrows=True,
        arrowsize=8,
        alpha=0.45,
        width=0.7,
        connectionstyle="arc3,rad=0.08",
        min_source_margin=12,
        min_target_margin=12,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.92,
        linewidths=0.5,
        edgecolors="#ffffff40",
    )
    # Only label nodes with degree >= 3 to avoid clutter.
    labels = {n: n for n in G.nodes() if G.degree(n) >= 3}
    nx.draw_networkx_labels(
        G, pos, labels=labels, ax=ax,
        font_color="white",
        font_size=6.5,
        font_weight="bold",
    )

    legend = ax.legend(
        handles=LEGEND_PATCHES,
        loc="upper left",
        facecolor="#1a1a3e",
        edgecolor="#5a6694",
        labelcolor="white",
        fontsize=9,
        framealpha=0.85,
        title="Entity Type",
        title_fontsize=9,
    )
    legend.get_title().set_color("white")

    ax.set_title(
        f"Knowledge Graph — HotpotQA Corpus\n"
        f"{G.number_of_nodes()} Named Entities · "
        f"{G.number_of_edges()} RELATED_TO Edges  "
        f"(Top-{top_n} by degree, hub entities filtered)",
        color="white",
        fontsize=11,
        pad=15,
    )

    # Stats annotation box — counts read live from KuzuDB, not hardcoded.
    stats_text = (
        f"Total graph: {stats['entities']:,} entities · "
        f"{stats['relations']:,} relations · "
        f"{stats['mentions']:,} mentions\n"
        f"Extraction: GLiNER (urchade/gliner_small-v2.1) + REBEL"
    )
    ax.text(
        0.99, 0.01, stats_text,
        transform=ax.transAxes,
        fontsize=7.5,
        color="#aaaacc",
        ha="right", va="bottom",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#1a1a3e",
            alpha=0.7,
            edgecolor="#5a6694",
        ),
    )
    ax.axis("off")
    plt.tight_layout(pad=1.5)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a knowledge graph figure for the paper."
    )
    parser.add_argument(
        "--top", type=int, default=80,
        help="Maximum number of entities to display (default: 80)"
    )
    parser.add_argument(
        "--dataset", default="hotpotqa",
        help="Dataset name used to locate the KuzuDB (default: hotpotqa)"
    )
    parser.add_argument(
        "--no-html", action="store_true",
        help="Skip interactive HTML output (PNG only)"
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Do not call plt.show() — use for headless/CI environments"
    )
    args = parser.parse_args()

    # ── Connect to KuzuDB ────────────────────────────────────────────────────
    try:
        db_path = _resolve_db_path(args.dataset)
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)

    logger.info("Connecting to KuzuDB: %s", db_path)
    db   = kuzu.Database(str(db_path))
    conn = kuzu.Connection(db)

    # ── Load edges and graph stats ───────────────────────────────────────────
    logger.info("Loading RELATED_TO edges ...")
    edges = _load_edges(conn)
    logger.info("After hub-filter: %d edges remain", len(edges))

    if not edges:
        logger.error(
            "No RELATED_TO edges found after filtering. "
            "Check that entity types in the graph match the filter list "
            "(expected UPPERCASE: PERSON, ORGANIZATION, GPE, …). "
            "Run: python test_system/graph_inspect.py to inspect stored types."
        )
        sys.exit(1)

    stats = _read_graph_stats(conn)
    logger.info(
        "Graph totals — entities: %d, relations: %d, mentions: %d",
        stats["entities"], stats["relations"], stats["mentions"],
    )

    # ── Build and render ─────────────────────────────────────────────────────
    G, entity_types = _build_networkx_graph(edges, top_n=args.top)

    if G.number_of_nodes() == 0:
        logger.error(
            "No nodes in filtered graph. Try --top with a higher value."
        )
        sys.exit(1)

    fig = _draw(G, entity_types, top_n=args.top, stats=stats)

    # ── Save PNG ─────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_png = OUTPUT_DIR / "graph_preview.png"
    fig.savefig(str(output_png), dpi=200, bbox_inches="tight", facecolor="#16213e")
    logger.info("PNG saved: %s  (use dpi=300 for print quality)", output_png)

    # ── Optional: interactive HTML via pyvis ─────────────────────────────────
    if not args.no_html:
        try:
            from pyvis.network import Network  # noqa: PLC0415

            net = Network(
                height="900px", width="100%",
                bgcolor="#16213e", font_color="white", directed=True,
            )
            net.barnes_hut(
                gravity=-8000, central_gravity=0.3, spring_length=120
            )
            for node in G.nodes():
                etype = entity_types.get(node, "")
                color = TYPE_COLORS.get(etype, DEFAULT_COLOR)
                size  = 12 + 3 * G.degree(node)
                net.add_node(
                    node, label=node, color=color, size=size,
                    title=f"{node} ({etype})",
                )
            for e1, e2, data in G.edges(data=True):
                net.add_edge(
                    e1, e2,
                    title=data.get("relation", ""),
                    color="#5a6694", width=1.2,
                )
            output_html = OUTPUT_DIR / "graph_preview.html"
            net.show(str(output_html), notebook=False)
            logger.info("HTML saved: %s", output_html)
        except ImportError:
            logger.warning(
                "pyvis not installed — skipping HTML output. "
                "Install with: pip install pyvis"
            )

    # ── Display (skip in headless/CI environments) ───────────────────────────
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
