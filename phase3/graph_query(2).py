from langchain_neo4j import Neo4jGraph

# Connect LangChain to your Neo4j instance
graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="password123",
)

# Run a Cypher query and get results as Python dicts
result = graph.query("""
    MATCH (n)-[r]->(m)
    RETURN n.name as from_node, 
           type(r) as relationship, 
           m.name as to_node
""")

for row in result:
    print(row)

print("=====================================\n")

# Query 1: Get all nodes connected to AI
nodename = "Artificial Intelligence"

def get_all_nodes_connected(nodename):
    query = """
        MATCH (ai {name: $ai_name})-[r]-(related)
        RETURN
            ai.name AS ai_node,
            type(r) AS relationship,
            related.name AS related_node
        """
    result = graph.query(
        query,
        params= {"ai_name":nodename}
    )

    return result

custome_node_result = get_all_nodes_connected(nodename)

for row in custome_node_result:
    print(row)

print("=====================================\n")
# Query 2: Find what Python is required for (multi-hop)  

def find_what_python_is_required_for():
    query = """
    MATCH path = (python {name:"Python"})-[:REQUIRED_FOR*1..3]->(target)
    RETURN
        target.name AS required_for,
        length(path) AS hops,
        [node IN nodes(path) | node.name] AS pat_nodes,
        [rel IN relationships(path) | type(rel)] AS path_relationships
    ORDER BY hops
    """
    return graph.query(query)

results_python_require = find_what_python_is_required_for()

for row in results_python_require:
    print(row)

print("=====================================\n")
# Query 3: Find the path from Mathematics to AI

def find_path_mathematics_to_ai():
    query = """
    MATCH path =
        (start {name: $start_node})
        -[*1..5]->
        (target {name: $target_node})

    RETURN
        [node IN nodes(path) | node.name] AS path_nodes,
        [rel IN relationships(path) | type(rel)] AS relationships,
        length(path) AS hops
    ORDER BY hops
    LIMIT 1
    """

    return graph.query(
        query,
        params={
            "start_node": "Mathematics",
            "target_node": "Artificial Intelligence"
        }
    )


result_path_math = find_path_mathematics_to_ai()

for row in result_path_math:
    print(row)

print("=====================================\n")
# Query 4: Find all skills that support or improve AI directly or indirectly

def find_skills_supporting_ai():
    query = """
    MATCH path =
        (skill)
        -[:SUPPORTS|IMPROVES|REQUIRED_FOR|CORE_COMPONENT_OF|
          ADVANCED_AREA_OF|EXTENDS|USED_FOR*1..5]->
        (ai {name: $ai_name})

    RETURN DISTINCT
        skill.name AS skill,
        length(path) AS hops,
        [node IN nodes(path) | node.name] AS path_nodes,
        [rel IN relationships(path) | type(rel)] AS relationships
    ORDER BY hops, skill
    """

    return graph.query(
        query,
        params={"ai_name": "Artificial Intelligence"}
    )


results_skills_support = find_skills_supporting_ai()

for row in results_skills_support:
    print(row)

