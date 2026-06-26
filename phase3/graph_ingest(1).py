from neo4j import GraphDatabase




neo_connection = GraphDatabase.driver("neo4j://localhost:7687", auth=("neo4j", "password123"))

Topic_nodes = [
  "Artificial Intelligence",
  "Computer Science",
  "Machine Learning",
  "Deep Learning"
]

Skill_nodes = [
  "Programming",
  "Python",
  "Mathematics",
  "Data Analysis",
  "Cloud Computing"
]

Performance_node = [
  "AI Model Performance"
]

def create_topic_nodes(neo_connection):
  try:
    with neo_connection.session() as session:
      for topic in Topic_nodes:
        session.run("MERGE (t:Topic {name : $name})", name= topic)
        print(f"Created topic node: {topic}")
  except Exception as e:
    print(f"Error creating topic nodes: {e}")

def create_skill_nodes(neo_connection):
  try:
    with neo_connection.session() as session:
      for skill in Skill_nodes:
        session.run("MERGE (s:Skill {name : $name})", name= skill)
        print(f"Created skill node: {skill}")
  except Exception as e:
    print(f"Error creating skill nodes: {e}")

def create_performance_node(neo_connection):
  try:
    with neo_connection.session() as session:
      for performance in Performance_node:
        session.run("MERGE (p:Performance {name: $name})", name= performance)
        print(f"Created performance node: {performance}")
  except Exception as e:
    print(f"Error creating performance node: {e}")

relationships = [
    {"from": "Artificial Intelligence", "to": "Computer Science",      "type": "IS_BRANCH_OF"},
    {"from": "Artificial Intelligence", "to": "Programming",           "type": "REQUIRES_SKILL"},
    {"from": "Python",                  "to": "Artificial Intelligence","type": "USED_FOR"},
    {"from": "Machine Learning",        "to": "Artificial Intelligence","type": "CORE_COMPONENT_OF"},
    {"from": "Data Analysis",           "to": "AI Model Performance",  "type": "IMPROVES"},
    {"from": "Mathematics",             "to": "Machine Learning",      "type": "REQUIRED_FOR"},
    {"from": "Deep Learning",           "to": "Artificial Intelligence","type": "ADVANCED_AREA_OF"},
    {"from": "Python",                  "to": "Deep Learning",         "type": "REQUIRED_FOR"},
    {"from": "Deep Learning",           "to": "Machine Learning",      "type": "EXTENDS"},
    {"from": "Cloud Computing",         "to": "Artificial Intelligence","type": "SUPPORTS"},
]

def create_relationships(neo_connection):
    with neo_connection.session() as session:
        for rel in relationships:
            session.run("""
                MATCH (a {name: $from_name})
                MATCH (b {name: $to_name})
                MERGE (a)-[r:""" + rel["type"] + """]->(b)
            """, from_name=rel["from"], to_name=rel["to"])
            print(f"Created: ({rel['from']}) -[{rel['type']}]-> ({rel['to']})")


def main():
  create_topic_nodes(neo_connection)
  create_skill_nodes(neo_connection)
  create_performance_node(neo_connection)
  create_relationships(neo_connection)

if __name__ == "__main__":
  main()