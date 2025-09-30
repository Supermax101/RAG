#!/usr/bin/env python3
"""
Build TPN Knowledge Graph from all 52 parsed documents
"""

import asyncio
import json
from pathlib import Path
from neo4j import GraphDatabase
from typing import List, Dict
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rag.config.settings import settings
from src.rag.infrastructure.llm_providers.ollama_provider import OllamaLLMProvider


class TPNKnowledgeGraphBuilder:
    """Build TPN clinical knowledge graph from parsed documents"""
    
    def __init__(self, neo4j_uri="bolt://localhost:7687", neo4j_user="neo4j", neo4j_password="yourpassword"):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.llm_provider = OllamaLLMProvider(default_model="mistral-nemo:latest")
        self.parsed_dir = settings.parsed_dir
    
    def clear_graph(self):
        """Clear existing graph"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("✅ Cleared existing graph")
    
    async def extract_entities_and_relationships(self, text: str, doc_name: str) -> Dict:
        """Use LLM to extract TPN entities and relationships"""
        
        prompt = f"""Extract TPN clinical entities and relationships from this text.

Text: {text[:2000]}

Extract:
1. Entities (TPN components, conditions, symptoms, lab values, drugs)
2. Relationships (causes, treats, monitors, contraindicates)

Respond in JSON:
{{
  "entities": [
    {{"name": "Dextrose 10%", "type": "TPN_Component"}},
    {{"name": "Hyperglycemia", "type": "Condition"}}
  ],
  "relationships": [
    {{"from": "Dextrose 10%", "to": "Hyperglycemia", "type": "CAUSES", "condition": "if >10mg/kg/min"}}
  ]
}}"""

        try:
            response = await self.llm_provider.generate(prompt, max_tokens=1000)
            # Try to parse JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"entities": [], "relationships": []}
        except Exception as e:
            print(f"Warning: Entity extraction failed for {doc_name}: {e}")
            return {"entities": [], "relationships": []}
    
    def create_entities(self, entities: List[Dict], doc_name: str):
        """Create entity nodes in Neo4j"""
        with self.driver.session() as session:
            for entity in entities:
                session.run("""
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type, e.source_doc = $doc
                """, name=entity['name'], type=entity['type'], doc=doc_name)
    
    def create_relationships(self, relationships: List[Dict]):
        """Create relationships in Neo4j"""
        with self.driver.session() as session:
            for rel in relationships:
                session.run("""
                    MATCH (a:Entity {name: $from})
                    MATCH (b:Entity {name: $to})
                    MERGE (a)-[r:RELATES {type: $type}]->(b)
                    SET r.condition = $condition
                """, 
                from_=rel['from'], 
                to=rel['to'], 
                type=rel['type'],
                condition=rel.get('condition', ''))
    
    async def build_graph_from_documents(self):
        """Build complete TPN knowledge graph"""
        
        print("🔨 Building TPN Knowledge Graph from 52 documents...")
        
        # Clear existing graph
        self.clear_graph()
        
        # Process each document
        doc_dirs = [d for d in self.parsed_dir.iterdir() if d.is_dir()]
        
        for i, doc_dir in enumerate(doc_dirs, 1):
            md_files = list(doc_dir.glob("*.md"))
            if not md_files:
                continue
            
            md_file = md_files[0]
            doc_name = doc_dir.name
            
            print(f"\n📄 Processing {i}/{len(doc_dirs)}: {doc_name}")
            
            # Read document
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract entities and relationships
            extracted = await self.extract_entities_and_relationships(content, doc_name)
            
            # Create in Neo4j
            if extracted['entities']:
                self.create_entities(extracted['entities'], doc_name)
                print(f"  ✅ Created {len(extracted['entities'])} entities")
            
            if extracted['relationships']:
                self.create_relationships(extracted['relationships'])
                print(f"  ✅ Created {len(extracted['relationships'])} relationships")
        
        print("\n✅ Knowledge graph build complete!")
        
        # Show stats
        with self.driver.session() as session:
            entity_count = session.run("MATCH (e:Entity) RETURN count(e) as count").single()['count']
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()['count']
            print(f"\n📊 Graph Stats:")
            print(f"   - Entities: {entity_count}")
            print(f"   - Relationships: {rel_count}")
    
    def close(self):
        self.driver.close()


async def main():
    """Main function"""
    builder = TPNKnowledgeGraphBuilder(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j", 
        neo4j_password="yourpassword"
    )
    
    try:
        await builder.build_graph_from_documents()
    finally:
        builder.close()


if __name__ == "__main__":
    asyncio.run(main())
