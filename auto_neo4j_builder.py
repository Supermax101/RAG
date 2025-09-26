#!/usr/bin/env python3
"""
Auto Neo4j Knowledge Graph Builder - Using Neo4j's Full Capabilities
Leverages APOC, GDS, and automatic relationship detection
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List
from neo4j import GraphDatabase


class AutoNeo4jKGBuilder:
    """Advanced Neo4j builder using automatic relationship detection"""
    
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.doc_path = Path("data/parsed/1. ASPEN Fluids and electrolytes/1. ASPEN Fluids and electrolytes.md")
        self.index_path = Path("data/parsed/1. ASPEN Fluids and electrolytes/1_aspen_fluids_and_electrolytes__5dab52.index.json")
        
    def setup_neo4j_extensions(self):
        """Install and configure Neo4j extensions for auto capabilities"""
        print("🔧 Setting up Neo4j advanced capabilities...")
        
        with self.driver.session() as session:
            # Check if APOC is available
            try:
                result = session.run("RETURN apoc.version() as version")
                apoc_version = result.single()
                if apoc_version:
                    print(f"✅ APOC available: {apoc_version['version']}")
                else:
                    print("❌ APOC not available - auto relationships will be limited")
            except:
                print("❌ APOC not available - installing automatically...")
                self.install_apoc_gds()
            
            # Check if GDS is available
            try:
                result = session.run("RETURN gds.version() as version")
                gds_version = result.single()
                if gds_version:
                    print(f"✅ GDS available: {gds_version['version']}")
                else:
                    print("❌ GDS not available - will use APOC instead")
            except:
                print("❌ GDS not available - using APOC similarity instead")
    
    def install_apoc_gds(self):
        """Install APOC and GDS via Docker restart"""
        print("🚀 Installing APOC and GDS via Docker...")
        try:
            # Stop current container
            subprocess.run(["docker", "stop", "graphrag-neo4j"], capture_output=True)
            subprocess.run(["docker", "rm", "graphrag-neo4j"], capture_output=True)
            
            # Start with APOC and GDS
            cmd = [
                "docker", "run", "-d",
                "--name", "graphrag-neo4j",
                "-p", "7474:7474",
                "-p", "7687:7687",
                "-e", "NEO4J_AUTH=neo4j/password",
                "-e", "NEO4J_PLUGINS=[\"apoc\",\"graph-data-science\"]",
                "-e", "NEO4J_apoc_export_file_enabled=true",
                "-e", "NEO4J_apoc_import_file_enabled=true",
                "neo4j:5.15"
            ]
            
            subprocess.run(cmd, check=True)
            
            print("⏳ Waiting for Neo4j with extensions to start...")
            for i in range(60):
                try:
                    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
                    with driver.session() as session:
                        session.run("RETURN 1")
                    driver.close()
                    print("✅ Neo4j with extensions ready!")
                    return True
                except:
                    time.sleep(2)
            
            print("❌ Neo4j failed to start with extensions")
            return False
            
        except Exception as e:
            print(f"❌ Failed to install extensions: {e}")
            return False
    
    def clean_database(self):
        """Clean existing database"""
        print("🧹 Cleaning existing database...")
        with self.driver.session() as session:
            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")
            print("✅ Database cleaned")
    
    def create_constraints_indexes(self):
        """Create constraints and indexes for performance"""
        print("📋 Creating constraints and indexes...")
        with self.driver.session() as session:
            constraints = [
                "CREATE CONSTRAINT clinical_section_id IF NOT EXISTS FOR (cs:ClinicalSection) REQUIRE cs.id IS UNIQUE",
                "CREATE CONSTRAINT clinical_table_id IF NOT EXISTS FOR (ct:ClinicalTable) REQUIRE ct.id IS UNIQUE",
                "CREATE INDEX section_content_index IF NOT EXISTS FOR (cs:ClinicalSection) ON (cs.content)",
                "CREATE INDEX section_type_index IF NOT EXISTS FOR (cs:ClinicalSection) ON (cs.section_type)",
                "CREATE INDEX section_name_index IF NOT EXISTS FOR (cs:ClinicalSection) ON (cs.name)"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    if "already exists" not in str(e):
                        print(f"⚠️ Constraint/Index warning: {e}")
    
    def load_clinical_data(self):
        """Load clinical sections from the ASPEN document"""
        print("📖 Loading ASPEN clinical data...")
        
        # Read the markdown content
        with open(self.doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Read the index for section structure
        with open(self.index_path, 'r', encoding='utf-8') as f:
            index_data = json.load(f)
        
        sections = self.extract_clinical_sections(content, index_data)
        
        with self.driver.session() as session:
            for i, section in enumerate(sections):
                session.run("""
                    CREATE (cs:ClinicalSection {
                        id: $id,
                        name: $name,
                        section_type: $section_type,
                        content: $content,
                        content_length: $content_length,
                        doc_source: $doc_source,
                        order_index: $order_index
                    })
                """, 
                id=f"section_{i}",
                name=section['name'],
                section_type=section['section_type'],
                content=section['content'],
                content_length=len(section['content']),
                doc_source='aspen_fluids_electrolytes',
                order_index=i
                )
        
        print(f"✅ Loaded {len(sections)} clinical sections")
        return len(sections)
    
    def extract_clinical_sections(self, content: str, index_data: dict) -> List[Dict]:
        """Extract all clinical sections with smart classification"""
        sections = []
        
        # Split content by headers
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            if line.startswith('#'):
                # Save previous section
                if current_section and current_content:
                    section_content = '\n'.join(current_content).strip()
                    if len(section_content) > 50:  # Only meaningful sections
                        sections.append({
                            'name': current_section,
                            'content': section_content,
                            'section_type': self.classify_section_smart(current_section, section_content)
                        })
                
                # Start new section
                current_section = line.strip('#').strip()
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
        
        # Save last section
        if current_section and current_content:
            section_content = '\n'.join(current_content).strip()
            if len(section_content) > 50:
                sections.append({
                    'name': current_section,
                    'content': section_content,
                    'section_type': self.classify_section_smart(current_section, section_content)
                })
        
        return sections
    
    def classify_section_smart(self, title: str, content: str) -> str:
        """Smart section classification using content analysis"""
        title_lower = title.lower()
        content_lower = content.lower()
        
        # Combine title and content for classification
        text = f"{title_lower} {content_lower}"
        
        # Medical classifications
        if any(word in text for word in ['sodium', 'potassium', 'chloride', 'electrolyte', 'hypernatremia', 'hyponatremia']):
            return 'electrolyte_physiology'
        elif any(word in text for word in ['fluid', 'water', 'hydration', 'dehydration', 'volume']):
            return 'fluid_management'
        elif any(word in text for word in ['treatment', 'management', 'therapy', 'protocol']):
            return 'treatment_protocols'
        elif any(word in text for word in ['assessment', 'evaluation', 'diagnosis', 'clinical']):
            return 'clinical_assessment'
        elif any(word in text for word in ['composition', 'solution', 'concentration']):
            return 'iv_fluid_composition'
        elif any(word in text for word in ['normal', 'range', 'reference', 'laboratory']):
            return 'reference_values'
        else:
            return 'general_clinical'
    
    def create_automatic_relationships(self):
        """Use Neo4j's automatic relationship detection capabilities"""
        print("🤖 Creating automatic relationships using Neo4j capabilities...")
        
        with self.driver.session() as session:
            
            # 1. Content Similarity Relationships (APOC)
            print("  📊 Creating content similarity relationships...")
            try:
                session.run("""
                    CALL apoc.periodic.iterate(
                        "MATCH (cs1:ClinicalSection), (cs2:ClinicalSection) 
                         WHERE cs1.id < cs2.id RETURN cs1, cs2",
                        "WITH cs1, cs2, 
                              apoc.text.sorensenDiceSimilarity(cs1.content, cs2.content) AS content_similarity,
                              apoc.text.sorensenDiceSimilarity(cs1.name, cs2.name) AS title_similarity
                         WHERE content_similarity > 0.2 OR title_similarity > 0.3
                         CREATE (cs1)-[:CONTENT_SIMILAR {
                             content_score: content_similarity, 
                             title_score: title_similarity,
                             auto_created: true
                         }]->(cs2)",
                        {batchSize: 50, parallel: false}
                    )
                """)
                print("    ✅ Content similarity relationships created")
            except Exception as e:
                print(f"    ⚠️ APOC similarity not available: {e}")
                self.create_manual_similarity(session)
            
            # 2. Section Type Relationships
            print("  🏥 Creating medical domain relationships...")
            session.run("""
                MATCH (cs1:ClinicalSection), (cs2:ClinicalSection)
                WHERE cs1.section_type = cs2.section_type AND cs1.id <> cs2.id
                CREATE (cs1)-[:SAME_DOMAIN {
                    domain: cs1.section_type,
                    auto_created: true
                }]->(cs2)
            """)
            
            # 3. Sequential Relationships (document flow)
            print("  📖 Creating document flow relationships...")
            session.run("""
                MATCH (cs1:ClinicalSection), (cs2:ClinicalSection)
                WHERE cs2.order_index = cs1.order_index + 1
                CREATE (cs1)-[:FOLLOWS {auto_created: true}]->(cs2)
            """)
            
            # 4. Keyword Co-occurrence Relationships
            print("  🔍 Creating keyword-based relationships...")
            keywords = ['sodium', 'potassium', 'fluid', 'electrolyte', 'treatment', 'management', 'clinical']
            
            for keyword in keywords:
                session.run("""
                    MATCH (cs1:ClinicalSection), (cs2:ClinicalSection)
                    WHERE cs1.id <> cs2.id
                      AND toLower(cs1.content) CONTAINS $keyword
                      AND toLower(cs2.content) CONTAINS $keyword
                    CREATE (cs1)-[:KEYWORD_RELATED {
                        keyword: $keyword,
                        auto_created: true
                    }]->(cs2)
                """, keyword=keyword)
            
            # 5. Hierarchical Relationships (based on content length and detail)
            print("  📊 Creating hierarchical relationships...")
            session.run("""
                MATCH (cs1:ClinicalSection), (cs2:ClinicalSection)
                WHERE cs1.content_length > cs2.content_length * 2
                  AND cs1.section_type = cs2.section_type
                  AND toLower(cs1.content) CONTAINS toLower(cs2.name)
                CREATE (cs1)-[:ELABORATES_ON {auto_created: true}]->(cs2)
            """)
            
            print("✅ Automatic relationships created")
    
    def create_manual_similarity(self, session):
        """Fallback manual similarity when APOC is not available"""
        print("    🔧 Using fallback similarity calculation...")
        
        # Simple keyword-based similarity
        session.run("""
            MATCH (cs1:ClinicalSection), (cs2:ClinicalSection)
            WHERE cs1.id < cs2.id
            WITH cs1, cs2,
                 size([word IN split(toLower(cs1.content), ' ') 
                       WHERE word IN split(toLower(cs2.content), ' ') 
                       AND size(word) > 3]) as common_words,
                 size(split(cs1.content, ' ')) + size(split(cs2.content, ' ')) as total_words
            WHERE common_words > 10 AND common_words * 2.0 / total_words > 0.1
            CREATE (cs1)-[:CONTENT_SIMILAR {
                common_words: common_words,
                similarity_ratio: common_words * 2.0 / total_words,
                auto_created: true
            }]->(cs2)
        """)
    
    def create_graph_projections(self):
        """Create graph projections for advanced analytics"""
        print("📊 Creating graph projections for advanced analytics...")
        
        with self.driver.session() as session:
            try:
                # Drop existing projection if it exists
                try:
                    session.run("CALL gds.graph.drop('clinicalGraph')")
                except:
                    pass
                
                # Create graph projection
                session.run("""
                    CALL gds.graph.project(
                        'clinicalGraph',
                        'ClinicalSection',
                        ['CONTENT_SIMILAR', 'SAME_DOMAIN', 'KEYWORD_RELATED', 'FOLLOWS', 'ELABORATES_ON'],
                        {
                            nodeProperties: ['content_length'],
                            relationshipProperties: ['content_score', 'title_score', 'similarity_ratio']
                        }
                    )
                """)
                
                print("✅ Graph projection created for advanced analytics")
                
                # Run community detection
                print("  🏘️ Running community detection...")
                session.run("""
                    CALL gds.louvain.write('clinicalGraph', {
                        writeProperty: 'community'
                    })
                """)
                
                # Create community relationships
                session.run("""
                    MATCH (cs1:ClinicalSection), (cs2:ClinicalSection)
                    WHERE cs1.community = cs2.community AND cs1.id <> cs2.id
                    CREATE (cs1)-[:SAME_COMMUNITY {
                        community_id: cs1.community,
                        auto_created: true
                    }]->(cs2)
                """)
                
                print("    ✅ Community detection completed")
                
            except Exception as e:
                print(f"    ⚠️ GDS not available, skipping advanced analytics: {e}")
    
    def analyze_automatic_results(self):
        """Analyze the automatically created relationships"""
        print("\n🔍 ANALYZING AUTOMATIC RELATIONSHIP RESULTS")
        print("=" * 60)
        
        with self.driver.session() as session:
            # Count nodes and relationships
            result = session.run("MATCH (n:ClinicalSection) RETURN count(n) as nodes")
            node_count = result.single()['nodes']
            
            result = session.run("MATCH ()-[r]->() RETURN count(r) as rels")
            rel_count = result.single()['rels']
            
            print(f"📊 Total: {node_count} nodes, {rel_count} relationships")
            print(f"📈 Density: {rel_count/node_count:.1f} relationships per node")
            
            # Relationship type breakdown
            result = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as rel_type, count(*) as count
                ORDER BY count DESC
            """)
            
            print("\n🔗 Automatic Relationship Types:")
            for record in result:
                print(f"  {record['rel_type']}: {record['count']}")
            
            # Check connectivity
            result = session.run("""
                MATCH (cs:ClinicalSection)
                WHERE NOT (cs)-[]-()
                RETURN count(cs) as isolated
            """)
            isolated = result.single()['isolated']
            print(f"\n✅ Isolated nodes: {isolated}/{node_count} ({100*isolated/node_count:.1f}%)")
            
            # Sample relationships
            result = session.run("""
                MATCH (cs1:ClinicalSection)-[r]->(cs2:ClinicalSection)
                WHERE r.auto_created = true
                RETURN cs1.name as from_name, type(r) as rel_type, cs2.name as to_name,
                       coalesce(r.content_score, r.similarity_ratio, r.common_words, 0) as score
                ORDER BY score DESC
                LIMIT 5
            """)
            
            print("\n🏥 Top Automatic Relationships:")
            for record in result:
                from_name = record['from_name'][:40] + "..." if len(record['from_name']) > 40 else record['from_name']
                to_name = record['to_name'][:40] + "..." if len(record['to_name']) > 40 else record['to_name']
                score = record['score']
                print(f"  {from_name}")
                print(f"    --[{record['rel_type']} ({score:.3f})]-->")
                print(f"    {to_name}")
                print()
    
    def build_complete_graph(self):
        """Build the complete automatic knowledge graph"""
        print("🚀 BUILDING AUTO NEO4J KNOWLEDGE GRAPH")
        print("🎯 Using Neo4j's Full Automatic Capabilities")
        print("=" * 70)
        
        # Setup
        self.setup_neo4j_extensions()
        self.clean_database()
        self.create_constraints_indexes()
        
        # Load data
        section_count = self.load_clinical_data()
        
        # Create automatic relationships
        self.create_automatic_relationships()
        
        # Advanced analytics
        self.create_graph_projections()
        
        # Results
        self.analyze_automatic_results()
        
        print("\n🎯 AUTO NEO4J KNOWLEDGE GRAPH COMPLETE!")
        print("🤖 All relationships automatically detected by Neo4j")
        print("🌐 Neo4j Browser: http://localhost:7474")
        print("📊 Ready for intelligent GraphRAG queries")
    
    def close(self):
        self.driver.close()


if __name__ == "__main__":
    builder = AutoNeo4jKGBuilder()
    try:
        builder.build_complete_graph()
    finally:
        builder.close()
