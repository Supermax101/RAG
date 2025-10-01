#!/usr/bin/env python3
"""
Clinical-Focused Knowledge Graph Builder
Extracts ALL clinical content from ASPEN document based on index structure
"""

import json
import re
from pathlib import Path
from neo4j import GraphDatabase
from typing import List, Dict, Tuple

class ClinicalKGBuilder:
    """Build knowledge graph focused on ALL clinical content from ASPEN document"""
    
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="medicalpass123"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.doc_id = "aspen_complete_clinical"

    def load_index_structure(self) -> Dict:
        """Load the document index to understand structure"""
        index_path = "data/parsed/1. ASPEN Fluids and electrolytes/1_aspen_fluids_and_electrolytes__5dab52.index.json"
        with open(index_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def identify_clinical_sections(self, index_data: Dict) -> List[str]:
        """Identify all clinical sections (filter out administrative)"""
        clinical_keywords = [
            'fluid', 'water', 'electrolyte', 'sodium', 'potassium', 'calcium', 
            'magnesium', 'phosph', 'clinical', 'management', 'treatment', 
            'assessment', 'parenteral', 'composition', 'deficit', 'overload',
            'hypo', 'hyper', 'algorithm', 'refeeding', 'siadh', 'edema',
            'gastrointestinal', 'losses', 'requirements', 'calculation'
        ]
        
        administrative_keywords = [
            'equal opportunity', 'accreditation', 'summer 2025', 'ce credit',
            'access recording', 'commercial support', 'session handout', 
            'policies', 'pharmd', 'bcnsp', 'deadline', 'disclosure', 'important'
        ]
        
        clinical_sections = set()
        
        for block in index_data.get('blocks', []):
            section = block.get('section', '').lower()
            
            # Must contain clinical keywords AND not be administrative
            is_clinical = any(keyword in section for keyword in clinical_keywords)
            is_admin = any(keyword in section for keyword in administrative_keywords)
            
            if is_clinical and not is_admin and len(section) > 3:
                clinical_sections.add(block.get('section', ''))
        
        return list(clinical_sections)

    def extract_clinical_content_by_section(self, content: str, clinical_sections: List[str]) -> Dict[str, str]:
        """Extract content for each clinical section"""
        section_content = {}
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            # Check if this line starts a new clinical section
            line_clean = line.strip()
            
            # Header detection
            if line_clean.startswith('#'):
                header_text = re.sub(r'^#+\s*', '', line_clean)
                
                # Save previous section
                if current_section and current_content:
                    section_content[current_section] = '\n'.join(current_content)
                
                # Check if this is a clinical section
                if any(clinical_sec.lower() == header_text.lower() for clinical_sec in clinical_sections):
                    current_section = header_text
                    current_content = [line]
                else:
                    current_section = None
                    current_content = []
            else:
                # Add content to current section
                if current_section:
                    current_content.append(line)
        
        # Save final section
        if current_section and current_content:
            section_content[current_section] = '\n'.join(current_content)
        
        return section_content

    def classify_clinical_section(self, section_name: str) -> str:
        """Classify clinical sections into meaningful categories"""
        section_lower = section_name.lower()
        
        # Fluid Management
        if any(word in section_lower for word in ['water', 'fluid', 'tbw', 'exchange', 'deficit', 'overload', 'edema']):
            return 'fluid_management'
        
        # IV Fluid Compositions
        elif any(word in section_lower for word in ['composition', 'crystalloid', 'plasma', 'gastrointestinal', 'losses']):
            return 'iv_fluid_composition'
        
        # Electrolyte Disorders
        elif any(word in section_lower for word in ['sodium', 'potassium', 'calcium', 'magnesium', 'phosph']):
            if any(word in section_lower for word in ['hypo', 'hyper', 'management', 'algorithm']):
                return 'electrolyte_disorders'
            else:
                return 'electrolyte_physiology'
        
        # Clinical Assessment
        elif any(word in section_lower for word in ['assessment', 'evaluation', 'clinical presentation', 'workup']):
            return 'clinical_assessment'
        
        # Treatment Protocols
        elif any(word in section_lower for word in ['management', 'treatment', 'algorithm', 'consensus', 'refeeding']):
            return 'treatment_protocols'
        
        # Calculations & Requirements
        elif any(word in section_lower for word in ['calculating', 'calculation', 'requirements', 'formula']):
            return 'calculations'
        
        # Special Conditions
        elif any(word in section_lower for word in ['siadh', 'refeeding', 'syndrome']):
            return 'special_conditions'
        
        # Parenteral/Enteral
        elif any(word in section_lower for word in ['parenteral', 'oral', 'supplement']):
            return 'administration_routes'
        
        else:
            return 'general_clinical'

    def extract_clinical_tables(self, content: str) -> List[Dict]:
        """Extract all clinical tables with enhanced classification"""
        tables = []
        table_pattern = r'\|.*?\|.*?\n(?:\|.*?\|.*?\n)*'
        
        for match in re.finditer(table_pattern, content, re.MULTILINE):
            table_text = match.group(0)
            lines = [line.strip() for line in table_text.split('\n') if line.strip()]
            
            if len(lines) >= 2:
                # Parse header
                header = [cell.strip() for cell in lines[0].split('|') if cell.strip()]
                
                # Parse data rows (skip separator line with ---)
                data_rows = []
                for line in lines[1:]:
                    if '---' not in line:
                        row = [cell.strip() for cell in line.split('|') if cell.strip()]
                        if len(row) >= 2:
                            data_rows.append(row)
                
                if data_rows and header:
                    table_type = self.classify_clinical_table(header, data_rows)
                    tables.append({
                        'header': header,
                        'rows': data_rows,
                        'table_type': table_type,
                        'raw_text': table_text,
                        'clinical_significance': self.assess_table_significance(header, data_rows)
                    })
        
        return tables

    def classify_clinical_table(self, header: List[str], rows: List[List[str]]) -> str:
        """Classify tables based on clinical content"""
        header_text = ' '.join(header).lower()
        sample_data = ' '.join([' '.join(row) for row in rows[:3]]).lower()
        combined_text = header_text + ' ' + sample_data
        
        # Fluid Composition Tables
        if any(word in combined_text for word in ['meq/l', 'fluid', 'plasma', 'nacl', 'tonicity', 'crystalloid']):
            return 'fluid_composition'
        
        # Age/Weight Calculation Tables
        elif any(word in combined_text for word in ['age', 'kg', 'weight', 'formula', 'ml/kg']):
            return 'calculation_reference'
        
        # Normal Values/Lab Reference
        elif any(word in combined_text for word in ['normal', 'serum', 'concentrations', 'adult', 'reference']):
            return 'reference_values'
        
        # Clinical Assessment Tables
        elif any(word in combined_text for word in ['symptom', 'clinical', 'assessment', 'presentation']):
            return 'clinical_assessment'
        
        # Treatment/Management Tables
        elif any(word in combined_text for word in ['dose', 'treatment', 'management', 'agent', 'route']):
            return 'treatment_protocols'
        
        # GI Losses/Body Fluid Tables
        elif any(word in combined_text for word in ['stomach', 'bile', 'pancreas', 'body fluid', 'losses']):
            return 'body_fluid_composition'
        
        else:
            return 'clinical_data'

    def assess_table_significance(self, header: List[str], rows: List[List[str]]) -> str:
        """Assess the clinical significance of a table"""
        if any('meq' in ' '.join(header).lower() for _ in [1]):
            return 'Critical for electrolyte replacement calculations'
        elif any('fluid' in ' '.join(header).lower() for _ in [1]):
            return 'Essential for fluid management decisions'
        elif any('age' in ' '.join(header).lower() or 'weight' in ' '.join(header).lower() for _ in [1]):
            return 'Required for dose calculations'
        else:
            return 'Important clinical reference data'

    def create_comprehensive_knowledge_graph(self):
        """Create comprehensive clinical knowledge graph"""
        with self.driver.session() as session:
            # Clean existing data
            session.run("MATCH (n) DETACH DELETE n")
            print("🧹 Cleaned existing database")
            
            # Load index and identify clinical sections
            index_data = self.load_index_structure()
            clinical_sections = self.identify_clinical_sections(index_data)
            print(f"📋 Identified {len(clinical_sections)} clinical sections")
            
            # Load document content
            doc_path = "data/parsed/1. ASPEN Fluids and electrolytes/1. ASPEN Fluids and electrolytes.md"
            content = Path(doc_path).read_text(encoding='utf-8')
            
            # Extract content by clinical section
            section_content = self.extract_clinical_content_by_section(content, clinical_sections)
            print(f"📖 Extracted content for {len(section_content)} sections")
            
            # Create clinical section nodes
            for section_name, section_text in section_content.items():
                if len(section_text) > 100:  # Only meaningful content
                    section_type = self.classify_clinical_section(section_name)
                    
                    session.run("""
                        CREATE (cs:ClinicalSection {
                            name: $name,
                            content: $content,
                            section_type: $section_type,
                            doc_source: $doc_source,
                            content_length: $content_length
                        })
                    """, 
                    name=section_name,
                    content=section_text[:5000],  # Limit for performance
                    section_type=section_type,
                    doc_source=self.doc_id,
                    content_length=len(section_text)
                    )
            
            # Extract and store clinical tables
            tables = self.extract_clinical_tables(content)
            for i, table in enumerate(tables):
                session.run("""
                    CREATE (ct:ClinicalTable {
                        table_id: $table_id,
                        table_type: $table_type,
                        headers: $headers,
                        clinical_significance: $significance,
                        raw_content: $raw_content,
                        doc_source: $doc_source
                    })
                """,
                table_id=f"table_{i}",
                table_type=table['table_type'],
                headers=str(table['header']),
                significance=table['clinical_significance'],
                raw_content=table['raw_text'][:2000],
                doc_source=self.doc_id
                )
            
            # Create clinical relationships
            self.create_clinical_relationships(session)
            
            print("✅ Comprehensive clinical knowledge graph created")

    def create_clinical_relationships(self, session):
        """Create comprehensive meaningful clinical relationships"""
        print("🔗 Creating comprehensive clinical relationships...")
        
        # 1. Link all fluid management sections together
        session.run("""
            MATCH (cs1:ClinicalSection {section_type: 'fluid_management'})
            MATCH (cs2:ClinicalSection {section_type: 'fluid_management'})
            WHERE cs1.name <> cs2.name
            CREATE (cs1)-[:RELATED_TO]->(cs2)
        """)
        
        # 2. Link all electrolyte sections together
        session.run("""
            MATCH (cs1:ClinicalSection)
            MATCH (cs2:ClinicalSection)
            WHERE cs1.section_type IN ['electrolyte_physiology', 'electrolyte_disorders']
              AND cs2.section_type IN ['electrolyte_physiology', 'electrolyte_disorders']
              AND cs1.name <> cs2.name
            CREATE (cs1)-[:ELECTROLYTE_RELATED]->(cs2)
        """)
        
        # 3. Link treatment protocols to clinical assessments
        session.run("""
            MATCH (cs1:ClinicalSection {section_type: 'treatment_protocols'})
            MATCH (cs2:ClinicalSection {section_type: 'clinical_assessment'})
            CREATE (cs1)-[:GUIDES_TREATMENT_FOR]->(cs2)
        """)
        
        # 4. Link IV fluid composition to fluid management
        session.run("""
            MATCH (cs1:ClinicalSection {section_type: 'iv_fluid_composition'})
            MATCH (cs2:ClinicalSection {section_type: 'fluid_management'})
            CREATE (cs1)-[:SUPPORTS_MANAGEMENT]->(cs2)
        """)
        
        # 5. Link special conditions to treatment protocols
        session.run("""
            MATCH (cs1:ClinicalSection {section_type: 'special_conditions'})
            MATCH (cs2:ClinicalSection {section_type: 'treatment_protocols'})
            CREATE (cs1)-[:REQUIRES_SPECIAL_PROTOCOL]->(cs2)
        """)
        
        # 6. Create content-based relationships (more flexible)
        session.run("""
            MATCH (cs1:ClinicalSection), (cs2:ClinicalSection)
            WHERE cs1.name <> cs2.name
              AND (
                (toLower(cs1.name) CONTAINS 'sodium' AND toLower(cs2.name) CONTAINS 'sodium') OR
                (toLower(cs1.name) CONTAINS 'potassium' AND toLower(cs2.name) CONTAINS 'potassium') OR
                (toLower(cs1.name) CONTAINS 'calcium' AND toLower(cs2.name) CONTAINS 'calcium') OR
                (toLower(cs1.name) CONTAINS 'magnesium' AND toLower(cs2.name) CONTAINS 'magnesium') OR
                (toLower(cs1.name) CONTAINS 'fluid' AND toLower(cs2.name) CONTAINS 'fluid') OR
                (toLower(cs1.name) CONTAINS 'water' AND toLower(cs2.name) CONTAINS 'water')
              )
            CREATE (cs1)-[:CONTENT_RELATED]->(cs2)
        """)
        
        # 7. Link management sections to their base concepts
        session.run("""
            MATCH (cs1:ClinicalSection), (cs2:ClinicalSection)
            WHERE toLower(cs1.name) CONTAINS 'management' 
              AND cs1.name <> cs2.name
              AND NOT cs1.section_type = cs2.section_type
            CREATE (cs1)-[:MANAGES_CONDITION]->(cs2)
        """)
        
        # 8. Link all tables to relevant sections
        session.run("""
            MATCH (cs:ClinicalSection), (ct:ClinicalTable)
            WHERE cs.section_type = 'iv_fluid_composition' 
              AND ct.table_type = 'fluid_composition'
            CREATE (cs)-[:CONTAINS_TABLE]->(ct)
        """)
        
        session.run("""
            MATCH (cs:ClinicalSection), (ct:ClinicalTable)
            WHERE cs.section_type = 'fluid_management' 
              AND ct.table_type = 'calculation_reference'
            CREATE (cs)-[:USES_TABLE]->(ct)
        """)
        
        # 9. Create hierarchical relationships
        session.run("""
            MATCH (cs1:ClinicalSection), (cs2:ClinicalSection)
            WHERE cs1.section_type = 'general_clinical'
              AND cs2.section_type IN ['fluid_management', 'electrolyte_physiology', 'clinical_assessment']
            CREATE (cs1)-[:ENCOMPASSES]->(cs2)
        """)
        
        print("✅ Comprehensive clinical relationships created")

    def show_clinical_results(self):
        """Show the comprehensive clinical knowledge graph"""
        with self.driver.session() as session:
            # Show section distribution
            result = session.run("""
                MATCH (cs:ClinicalSection)
                RETURN cs.section_type as section_type, count(*) as count
                ORDER BY count DESC
            """)
            
            print("\n📊 Clinical Section Distribution:")
            for record in result:
                print(f"  {record['section_type']}: {record['count']} sections")
            
            # Show table distribution
            result = session.run("""
                MATCH (ct:ClinicalTable)
                RETURN ct.table_type as table_type, count(*) as count
                ORDER BY count DESC
            """)
            
            print("\n📋 Clinical Table Distribution:")
            for record in result:
                print(f"  {record['table_type']}: {record['count']} tables")
            
            # Show sample clinical content
            result = session.run("""
                MATCH (cs:ClinicalSection)
                WHERE cs.section_type IN ['fluid_management', 'electrolyte_disorders', 'treatment_protocols']
                RETURN cs.name, cs.section_type
                LIMIT 10
            """)
            
            print("\n🏥 Sample Clinical Sections:")
            for record in result:
                print(f"  • {record['cs.name']} ({record['cs.section_type']})")

    def close(self):
        self.driver.close()

if __name__ == "__main__":
    builder = ClinicalKGBuilder()
    try:
        print("🚀 Building COMPREHENSIVE Clinical Knowledge Graph")
        print("🎯 Focus: ALL clinical content from ASPEN document")
        print("=" * 70)
        
        builder.create_comprehensive_knowledge_graph()
        builder.show_clinical_results()
        
        print("\n🎯 Clinical Knowledge Graph Complete!")
        print("🏥 ALL clinical content now accessible for medical queries")
        print("🌐 Neo4j Browser: http://localhost:7474")
        
    finally:
        builder.close()
