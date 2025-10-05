#!/usr/bin/env python3
"""
Test script to demonstrate medspaCy-based vocabulary loading.
This shows how medspaCy can replace licensed vocabularies for development.
"""

import logging
import asyncio
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_medspacy_vocabularies():
    """Test medspaCy-based vocabulary loading."""
    print("🧪 Testing medspaCy-based Vocabulary Loading")
    print("=" * 50)
    
    try:
        from docintel.repository.medspacy_loaders import (
            MedSpaCyUMLSLoader,
            MedSpaCyEnhancedRxNormLoader, 
            MedSpaCyClinicalSnomedLoader
        )
        
        # Test UMLS loader
        print("\n📋 Testing medspaCy UMLS Loader:")
        umls_loader = MedSpaCyUMLSLoader()
        print(f"  Description: {umls_loader.describe()}")
        print(f"  Version: {umls_loader.source_version()}")
        
        print("  Sample UMLS concepts:")
        for i, node in enumerate(umls_loader.iter_nodes()):
            if i >= 5:  # Show first 5
                break
            print(f"    • {node.code}: {node.display_name}")
            if node.metadata.get("synonyms"):
                print(f"      Synonyms: {', '.join(node.metadata['synonyms'][:3])}")
        
        print("  Sample UMLS relationships:")
        for i, edge in enumerate(umls_loader.iter_edges()):
            if i >= 3:  # Show first 3
                break
            print(f"    • {edge.source_code} --{edge.predicate}--> {edge.target_code}")
        
        # Test RxNorm loader
        print("\n💊 Testing medspaCy Enhanced RxNorm Loader:")
        rxnorm_loader = MedSpaCyEnhancedRxNormLoader()
        print(f"  Description: {rxnorm_loader.describe()}")
        
        print("  Sample medications:")
        for i, node in enumerate(rxnorm_loader.iter_nodes()):
            if i >= 4:  # Show first 4
                break
            print(f"    • {node.code}: {node.display_name}")
            if node.metadata.get("drug_class"):
                print(f"      Class: {node.metadata['drug_class']}")
        
        # Test SNOMED loader
        print("\n🏥 Testing medspaCy Clinical SNOMED Loader:")
        snomed_loader = MedSpaCyClinicalSnomedLoader()
        print(f"  Description: {snomed_loader.describe()}")
        
        print("  Sample clinical concepts:")
        for i, node in enumerate(snomed_loader.iter_nodes()):
            if i >= 4:  # Show first 4
                break
            print(f"    • {node.code}: {node.display_name}")
            if node.metadata.get("category"):
                print(f"      Category: {node.metadata['category']}")
        
        print("\n✅ medspaCy vocabulary loading test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ medspaCy vocabularies not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing medspaCy vocabularies: {e}")
        return False

async def test_medspacy_entity_extraction():
    """Test medspaCy entity extraction capabilities."""
    print("\n🔍 Testing medspaCy Entity Extraction")
    print("=" * 40)
    
    try:
        import medspacy
        
        # Load medspaCy pipeline
        nlp = medspacy.load()
        print(f"📦 Loaded medspaCy pipeline with components: {nlp.pipe_names}")
        
        # Test clinical text
        test_text = """
        Patient has a history of myocardial infarction and diabetes mellitus.
        Currently taking aspirin 81mg daily and metformin 500mg twice daily.
        No evidence of pneumonia. Continue current medications.
        Blood pressure is elevated at 150/90 mmHg.
        """
        
        # Process with medspaCy
        doc = nlp(test_text)
        
        print(f"\n📝 Sample clinical text processed:")
        print(f"   \"{test_text.strip()}\"")
        
        print(f"\n🏷️  Extracted entities:")
        for ent in doc.ents:
            context_info = ""
            if hasattr(ent, 'is_negated') and ent.is_negated:
                context_info += " [NEGATED]"
            if hasattr(ent, 'is_uncertain') and ent.is_uncertain:
                context_info += " [UNCERTAIN]"
                
            print(f"    • {ent.text} ({ent.label_}){context_info}")
        
        print(f"\n📊 Entity extraction summary:")
        print(f"    Total entities: {len(doc.ents)}")
        print(f"    Entity types: {set(ent.label_ for ent in doc.ents)}")
        
        print("\n✅ medspaCy entity extraction test completed!")
        return True
        
    except ImportError as e:
        print(f"❌ medspaCy not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing medspaCy extraction: {e}")
        return False

async def demonstrate_integration():
    """Demonstrate how medspaCy integrates with the existing pipeline."""
    print("\n🔗 Demonstrating medspaCy Integration")
    print("=" * 40)
    
    try:
        from docintel.config import get_config
        
        config = get_config()
        repo_settings = config.repository_ingestion
        
        print(f"📋 Repository settings:")
        print(f"    Use medspaCy vocabularies: {repo_settings.use_medspacy_vocabularies}")
        print(f"    medspaCy QuickUMLS path: {repo_settings.medspacy_quickumls_path}")
        
        # Test building loaders
        from docintel.repository.ingestion import build_loaders
        loaders = build_loaders(repo_settings)
        
        print(f"\n📚 Available vocabulary loaders: {len(loaders)}")
        for loader in loaders:
            print(f"    • {loader.vocabulary}: {loader.describe()}")
        
        print("\n✅ Integration demonstration complete!")
        return True
        
    except Exception as e:
        print(f"❌ Error demonstrating integration: {e}")
        return False

async def main():
    """Main test function."""
    print("🚀 medspaCy Vocabulary System Test")
    print("=" * 50)
    
    success_count = 0
    total_tests = 3
    
    # Test 1: Vocabulary loading
    if await test_medspacy_vocabularies():
        success_count += 1
    
    # Test 2: Entity extraction
    if await test_medspacy_entity_extraction():
        success_count += 1
    
    # Test 3: Integration
    if await demonstrate_integration():
        success_count += 1
    
    print(f"\n📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! medspaCy is ready to replace licensed vocabularies.")
        print("\n💡 Next steps:")
        print("   1. Set DOCINTEL_REPO_USE_MEDSPACY_VOCABULARIES=true in your .env file")
        print("   2. Run: pixi run ingest-vocab --vocab umls --vocab rxnorm --vocab snomed")
        print("   3. Your pipeline will use medspaCy's QuickUMLS + clinical patterns instead of licensed vocabularies")
    else:
        print("⚠️  Some tests failed. Check medspaCy installation and dependencies.")

if __name__ == "__main__":
    asyncio.run(main())