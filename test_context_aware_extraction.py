"""
Test Context-Aware Relation Extraction

This script tests the new pipeline where medspaCy context extraction happens
BEFORE GPT-4.1 relation extraction, ensuring that negated, historical, and
hypothetical entities don't create inappropriate causal relations.

Expected behavior:
1. Negated adverse events → NO causal relations created
2. Historical conditions → marked as historical context, not current
3. Hypothetical scenarios → NO definite causal relations
4. Positive findings → SHOULD create appropriate relations

Run: pixi run -- python test_context_aware_extraction.py
"""

import json
import logging
from uuid import uuid4
from pathlib import Path

# Setup logging and suppress noisy libraries
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress annoying medspaCy/PyRuSH log spam
logging.getLogger('root').setLevel(logging.WARNING)
logging.getLogger('PyRuSH').setLevel(logging.WARNING)
logging.getLogger('medspacy').setLevel(logging.WARNING)

# Import after logging setup
from docintel.knowledge_graph.triple_extraction import ClinicalTripleExtractor

# Test cases covering different clinical contexts
TEST_CASES = [
    {
        "name": "Negated Adverse Event (should NOT create causal relation)",
        "text": """
        Patient received niraparib 300mg daily for 6 cycles. 
        No evidence of hepatotoxicity was observed during treatment.
        Laboratory values remained within normal limits.
        """,
        "expected_entities": ["niraparib", "hepatotoxicity"],
        "expected_context": {
            "hepatotoxicity": {"is_negated": True}
        },
        "should_have_causal_relation": False,
        "explanation": "Hepatotoxicity is NEGATED, so 'niraparib causes hepatotoxicity' should NOT be created"
    },
    
    {
        "name": "Historical Condition (should mark as past, not current)",
        "text": """
        Patient with history of BRCA1 mutation was enrolled in the study.
        Patient received niraparib as maintenance therapy.
        """,
        "expected_entities": ["BRCA1 mutation", "niraparib"],
        "expected_context": {
            "BRCA1 mutation": {"is_historical": True}
        },
        "should_have_causal_relation": False,
        "explanation": "BRCA1 mutation is HISTORICAL. May have 'associated_with' relations (correct), but should NOT have strong causal relations like 'causes' or 'treats'."
    },
    
    {
        "name": "Hypothetical Scenario (should NOT create definite relation)",
        "text": """
        If grade 3 or higher toxicity occurs, niraparib dose should be reduced to 200mg.
        Patients were monitored for adverse events throughout treatment.
        """,
        "expected_entities": ["grade 3", "toxicity", "niraparib"],
        "expected_context": {
            "toxicity": {"is_hypothetical": True}
        },
        "should_have_causal_relation": False,
        "explanation": "Toxicity is HYPOTHETICAL (protocol instruction), not an observed event"
    },
    
    {
        "name": "Positive Adverse Event (SHOULD create causal relation)",
        "text": """
        Patient experienced grade 2 nausea after niraparib administration.
        Symptoms were manageable with antiemetic therapy.
        """,
        "expected_entities": ["nausea", "niraparib"],
        "expected_context": {
            "nausea": {"is_negated": False}
        },
        "should_have_causal_relation": True,
        "explanation": "Nausea is a real observed event, should create 'niraparib causes nausea' relation"
    },
    
    {
        "name": "Family History (should NOT assign to patient)",
        "text": """
        Patient's mother had ovarian cancer. Patient was diagnosed with BRCA mutation.
        Patient enrolled in niraparib maintenance study.
        """,
        "expected_entities": ["ovarian cancer", "mother"],
        "expected_context": {
            "ovarian cancer": {"is_family": True}
        },
        "should_have_causal_relation": False,
        "explanation": "Ovarian cancer is FAMILY_HISTORY. May have 'associated_with' for mother (correct), but should NOT create 'causes'/'treats' relations with patient's treatment."
    },
    
    {
        "name": "Uncertain Finding (should have lower confidence or be marked)",
        "text": """
        Imaging showed possible disease progression.
        Further evaluation with CT scan was recommended.
        """,
        "expected_entities": ["disease progression"],
        "expected_context": {
            "disease progression": {"is_uncertain": True}
        },
        "should_have_causal_relation": False,
        "explanation": "Disease progression is UNCERTAIN (possible), not confirmed"
    }
]


def print_separator(char="=", length=80):
    """Print a visual separator."""
    print(f"\n{char * length}\n")


def print_entities_with_context(entities):
    """Pretty print entities with their context flags."""
    print(f"\n📋 EXTRACTED ENTITIES ({len(entities)} total):")
    print("-" * 80)
    
    for i, entity in enumerate(entities):
        context_str = ""
        if entity.context_flags:
            flags = []
            if entity.context_flags.get('is_negated'):
                flags.append("❌ NEGATED")
            if entity.context_flags.get('is_historical'):
                flags.append("📅 HISTORICAL")
            if entity.context_flags.get('is_hypothetical'):
                flags.append("🤔 HYPOTHETICAL")
            if entity.context_flags.get('is_uncertain'):
                flags.append("❓ UNCERTAIN")
            if entity.context_flags.get('is_family'):
                flags.append("👨‍👩‍👧 FAMILY")
            
            if flags:
                context_str = f"  [{', '.join(flags)}]"
        
        confidence_bar = "█" * int(entity.confidence * 10)
        print(f"  E{i}: {entity.text:30} | {entity.entity_type:20} | {confidence_bar} {entity.confidence:.2f}{context_str}")


def print_relations(relations, entities):
    """Pretty print relations with subject/object details."""
    print(f"\n🔗 EXTRACTED RELATIONS ({len(relations)} total):")
    print("-" * 80)
    
    if not relations:
        print("  (No relations extracted)")
        return
    
    for i, rel in enumerate(relations):
        # Get context for subject and object
        subj_context = ""
        obj_context = ""
        
        if rel.subject_entity.context_flags:
            flags = [k.replace('is_', '').upper() for k, v in rel.subject_entity.context_flags.items() if v]
            if flags:
                subj_context = f" [{', '.join(flags)}]"
        
        if rel.object_entity.context_flags:
            flags = [k.replace('is_', '').upper() for k, v in rel.object_entity.context_flags.items() if v]
            if flags:
                obj_context = f" [{', '.join(flags)}]"
        
        confidence_bar = "█" * int(rel.confidence * 10)
        print(f"\n  R{i}: {rel.subject_entity.text}{subj_context}")
        print(f"      --[{rel.predicate}]-->")
        print(f"      {rel.object_entity.text}{obj_context}")
        print(f"      Confidence: {confidence_bar} {rel.confidence:.2f}")
        print(f"      Evidence: \"{rel.evidence_span[:100]}...\"" if len(rel.evidence_span) > 100 else f"      Evidence: \"{rel.evidence_span}\"")


def check_test_expectations(test_case, entities, relations):
    """Verify test case expectations and return results."""
    results = {
        "passed": True,
        "checks": []
    }
    
    # Check 1: Expected entities were extracted
    entity_texts = [e.text.lower() for e in entities]
    for expected_entity in test_case.get("expected_entities", []):
        found = any(expected_entity.lower() in et for et in entity_texts)
        results["checks"].append({
            "check": f"Entity '{expected_entity}' extracted",
            "passed": found,
            "details": f"Found in: {[et for et in entity_texts if expected_entity.lower() in et]}" if found else "Not found"
        })
        if not found:
            results["passed"] = False
    
    # Check 2: Context flags are correct
    for entity_text, expected_flags in test_case.get("expected_context", {}).items():
        matching_entities = [e for e in entities if entity_text.lower() in e.text.lower()]
        
        if matching_entities:
            entity = matching_entities[0]
            for flag_name, expected_value in expected_flags.items():
                actual_value = entity.context_flags.get(flag_name, False) if entity.context_flags else False
                passed = actual_value == expected_value
                
                results["checks"].append({
                    "check": f"Context '{flag_name}' for '{entity_text}' = {expected_value}",
                    "passed": passed,
                    "details": f"Actual: {actual_value}, Expected: {expected_value}"
                })
                
                if not passed:
                    results["passed"] = False
        else:
            results["checks"].append({
                "check": f"Entity '{entity_text}' found for context check",
                "passed": False,
                "details": "Entity not found"
            })
            results["passed"] = False
    
    # Check 3: TRUE causal relations (causes/treats, not just associations)
    # Note: 'associated_with' is legitimate for historical/family contexts
    # Only check for strong causal predicates
    causal_predicates = ["causes", "treats", "prevents", "results_in", "leads_to"]
    has_causal_relation = len(relations) > 0 and any(
        rel.predicate in causal_predicates
        for rel in relations
    )
    
    should_have = test_case.get("should_have_causal_relation", False)
    causal_check_passed = has_causal_relation == should_have
    
    causal_count = len([r for r in relations if r.predicate in causal_predicates])
    all_predicates = [r.predicate for r in relations]
    
    results["checks"].append({
        "check": f"True causal relation ({', '.join(causal_predicates)}) {'should' if should_have else 'should NOT'} exist",
        "passed": causal_check_passed,
        "details": f"Found {causal_count} causal relations. All predicates: {set(all_predicates)}"
    })
    
    if not causal_check_passed:
        results["passed"] = False
    
    return results


def run_test_case(extractor, test_case, test_num, total_tests):
    """Run a single test case and display results."""
    print_separator("=")
    print(f"🧪 TEST {test_num}/{total_tests}: {test_case['name']}")
    print_separator("=")
    
    print(f"\n📝 TEST CONTEXT:\n{test_case['text'].strip()}")
    print(f"\n💡 EXPECTED BEHAVIOR:\n{test_case['explanation']}")
    
    # Run extraction
    print(f"\n⚙️  RUNNING EXTRACTION...")
    chunk_id = uuid4()
    
    try:
        result = extractor.extract_triples(test_case['text'], chunk_id)
        
        # Display results
        print_entities_with_context(result.entities)
        print_relations(result.relations, result.entities)
        
        # Check expectations
        print(f"\n✅ VALIDATION CHECKS:")
        print("-" * 80)
        
        validation_results = check_test_expectations(test_case, result.entities, result.relations)
        
        for check in validation_results["checks"]:
            status = "✅ PASS" if check["passed"] else "❌ FAIL"
            print(f"  {status}: {check['check']}")
            if not check["passed"]:
                print(f"         Details: {check['details']}")
        
        # Overall result
        print_separator("-")
        if validation_results["passed"]:
            print("🎉 TEST PASSED!")
            return True
        else:
            print("⚠️  TEST FAILED - See validation checks above")
            return False
            
    except Exception as e:
        print(f"\n❌ TEST ERROR: {str(e)}")
        logger.exception("Test execution failed")
        return False


def main():
    """Run all test cases."""
    print_separator("=")
    print("🔬 CONTEXT-AWARE RELATION EXTRACTION TEST SUITE")
    print("Testing: medspaCy context extraction → GPT-4.1 relation extraction")
    print_separator("=")
    
    # Initialize extractor (NOT in fast mode - we want GPT-4.1 + medspaCy)
    print("⚙️  Initializing ClinicalTripleExtractor...")
    extractor = ClinicalTripleExtractor(fast_mode=False, skip_relations=False)
    print("✅ Extractor initialized\n")
    
    # Run all test cases
    results = []
    total_tests = len(TEST_CASES)
    
    for i, test_case in enumerate(TEST_CASES, 1):
        passed = run_test_case(extractor, test_case, i, total_tests)
        results.append({
            "name": test_case["name"],
            "passed": passed
        })
    
    # Summary
    print_separator("=")
    print("📊 TEST SUMMARY")
    print_separator("=")
    
    passed_count = sum(1 for r in results if r["passed"])
    failed_count = total_tests - passed_count
    
    for i, result in enumerate(results, 1):
        status = "✅ PASS" if result["passed"] else "❌ FAIL"
        print(f"{i}. {status}: {result['name']}")
    
    print_separator("-")
    print(f"Total: {total_tests} tests")
    print(f"Passed: {passed_count} ({passed_count/total_tests*100:.0f}%)")
    print(f"Failed: {failed_count} ({failed_count/total_tests*100:.0f}%)")
    print_separator("=")
    
    # Save detailed results to JSON
    output_file = Path("logs") / "context_aware_test_results.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            "summary": {
                "total": total_tests,
                "passed": passed_count,
                "failed": failed_count,
                "pass_rate": f"{passed_count/total_tests*100:.1f}%"
            },
            "results": results
        }, f, indent=2)
    
    print(f"📄 Detailed results saved to: {output_file}")
    
    # Exit code
    exit(0 if failed_count == 0 else 1)


if __name__ == "__main__":
    main()
