"""
Enhanced medspaCy configuration with clinical trial-specific entity rules.
"""

import medspacy
from medspacy.ner import TargetRule

def create_clinical_trial_nlp():
    """Create a medspaCy pipeline optimized for clinical trial documents."""
    
    # Load the base medspaCy model
    nlp = medspacy.load()
    
    # Get the target matcher component
    target_matcher = nlp.get_pipe("medspacy_target_matcher")
    
    # Add clinical trial-specific entity extraction rules
    clinical_rules = [
        # Cardiovascular conditions
        TargetRule("myocardial infarction", "CONDITION", 
                  pattern=[{"LOWER": {"IN": ["myocardial", "mi"]}}, {"LOWER": "infarction", "OP": "?"}]),
        TargetRule("heart failure", "CONDITION"),
        TargetRule("atrial fibrillation", "CONDITION", 
                  pattern=[{"LOWER": "atrial"}, {"LOWER": "fibrillation"}]),
        TargetRule("afib", "CONDITION"),
        TargetRule("hypertension", "CONDITION"),
        TargetRule("high blood pressure", "CONDITION", literal="hypertension"),
        
        # Diabetes-related
        TargetRule("diabetes mellitus", "CONDITION"),
        TargetRule("diabetes", "CONDITION"),
        TargetRule("type 2 diabetes", "CONDITION", 
                  pattern=[{"LOWER": "type"}, {"LOWER": {"IN": ["2", "ii", "two"]}}, 
                          {"LOWER": {"IN": ["diabetes", "dm"]}}]),
        TargetRule("hyperglycemia", "CONDITION"),
        TargetRule("hypoglycemia", "CONDITION"),
        
        # Common medications
        TargetRule("aspirin", "MEDICATION"),
        TargetRule("metformin", "MEDICATION"),
        TargetRule("lisinopril", "MEDICATION"),
        TargetRule("atorvastatin", "MEDICATION"),
        TargetRule("simvastatin", "MEDICATION"),
        TargetRule("warfarin", "MEDICATION"),
        TargetRule("insulin", "MEDICATION"),
        
        # Dosage patterns
        TargetRule("mg", "DOSAGE"),
        TargetRule("mcg", "DOSAGE"),
        TargetRule("units", "DOSAGE"),
        TargetRule("daily", "FREQUENCY"),
        TargetRule("twice daily", "FREQUENCY"),
        TargetRule("bid", "FREQUENCY"),
        TargetRule("qid", "FREQUENCY"),
        
        # Clinical measurements
        TargetRule("blood pressure", "MEASUREMENT"),
        TargetRule("systolic", "MEASUREMENT"),
        TargetRule("diastolic", "MEASUREMENT"),
        TargetRule("mmHg", "UNIT"),
        TargetRule("mg/dL", "UNIT"),
        TargetRule("hemoglobin A1c", "LAB_TEST"),
        TargetRule("HbA1c", "LAB_TEST"),
        
        # Adverse events
        TargetRule("adverse event", "ADVERSE_EVENT"),
        TargetRule("side effect", "ADVERSE_EVENT"),
        TargetRule("serious adverse event", "SERIOUS_ADVERSE_EVENT"),
        TargetRule("SAE", "SERIOUS_ADVERSE_EVENT"),
        
        # Clinical trial terms
        TargetRule("randomization", "STUDY_DESIGN"),
        TargetRule("placebo", "INTERVENTION"),
        TargetRule("treatment arm", "STUDY_DESIGN"),
        TargetRule("primary endpoint", "ENDPOINT"),
        TargetRule("secondary endpoint", "ENDPOINT"),
        TargetRule("inclusion criteria", "CRITERIA"),
        TargetRule("exclusion criteria", "CRITERIA"),
        
        # Safety terms
        TargetRule("contraindication", "SAFETY"),
        TargetRule("black box warning", "SAFETY"),
        TargetRule("drug interaction", "SAFETY"),
        TargetRule("hepatotoxicity", "TOXICITY"),
        TargetRule("nephrotoxicity", "TOXICITY"),
        TargetRule("cardiotoxicity", "TOXICITY"),
    ]
    
    # Add all rules to the matcher
    target_matcher.add(clinical_rules)
    
    print(f"Added {len(clinical_rules)} clinical trial entity rules to medspaCy pipeline")
    
    return nlp

if __name__ == "__main__":
    # Test the enhanced pipeline
    nlp = create_clinical_trial_nlp()
    
    test_text = """
    Patient enrolled in phase 3 randomized controlled trial for type 2 diabetes.
    Baseline HbA1c was 8.2%. Randomized to metformin 500mg twice daily vs placebo.
    Primary endpoint: reduction in HbA1c at 24 weeks.
    Adverse events: mild hypoglycemia reported in 3 patients.
    No serious adverse events related to study drug.
    """
    
    doc = nlp(test_text)
    
    print("Enhanced Entity Extraction Results:")
    for ent in doc.ents:
        context = ""
        if hasattr(ent, '_') and hasattr(ent._, 'is_negated') and ent._.is_negated:
            context += " [NEGATED]"
        if hasattr(ent, '_') and hasattr(ent._, 'is_uncertain') and ent._.is_uncertain:
            context += " [UNCERTAIN]"
        print(f"  • {ent.text} ({ent.label_}){context}")