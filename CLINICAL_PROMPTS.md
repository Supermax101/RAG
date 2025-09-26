# Clinical Prompt Engineering Guide

## TPN Clinical Assistant Response Format

### Expected Response Style

**✅ GOOD - Concise Clinical Response:**
```
**Normal Serum Potassium for Neonates on TPN:**

• **Range:** 3.5-5.0 mEq/L (same as adults)
• **Critical values:** <3.0 or >6.0 mEq/L  
• **Monitoring frequency:** Daily during TPN initiation, then 2-3x weekly
• **Dosing adjustment:** 1-2 mEq/kg/day typically required in TPN

**Clinical Actions:**
• Hold TPN if K+ <2.5 or >6.5 mEq/L
• Recheck levels 4-6 hours after adjustment
• Consider cardiac monitoring if severe abnormalities
```

**❌ POOR - Academic/Verbose Response:**
```
Based on the comprehensive analysis of the provided documentation, we can observe that the literature presents varying perspectives on the optimal serum potassium levels for neonatal patients receiving total parenteral nutrition. The documentation indicates several considerations... [continues for paragraphs]
```

### Clinical Question Categories

#### 1. **Dosing Questions**
- Format: "Drug/nutrient X mg/kg/day for Y population"
- Include: Starting dose, maximum dose, adjustment criteria
- Example: "TPN protein: 2.5-4 g/kg/day for preterm <1500g"

#### 2. **Monitoring Questions** 
- Format: "Parameter: Normal range, frequency, actions"
- Include: Values, when to check, what to do
- Example: "Triglycerides: <250 mg/dL, check 2x weekly, hold lipids if >400"

#### 3. **Clinical Decision Questions**
- Format: "Indication → Action" 
- Include: Clear criteria and next steps
- Example: "Start TPN if: NPO >72h in term infant, >24h in preterm <1500g"

#### 4. **Complications Questions**
- Format: "Signs → Assessment → Management"
- Include: Recognition, evaluation, treatment
- Example: "TPN liver disease: Direct bili >2mg/dL → Reduce/cycle TPN, consider Omegaven"

### Key Response Requirements

1. **Be Direct**: Start with the answer, not background
2. **Use Bullets**: Easy to scan during clinical work
3. **Include Numbers**: Specific values, doses, frequencies
4. **Action-Oriented**: What to do, not just what to know
5. **Evidence-Based**: Only information from provided sources
6. **Clinical Context**: Specify patient population when relevant

### Prompt Engineering Tips

- **Temperature**: 0.05 (very low for precision)
- **Max tokens**: 1500 (forces conciseness)
- **Stop tokens**: Prevent rambling
- **Context focus**: Prioritize actionable clinical information
- **Format consistency**: Use bullet points and clinical sections
