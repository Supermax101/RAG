#!/usr/bin/env python3
"""
Quick test to verify plotting dependencies work correctly
"""

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for compatibility
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    
    print("✅ All plotting libraries imported successfully!")
    
    # Create a simple test chart
    data = pd.DataFrame({
        'Model': ['mistral:7b', 'llama3:8b', 'phi3:3.8b'],
        'BLEU': [0.45, 0.42, 0.38],
        'Speed': [3.4, 4.1, 1.2]
    })
    
    plt.figure(figsize=(8, 6))
    plt.bar(data['Model'], data['BLEU'], color='skyblue')
    plt.title('Test Chart - BLEU Scores')
    plt.ylabel('BLEU Score')
    plt.tight_layout()
    plt.savefig('test_chart.png', dpi=150)
    plt.close()
    
    print("✅ Test chart created successfully: test_chart.png")
    print("📊 Plotting system is ready for model evaluation!")
    
except ImportError as e:
    print(f"❌ Missing plotting dependency: {e}")
    print("Install with: pip install matplotlib seaborn pandas")
except Exception as e:
    print(f"❌ Plotting test failed: {e}")
    print("This might affect chart generation in model comparison.")


if __name__ == "__main__":
    pass
