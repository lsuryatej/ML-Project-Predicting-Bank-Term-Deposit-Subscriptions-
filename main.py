#!/usr/bin/env python3
"""
Bank Term Deposit Prediction: Synthetic vs Real Data Analysis

Main entry point for the ML pipeline demonstrating synthetic data effectiveness
in financial machine learning applications.

This module provides an interactive interface for running experiments and
generating visualizations for the bank term deposit prediction project.
"""

import sys
import os
from pathlib import Path

# Add source directory to Python path
sys.path.append('src')

def main():
    """
    Main entry point with interactive menu options.
    
    Provides user interface for:
    - Running data analysis experiments
    - Generating visualization data
    - Displaying project structure
    """
    print("🏦 BANK TERM DEPOSIT PREDICTION: SYNTHETIC VS REAL DATA")
    print("=" * 60)
    print("Production-ready ML pipeline with comprehensive validation")
    print()
    
    print("Available options:")
    print("1. Run Enhanced Demo (Advanced Features)")
    print("2. Generate Graph Data for Report")
    print("3. Show Project Structure")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                print("\n🎯 Starting Enhanced Demo with Advanced Features...")
                _run_enhanced_demo()
                break
                
            elif choice == '2':
                print("\n📊 Generating graph data for report...")
                _run_graph_generation()
                break

            elif choice == '3':
                _show_project_structure()

            elif choice == '4':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-4.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

    # Removed demo_walkthrough references

def _run_enhanced_demo():
    """
    Execute the enhanced demo with advanced features.
    
    Runs the comprehensive enhanced demo that includes:
    - Optimized model configurations
    - Automatic threshold optimization
    - Advanced class imbalance handling
    - Data efficiency analysis
    - Business impact assessment
    """
    try:
        import enhanced_demo
        enhanced_demo.main()  
        print("✅ Enhanced demo completed successfully!")
    except ImportError as e:
        print(f"❌ Failed to import enhanced demo: {e}")
    except Exception as e:
        print(f"❌ Error during enhanced demo: {e}")

def _run_graph_generation():
    """
    Execute graph data generation for report visualizations.
    
    Imports and runs the graph data generator to create all necessary
    visualization data for the project report.
    """
    try:
        import graph_data_for_report
        print("✅ Graph data generation completed successfully!")
    except ImportError as e:
        print(f"❌ Failed to import graph generator: {e}")
    except Exception as e:
        print(f"❌ Error during graph generation: {e}")

def _show_project_structure():
    """
    Display the clean project structure and key features.
    
    Shows the organized directory layout and highlights the main
    components of the machine learning pipeline.
    """
    print("\n📁 PROJECT STRUCTURE:")
    print("=" * 40)
    
    structure = """
    bank-term-deposit-prediction/
    ├── main.py                              # Main entry point
    ├── enhanced_demo.py                     # Enhanced demo with advanced features
    ├── graph_data_for_report.py             # Visualization generator
    ├── config.yaml                          # Configuration settings
    ├── requirements.txt                     # Python dependencies
    ├── README.md                            # Project documentation
    ├── report.md                            # Comprehensive analysis report
    ├── GRAPH_DATA_SUMMARY.md                # Visualization data summary
    ├── LICENSE                              # MIT license
    ├── 
    ├── src/                                 # Core source code
    │   ├── __init__.py                      # Package initialization
    │   ├── data_utils.py                    # Data loading utilities
    │   ├── enhanced_preprocessing.py        # Advanced preprocessing
    │   └── enhanced_transfer_learning.py    # Transfer learning methods
    │
    ├── data/raw/                            # Dataset directory
    │   ├── real.csv                         # Real bank dataset
    │   ├── synth_train.csv                  # Synthetic training data
    │   └── synth_test.csv                   # Synthetic test data
    │
    └── artifacts/                           # Generated outputs
        ├── models/                          # Trained model files
        ├── results/                         # Experiment results
        └── plots/                           # Generated visualizations
    """
    
    print(structure)
    
    print("\n🎯 KEY FEATURES:")
    print("✅ Interactive and Enhanced Demos")
    print("✅ Automatic threshold optimization")
    print("✅ Data efficiency analysis")
    print("✅ Business impact assessment")
    print("✅ Clean, modular architecture")
    print("✅ Comprehensive validation methodology")
    print("✅ Production-ready implementation")
    print("✅ Extensive documentation")
    print("✅ Reproducible experiments")

if __name__ == "__main__":
    main()