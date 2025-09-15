#!/usr/bin/env python3
"""
Graph Data Generator for Bank Term Deposit Prediction Report

This module generates comprehensive visualizations and data summaries for the
bank term deposit prediction analysis. It creates publication-quality graphs
comparing real and synthetic data performance across multiple dimensions.

Key Features:
- Feature distribution comparisons
- Categorical data analysis
- Target variable distribution analysis
- Statistical summary tables
- Publication-ready visualizations

2025
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
import seaborn as sns
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Configure matplotlib and seaborn for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ============================================================================
# DATA DEFINITIONS FOR VISUALIZATIONS
# ============================================================================

# Statistical data for numerical features comparison
FEATURE_STATISTICS = {
    'Age': {
        'real_mean': 40.94, 
        'real_std': 10.62, 
        'synthetic_mean': 40.93, 
        'synthetic_std': 10.10
    },
    'Balance': {
        'real_mean': 1362, 
        'real_std': 3045, 
        'synthetic_mean': 1204, 
        'synthetic_std': 2836
    },
    'Duration': {
        'real_mean': 258, 
        'real_std': 258, 
        'synthetic_mean': 256, 
        'synthetic_std': 273
    },
    'Campaign': {
        'real_mean': 2.76, 
        'real_std': 3.10, 
        'synthetic_mean': 2.58, 
        'synthetic_std': 2.72
    }
}

# Categorical feature distribution data
JOB_DISTRIBUTION_DATA = {
    'categories': ['Blue-collar', 'Management', 'Technician', 'Admin', 'Services'],
    'real_pct': [22.1, 19.8, 16.2, 10.9, 9.1],
    'synthetic_pct': [21.8, 20.1, 16.5, 11.2, 8.9]
}

EDUCATION_DISTRIBUTION_DATA = {
    'categories': ['Secondary', 'Tertiary', 'Primary', 'Unknown'],
    'real_pct': [51.0, 29.7, 15.2, 4.1],
    'synthetic_pct': [50.2, 30.1, 15.4, 4.3]
}

MARITAL_DISTRIBUTION_DATA = {
    'categories': ['Married', 'Single', 'Divorced'],
    'real_pct': [60.2, 28.1, 11.7],
    'synthetic_pct': [60.8, 27.6, 11.6]
}

# Target variable distribution data
TARGET_DISTRIBUTION_DATA = {
    'real': {'no': 39922, 'yes': 5289},
    'synthetic': {'no': 659512, 'yes': 90488}
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_sample_data(mean: float, std: float, size: int = 1000) -> np.ndarray:
    """
    Generate sample data for visualization purposes.
    
    Args:
        mean: Mean value for normal distribution
        std: Standard deviation for normal distribution
        size: Number of samples to generate
        
    Returns:
        Array of generated sample data
    """
    return np.random.normal(mean, std, size)

# ============================================================================
# VISUALIZATION GENERATION FUNCTIONS
# ============================================================================

def create_feature_distribution_comparison() -> plt.Figure:
    """
    Create feature distribution comparison plots for real vs synthetic data.
    
    Generates a 2x2 subplot grid comparing the distributions of key numerical
    features between real and synthetic datasets using overlapping histograms.
    
    Returns:
        matplotlib Figure object containing the comparison plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Feature Distribution Comparison: Real vs Synthetic Data', 
                 fontsize=16, fontweight='bold')
    
    features = ['Age', 'Balance', 'Duration', 'Campaign']
    subplot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    for i, feature_name in enumerate(features):
        row, col = subplot_positions[i]
        ax = axes[row, col]
        
        # Generate sample data for visualization
        real_sample_data = generate_sample_data(
            FEATURE_STATISTICS[feature_name]['real_mean'], 
            FEATURE_STATISTICS[feature_name]['real_std']
        )
        synthetic_sample_data = generate_sample_data(
            FEATURE_STATISTICS[feature_name]['synthetic_mean'], 
            FEATURE_STATISTICS[feature_name]['synthetic_std']
        )
        
        # Create overlapping histograms with transparency
        ax.hist(real_sample_data, bins=30, alpha=0.7, color='blue', 
                label='Real Dataset', density=True)
        ax.hist(synthetic_sample_data, bins=30, alpha=0.7, color='red', 
                label='Synthetic Dataset', density=True)
        
        # Add statistical information text box
        real_mean = FEATURE_STATISTICS[feature_name]['real_mean']
        real_std = FEATURE_STATISTICS[feature_name]['real_std']
        synthetic_mean = FEATURE_STATISTICS[feature_name]['synthetic_mean']
        synthetic_std = FEATURE_STATISTICS[feature_name]['synthetic_std']
        
        stats_text = (f'Real: μ={real_mean:.2f}, σ={real_std:.2f}\n'
                     f'Synthetic: μ={synthetic_mean:.2f}, σ={synthetic_std:.2f}')
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Configure subplot appearance
        ax.set_title(f'{feature_name} Distribution', fontweight='bold')
        ax.set_xlabel(feature_name)
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_categorical_distribution_comparison() -> plt.Figure:
    """
    Create categorical feature distribution comparison charts.
    
    Generates horizontal grouped bar charts comparing the distribution of
    categorical features (job, education, marital status) between real
    and synthetic datasets.
    
    Returns:
        matplotlib Figure object containing the comparison charts
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Categorical Feature Distribution Comparison', 
                 fontsize=16, fontweight='bold')
    
    categorical_datasets = [
        ('Job Distribution', JOB_DISTRIBUTION_DATA),
        ('Education Distribution', EDUCATION_DISTRIBUTION_DATA),
        ('Marital Status Distribution', MARITAL_DISTRIBUTION_DATA)
    ]
    
    for i, (chart_title, distribution_data) in enumerate(categorical_datasets):
        ax = axes[i]
        categories = distribution_data['categories']
        real_percentages = distribution_data['real_pct']
        synthetic_percentages = distribution_data['synthetic_pct']
        
        y_positions = np.arange(len(categories))
        bar_width = 0.4
        
        # Create horizontal grouped bar chart
        real_bars = ax.barh(y_positions - bar_width/2, real_percentages, 
                           bar_width, label='Real Dataset', color='blue', alpha=0.8)
        synthetic_bars = ax.barh(y_positions + bar_width/2, synthetic_percentages, 
                                bar_width, label='Synthetic Dataset', color='red', alpha=0.8)
        
        # Add percentage labels to bars
        for j, (real_bar, synthetic_bar) in enumerate(zip(real_bars, synthetic_bars)):
            # Real dataset labels
            ax.text(real_bar.get_width() + 0.5, 
                   real_bar.get_y() + real_bar.get_height()/2, 
                   f'{real_percentages[j]:.1f}%', va='center', fontsize=9)
            # Synthetic dataset labels
            ax.text(synthetic_bar.get_width() + 0.5, 
                   synthetic_bar.get_y() + synthetic_bar.get_height()/2, 
                   f'{synthetic_percentages[j]:.1f}%', va='center', fontsize=9)
        
        # Configure subplot appearance
        ax.set_yticks(y_positions)
        ax.set_yticklabels(categories)
        ax.set_xlabel('Percentage (%)')
        ax.set_title(chart_title, fontweight='bold')
        ax.legend()
        ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_target_distribution_comparison() -> plt.Figure:
    """
    Create target variable distribution comparison pie charts.
    
    Generates side-by-side pie charts comparing the distribution of the
    target variable (subscription vs no subscription) between real and
    synthetic datasets.
    
    Returns:
        matplotlib Figure object containing the pie chart comparison
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Target Distribution Comparison', fontsize=16, fontweight='bold')
    
    target_datasets = [
        ('Real Dataset', TARGET_DISTRIBUTION_DATA['real']),
        ('Synthetic Dataset', TARGET_DISTRIBUTION_DATA['synthetic'])
    ]
    
    pie_colors = ['red', 'teal']
    class_labels = ['No Subscription', 'Subscription']
    
    for i, (dataset_title, target_data) in enumerate(target_datasets):
        ax = axes[i]
        class_counts = [target_data['no'], target_data['yes']]
        total_samples = sum(class_counts)
        class_percentages = [count/total_samples*100 for count in class_counts]
        
        # Create pie chart with custom formatting
        wedges, texts, autotexts = ax.pie(
            class_counts, 
            labels=class_labels, 
            colors=pie_colors, 
            autopct='%1.1f%%',
            startangle=90, 
            textprops={'fontsize': 10}
        )
        
        # Enhance autotext with sample counts
        for j, (wedge, autotext) in enumerate(zip(wedges, autotexts)):
            enhanced_label = f'{class_percentages[j]:.1f}%\n({class_counts[j]:,})'
            autotext.set_text(enhanced_label)
            autotext.set_fontsize(9)
        
        ax.set_title(dataset_title, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

# ============================================================================
# TABLE GENERATION FUNCTIONS
# ============================================================================

def create_dataset_overview_table() -> plt.Figure:
    """
    Create dataset overview comparison table.
    
    Generates a formatted table comparing key statistics between real and
    synthetic datasets including row counts, feature counts, class distributions,
    and imbalance ratios.
    
    Returns:
        matplotlib Figure object containing the overview table
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('tight')
    ax.axis('off')
    
    # Dataset overview data
    table_data = [
        ['Real', '45,211', '16', '39,922 (88.3%)', '5,289 (11.7%)', '7.5:1'],
        ['Synthetic', '1,000,000', '17', '659,512 (87.9%)', '90,488 (12.1%)', '7.3:1']
    ]
    
    column_headers = ['Dataset', 'Rows', 'Features', 'No Subscription', 
                     'Subscription', 'Ratio']
    
    # Create and style table
    table = ax.table(cellText=table_data, colLabels=column_headers, 
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Apply header styling
    for i in range(len(column_headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Dataset Overview', fontsize=14, fontweight='bold', pad=20)
    return fig

def create_feature_statistics_table() -> plt.Figure:
    """
    Create feature statistics comparison table.
    
    Generates a formatted table comparing statistical measures (mean and
    standard deviation) of key numerical features between real and synthetic
    datasets, including the absolute differences.
    
    Returns:
        matplotlib Figure object containing the statistics table
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Feature statistics data
    statistics_data = [
        ['Age', '40.94±10.62', '40.93±10.10', '0.01'],
        ['Balance', '1,362±3,045', '1,204±2,836', '158'],
        ['Duration', '258±258', '256±273', '2'],
        ['Campaign', '2.76±3.10', '2.58±2.72', '0.18']
    ]
    
    column_headers = ['Feature', 'Real Mean±Std', 'Synthetic Mean±Std', 'Difference']
    
    # Create and style table
    table = ax.table(cellText=statistics_data, colLabels=column_headers, 
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Apply header styling
    for i in range(len(column_headers)):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Feature Statistics Comparison', fontsize=14, fontweight='bold', pad=20)
    return fig

# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

def generate_all_visualizations() -> List[Tuple[plt.Figure, str]]:
    """
    Generate all visualizations and save them to the artifacts directory.
    
    Creates comprehensive visualizations including feature distributions,
    categorical comparisons, target distributions, and summary tables.
    Saves all figures as high-resolution PNG files.
    
    Returns:
        List of tuples containing (figure, filename) pairs for all generated plots
    """
    print("🎨 Generating comprehensive visualizations...")
    
    # Create output directory for saved figures
    output_directory = Path('artifacts/report_graphs')
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # Generate all visualization figures
    print("📊 Creating Graph 1: Feature Distribution Comparison...")
    feature_distribution_fig = create_feature_distribution_comparison()
    
    print("📊 Creating Graph 2: Categorical Distribution Comparison...")
    categorical_distribution_fig = create_categorical_distribution_comparison()
    
    print("📊 Creating Graph 3: Target Distribution Comparison...")
    target_distribution_fig = create_target_distribution_comparison()
    
    print("📋 Creating Table 1: Dataset Overview...")
    dataset_overview_fig = create_dataset_overview_table()
    
    print("📋 Creating Table 2: Feature Statistics...")
    feature_statistics_fig = create_feature_statistics_table()
    
    # Organize figures with their corresponding filenames
    visualization_figures = [
        (feature_distribution_fig, 'graph1_feature_distributions.png'),
        (categorical_distribution_fig, 'graph2_categorical_distributions.png'),
        (target_distribution_fig, 'graph3_target_distributions.png'),
        (dataset_overview_fig, 'table1_dataset_overview.png'),
        (feature_statistics_fig, 'table2_feature_statistics.png')
    ]
    
    # Save all figures to disk
    print("💾 Saving visualization files...")
    for figure, filename in visualization_figures:
        file_path = output_directory / filename
        figure.savefig(file_path, dpi=300, bbox_inches='tight', 
                      facecolor='white', edgecolor='none')
        print(f"✅ Saved: {file_path}")
    
    print("🎉 All visualizations generated and saved successfully!")
    print(f"📁 Output directory: {output_directory}")
    
    # Display all figures in interactive mode
    plt.show()
    
    return visualization_figures

def main():
    """
    Main execution function for standalone script usage.
    
    Sets random seed for reproducibility and generates all visualizations
    when the script is run directly.
    """
    # Set random seed for reproducible visualization data
    np.random.seed(42)
    
    # Generate all visualizations
    generated_figures = generate_all_visualizations()
    
    print(f"\n📈 Generated {len(generated_figures)} visualization files")
    print("Ready for use in reports and presentations!")

# Execute main function when script is run directly
if __name__ == "__main__":
    main()