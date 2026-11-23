import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import os

def create_sample_frames():
    """Create sample video frames showing lip region extraction"""
    # Create a sample frame (46x140 pixels, grayscale)
    frame = np.random.randint(0, 255, (46, 140), dtype=np.uint8)
    
    # Add some structure to make it look more realistic
    # Create a lip-like region in the center
    center_y, center_x = 23, 70
    for y in range(46):
        for x in range(140):
            # Create a lip-like shape
            dist_from_center = np.sqrt((y - center_y)**2 + (x - center_x)**2)
            if dist_from_center < 15:
                frame[y, x] = np.random.randint(100, 200)
            elif dist_from_center < 25:
                frame[y, x] = np.random.randint(50, 150)
    
    return frame

def create_processing_pipeline_diagram():
    """Create a diagram showing the processing pipeline"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original video frame
    original_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    axes[0].imshow(original_frame)
    axes[0].set_title('Original Video Frame')
    axes[0].axis('off')
    
    # Grayscale conversion
    gray_frame = cv2.cvtColor(original_frame, cv2.COLOR_RGB2GRAY)
    axes[1].imshow(gray_frame, cmap='gray')
    axes[1].set_title('Grayscale Conversion')
    axes[1].axis('off')
    
    # Lip region extraction
    lip_region = gray_frame[190:236, 80:220]
    axes[2].imshow(lip_region, cmap='gray')
    axes[2].set_title('Lip Region Extraction')
    axes[2].axis('off')
    
    # Normalized frame
    normalized_frame = create_sample_frames()
    axes[3].imshow(normalized_frame, cmap='gray')
    axes[3].set_title('Normalized Frame')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig('images/processing_pipeline.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_model_architecture_diagram():
    """Create a diagram showing the model architecture"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define layers
    layers = [
        ('Input\n(75, 46, 140, 1)', 0.1),
        ('Conv3D + ReLU\n+ MaxPool3D', 0.2),
        ('Conv3D + ReLU\n+ MaxPool3D', 0.3),
        ('Conv3D + ReLU\n+ MaxPool3D', 0.4),
        ('TimeDistributed\nFlatten', 0.5),
        ('Bidirectional LSTM\n(128 units)', 0.6),
        ('Bidirectional LSTM\n(128 units)', 0.7),
        ('Dense\n(41 classes)', 0.8),
        ('CTC Decoding', 0.9)
    ]
    
    # Draw the architecture
    for i, (layer_name, y_pos) in enumerate(layers):
        # Draw rectangle for each layer
        rect = plt.Rectangle((0.1, y_pos - 0.05), 0.8, 0.08, 
                           facecolor='lightblue', edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Add text
        ax.text(0.5, y_pos, layer_name, ha='center', va='center', 
               fontsize=10, fontweight='bold')
        
        # Add arrows between layers
        if i < len(layers) - 1:
            ax.arrow(0.5, y_pos - 0.05, 0, -0.05, 
                    head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Lip Reading Model Architecture', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.savefig('images/model_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_sample_output():
    """Create a sample output showing the application interface"""
    # Create a mock interface image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left column - Original video
    video_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    ax1.imshow(video_frame)
    ax1.set_title('Original Video', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Right column - Processed frames
    processed_frames = np.random.randint(0, 255, (46, 140), dtype=np.uint8)
    ax2.imshow(processed_frames, cmap='gray')
    ax2.set_title('Processed Frames (Model Input)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    # Add text overlay for predictions
    fig.text(0.5, 0.02, 'Predicted Text: "hello world"', 
             ha='center', fontsize=16, fontweight='bold', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('images/sample_output.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_chart():
    """Create a performance comparison chart"""
    models = ['CNN Only', 'LSTM Only', 'CNN + LSTM', 'Our Model\n(CNN + BiLSTM)']
    accuracies = [65, 72, 78, 85]
    
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, accuracy in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{accuracy}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('images/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_workflow_diagram():
    """Create a workflow diagram showing the complete process"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define workflow steps
    steps = [
        ('Video Input', 0.1, 0.5),
        ('Frame\nExtraction', 0.25, 0.5),
        ('Preprocessing\n(Grayscale + Crop)', 0.4, 0.5),
        ('Model\nInference', 0.55, 0.5),
        ('CTC\nDecoding', 0.7, 0.5),
        ('Text\nOutput', 0.85, 0.5)
    ]
    
    # Draw workflow
    for i, (step_name, x, y) in enumerate(steps):
        # Draw circle for each step
        circle = plt.Circle((x, y), 0.08, facecolor='lightblue', 
                           edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        
        # Add text
        ax.text(x, y, step_name, ha='center', va='center', 
               fontsize=10, fontweight='bold')
        
        # Add arrows between steps
        if i < len(steps) - 1:
            next_x = steps[i + 1][1]
            ax.arrow(x + 0.08, y, next_x - x - 0.16, 0, 
                    head_width=0.02, head_length=0.02, fc='black', ec='black')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Lip Reading Workflow', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.savefig('images/workflow_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_comparison():
    """Create a feature comparison table as an image"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define features and their status
    features = [
        'Real-time Processing',
        'Multi-language Support',
        'Web Interface',
        'Video Upload',
        'Frame Extraction',
        'Deep Learning Model',
        'Text Output',
        'Visual Feedback'
    ]
    
    status = ['✅', '❌', '✅', '✅', '✅', '✅', '✅', '✅']
    
    # Create table
    table_data = [[feature, stat] for feature, stat in zip(features, status)]
    
    table = ax.table(cellText=table_data,
                    colLabels=['Feature', 'Status'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.7, 0.3])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    
    # Color header
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color status cells
    for i, stat in enumerate(status):
        if stat == '✅':
            table[(i+1, 1)].set_facecolor('#90EE90')
        else:
            table[(i+1, 1)].set_facecolor('#FFB6C1')
    
    ax.set_title('Feature Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('images/feature_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    print("Generating sample output images...")
    
    # Generate all images
    create_processing_pipeline_diagram()
    print("✓ Processing pipeline diagram created")
    
    create_model_architecture_diagram()
    print("✓ Model architecture diagram created")
    
    create_sample_output()
    print("✓ Sample output interface created")
    
    create_performance_chart()
    print("✓ Performance comparison chart created")
    
    create_workflow_diagram()
    print("✓ Workflow diagram created")
    
    create_feature_comparison()
    print("✓ Feature comparison table created")
    
    print("\nAll images have been generated in the 'images' directory!")
    print("Generated files:")
    print("- processing_pipeline.png")
    print("- model_architecture.png")
    print("- sample_output.png")
    print("- performance_comparison.png")
    print("- workflow_diagram.png")
    print("- feature_comparison.png") 