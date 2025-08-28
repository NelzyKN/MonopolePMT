#!/usr/bin/env python3
"""
Analyze and visualize potential magnetic monopole signatures in Pierre Auger data
Compares monopole candidates with regular cosmic ray showers
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import ROOT
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

def load_trace_from_root(filename, tree_name="TraceTree", event_id=None):
    """Load FADC traces from ROOT file"""
    file = ROOT.TFile(filename)
    tree = file.Get(tree_name)
    
    traces = []
    metadata = []
    
    for entry in tree:
        if event_id is not None and entry.eventId != event_id:
            continue
            
        # Convert ROOT vector to numpy array
        trace_data = np.array([entry.traceData[i] for i in range(entry.traceSize)])
        
        meta = {
            'eventId': entry.eventId,
            'stationId': entry.stationId,
            'energy': entry.energy,
            'vemCharge': entry.vemCharge,
            'primaryType': str(entry.primaryType) if hasattr(entry, 'primaryType') else 'unknown'
        }
        
        traces.append(trace_data)
        metadata.append(meta)
    
    file.Close()
    return traces, metadata

def simulate_monopole_trace(n_bins=2048, baseline=50, monopole_charge=100):
    """
    Simulate what a magnetic monopole trace would look like
    
    Monopole characteristics:
    - Very high sustained signal (100+ VEM)
    - Smooth, nearly rectangular pulse
    - Long duration (>2 microseconds)
    - Minimal fluctuations
    """
    ADC_PER_VEM = 180
    trace = np.ones(n_bins) * baseline
    
    # Monopole enters detector around bin 500
    entry_bin = 500
    # Monopole exits around bin 1600 (>1 microsecond later)
    exit_bin = 1600
    
    # Rise time (~50 ns = 2 bins at 40 MHz)
    rise_bins = 2
    fall_bins = 2
    
    # Create smooth rise
    for i in range(rise_bins):
        trace[entry_bin + i] = baseline + (i+1) * monopole_charge * ADC_PER_VEM / rise_bins
    
    # Sustained high signal with small fluctuations
    plateau_level = baseline + monopole_charge * ADC_PER_VEM
    for i in range(entry_bin + rise_bins, exit_bin - fall_bins):
        # Add small Poisson fluctuations
        fluctuation = np.random.poisson(5) - 5  # Small variations
        trace[i] = plateau_level + fluctuation
    
    # Smooth fall
    for i in range(fall_bins):
        trace[exit_bin - fall_bins + i] = plateau_level * (1 - (i+1)/fall_bins) + baseline
    
    return trace

def simulate_hadronic_trace(n_bins=2048, baseline=50, peak_vem=30):
    """Simulate typical hadronic shower trace with muon spikes"""
    ADC_PER_VEM = 180
    trace = np.ones(n_bins) * baseline
    
    # Main electromagnetic component
    main_start = 800
    main_width = 200
    
    # Create main pulse with exponential decay
    for i in range(main_start, main_start + main_width):
        t = (i - main_start) / main_width
        signal = peak_vem * ADC_PER_VEM * np.exp(-3*t) * (1 - np.exp(-10*t))
        trace[i] = baseline + signal
    
    # Add muon spikes
    n_muons = np.random.poisson(5)
    for _ in range(n_muons):
        muon_time = np.random.randint(main_start - 100, main_start + main_width + 200)
        muon_amplitude = np.random.exponential(5) * ADC_PER_VEM
        muon_width = np.random.randint(2, 5)
        
        for j in range(max(0, muon_time), min(n_bins, muon_time + muon_width)):
            trace[j] += muon_amplitude * np.exp(-(j - muon_time)/2)
    
    # Add noise
    noise = np.random.normal(0, 2, n_bins)
    trace += noise
    
    return trace

def analyze_trace_features(trace, baseline=50):
    """Extract features relevant for monopole identification"""
    ADC_PER_VEM = 180
    signal_vem = (trace - baseline) / ADC_PER_VEM
    signal_vem[signal_vem < 0] = 0
    
    features = {}
    
    # Total charge
    features['total_charge'] = np.sum(signal_vem) * 25e-9  # Convert to VEM*seconds
    
    # Peak value
    features['peak_vem'] = np.max(signal_vem)
    
    # Sustained signal metrics
    high_threshold = 50  # VEM
    very_high_threshold = 100  # VEM
    features['bins_above_50vem'] = np.sum(signal_vem > high_threshold)
    features['bins_above_100vem'] = np.sum(signal_vem > very_high_threshold)
    
    # Signal duration (first to last bin above 50 VEM)
    above_threshold = np.where(signal_vem > high_threshold)[0]
    if len(above_threshold) > 0:
        features['duration_ns'] = (above_threshold[-1] - above_threshold[0]) * 25  # ns
    else:
        features['duration_ns'] = 0
    
    # Smoothness (RMS of derivative for high signal regions)
    high_signal_mask = signal_vem > 10
    if np.any(high_signal_mask):
        derivative = np.diff(signal_vem)
        smoothness_regions = high_signal_mask[:-1]
        if np.any(smoothness_regions):
            features['smoothness'] = np.std(derivative[smoothness_regions])
        else:
            features['smoothness'] = 999
    else:
        features['smoothness'] = 999
    
    # Peak to plateau ratio
    if len(above_threshold) > 10:
        plateau_region = signal_vem[above_threshold[5:-5]]  # Exclude edges
        if len(plateau_region) > 0:
            features['peak_to_plateau'] = features['peak_vem'] / np.mean(plateau_region)
        else:
            features['peak_to_plateau'] = 999
    else:
        features['peak_to_plateau'] = 999
    
    # Calculate monopole score
    score = 0
    if features['duration_ns'] > 1000: score += 0.2
    if features['duration_ns'] > 2000: score += 0.1
    if features['bins_above_100vem'] > 40: score += 0.2
    if features['total_charge'] > 5000e-9: score += 0.2
    if features['smoothness'] < 5: score += 0.15
    if features['peak_to_plateau'] < 1.5: score += 0.15
    
    features['monopole_score'] = score
    features['is_monopole_candidate'] = score > 0.6
    
    return features

def plot_monopole_comparison():
    """Create comparison plot of monopole vs hadronic traces"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # Time axis (40 MHz sampling)
    time_bins = np.arange(2048) * 25  # nanoseconds
    
    # Generate example traces
    monopole_trace = simulate_monopole_trace(monopole_charge=120)
    hadronic_trace = simulate_hadronic_trace(peak_vem=40)
    
    # Plot 1: Monopole trace
    ax = axes[0, 0]
    ax.plot(time_bins, monopole_trace, 'b-', linewidth=0.8)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax.axhline(y=50 + 100*180, color='r', linestyle='--', alpha=0.5, label='100 VEM')
    ax.set_title('Simulated Magnetic Monopole Signal', fontweight='bold')
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('ADC Counts')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add monopole signature region
    rect = Rectangle((500*25, 50), (1600-500)*25, 120*180, 
                    alpha=0.2, facecolor='blue', edgecolor='blue', linewidth=2)
    ax.add_patch(rect)
    ax.text(1050*25, 50 + 130*180, 'Monopole Transit', 
           fontsize=10, ha='center', color='blue', fontweight='bold')
    
    # Plot 2: Hadronic shower trace
    ax = axes[0, 1]
    ax.plot(time_bins, hadronic_trace, 'g-', linewidth=0.8)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Baseline')
    ax.set_title('Typical Hadronic Shower Signal', fontweight='bold')
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('ADC Counts')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Annotate muon spikes
    peaks, _ = signal.find_peaks(hadronic_trace, height=50+10*180, distance=20)
    for peak in peaks[:3]:  # Annotate first 3 peaks
        ax.annotate('Muon', xy=(peak*25, hadronic_trace[peak]), 
                   xytext=(peak*25, hadronic_trace[peak]+500),
                   arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                   fontsize=8, color='red')
    
    # Plot 3: Signal in VEM units (Monopole)
    ax = axes[1, 0]
    monopole_vem = (monopole_trace - 50) / 180
    monopole_vem[monopole_vem < 0] = 0
    ax.fill_between(time_bins, 0, monopole_vem, alpha=0.5, color='blue')
    ax.set_title('Monopole Signal in VEM Units')
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Signal [VEM]')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Signal in VEM units (Hadronic)
    ax = axes[1, 1]
    hadronic_vem = (hadronic_trace - 50) / 180
    hadronic_vem[hadronic_vem < 0] = 0
    ax.fill_between(time_bins, 0, hadronic_vem, alpha=0.5, color='green')
    ax.set_title('Hadronic Signal in VEM Units')
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Signal [VEM]')
    ax.grid(True, alpha=0.3)
    
    # Extract and compare features
    monopole_features = analyze_trace_features(monopole_trace)
    hadronic_features = analyze_trace_features(hadronic_trace)
    
    # Plot 5: Feature comparison bar chart
    ax = axes[2, 0]
    features_to_compare = ['total_charge', 'peak_vem', 'duration_ns', 'smoothness']
    feature_labels = ['Total Charge\n[VEM·s]', 'Peak\n[VEM]', 'Duration\n[ns]', 'Smoothness']
    
    x = np.arange(len(features_to_compare))
    width = 0.35
    
    monopole_values = [monopole_features[f] * (1e9 if f == 'total_charge' else 1) 
                      for f in features_to_compare]
    hadronic_values = [hadronic_features[f] * (1e9 if f == 'total_charge' else 1) 
                      for f in features_to_compare]
    
    bars1 = ax.bar(x - width/2, monopole_values, width, label='Monopole', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, hadronic_values, width, label='Hadronic', color='green', alpha=0.7)
    
    ax.set_ylabel('Value')
    ax.set_title('Feature Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels)
    ax.legend()
    ax.set_yscale('log')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1e}' if height > 100 else f'{height:.1f}',
               ha='center', va='bottom', fontsize=8)
    
    # Plot 6: Monopole Score
    ax = axes[2, 1]
    scores = [monopole_features['monopole_score'], hadronic_features['monopole_score']]
    colors = ['blue', 'green']
    labels = ['Monopole', 'Hadronic']
    
    bars = ax.bar(labels, scores, color=colors, alpha=0.7)
    ax.axhline(y=0.6, color='red', linestyle='--', label='Detection Threshold')
    ax.set_ylabel('Monopole Score')
    ax.set_title('Monopole Detection Score')
    ax.set_ylim(0, 1)
    ax.legend()
    
    # Add score values on bars
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2., score + 0.02,
               f'{score:.2f}', ha='center', fontsize=10, fontweight='bold')
    
    # Add candidate status
    for i, (bar, score) in enumerate(zip(bars, scores)):
        status = 'CANDIDATE' if score > 0.6 else 'NOT CANDIDATE'
        color = 'darkgreen' if score > 0.6 else 'darkred'
        ax.text(bar.get_x() + bar.get_width()/2., 0.05,
               status, ha='center', fontsize=9, color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.suptitle('Magnetic Monopole vs Hadronic Shower Signatures in Pierre Auger SD', 
                fontsize=14, fontweight='bold', y=1.02)
    
    return fig, monopole_features, hadronic_features

# Main analysis
if __name__ == "__main__":
    print("Pierre Auger Magnetic Monopole Signal Analysis")
    print("=" * 50)
    
    # Create comparison plots
    fig, monopole_feat, hadronic_feat = plot_monopole_comparison()
    
    print("\nMagnetic Monopole Features:")
    print("-" * 30)
    for key, value in monopole_feat.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.3f}")
        else:
            print(f"  {key:20s}: {value}")
    
    print("\nHadronic Shower Features:")
    print("-" * 30)
    for key, value in hadronic_feat.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.3f}")
        else:
            print(f"  {key:20s}: {value}")
    
    print("\nDetection Criteria for Magnetic Monopoles:")
    print("-" * 40)
    print("1. Sustained high signal (>100 VEM) for >1 microsecond")
    print("2. Very smooth trace (RMS derivative <5 VEM)")
    print("3. Total charge >5000 VEM·ns")
    print("4. Nearly rectangular pulse shape (peak/plateau <1.5)")
    print("5. Multiple stations with similar signals in straight line")
    
    plt.savefig('monopole_signature_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nAnalysis complete. Figure saved as 'monopole_signature_analysis.png'")
