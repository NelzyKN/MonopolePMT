#ifndef _MonopoleSignatureAnalyzer_h_
#define _MonopoleSignatureAnalyzer_h_

/**
 * \file MonopoleSignatureAnalyzer.h
 * \brief Analyzer for identifying magnetic monopole signatures in FADC traces
 * 
 * Magnetic monopoles would produce distinctive patterns in water-Cherenkov detectors:
 * - Extremely high, sustained signal levels
 * - Smooth traces without muon spikes
 * - Very long signal duration
 */

#include <vector>
#include <cmath>

class MonopoleSignatureAnalyzer {
public:
    struct MonopoleFeatures {
        double sustainedSignalFraction;  // Fraction of trace above high threshold
        double signalSmoothness;         // RMS of derivative (lower = smoother)
        double peakToPlateau;            // Ratio of peak to sustained level
        double totalCharge;              // Total integrated charge [VEM]
        double signalDuration;           // Duration above threshold [ns]
        double riseTime;                 // Time to reach plateau [ns]
        bool isMonopoleCandidate;        // Final classification
        double monopoleScore;            // Confidence score (0-1)
    };
    
    /**
     * \brief Analyze FADC trace for monopole signatures
     * 
     * Monopoles produce:
     * - Sustained high signals (>100 VEM typical)
     * - Smooth traces (no muon spikes)
     * - Long duration (>1000 ns)
     * - Nearly rectangular pulse shape
     */
    static MonopoleFeatures AnalyzeTrace(const std::vector<double>& trace, 
                                         double baseline = 50.0) {
        MonopoleFeatures features;
        const double ADC_PER_VEM = 180.0;
        const double NS_PER_BIN = 25.0;  // 40 MHz sampling
        
        // Thresholds for monopole identification
        const double HIGH_VEM_THRESHOLD = 50.0;   // 50 VEM sustained
        const double VERY_HIGH_VEM_THRESHOLD = 100.0;  // 100 VEM for strong candidates
        
        // Calculate basic properties
        double maxSignal = 0;
        double totalCharge = 0;
        int firstHighBin = -1;
        int lastHighBin = -1;
        int binsAboveHigh = 0;
        int binsAboveVeryHigh = 0;
        
        // Find signal characteristics
        for (size_t i = 0; i < trace.size(); i++) {
            double signal = (trace[i] - baseline) / ADC_PER_VEM;  // Convert to VEM
            
            if (signal > 0) {
                totalCharge += signal;
            }
            
            if (signal > maxSignal) {
                maxSignal = signal;
            }
            
            if (signal > HIGH_VEM_THRESHOLD) {
                if (firstHighBin < 0) firstHighBin = i;
                lastHighBin = i;
                binsAboveHigh++;
            }
            
            if (signal > VERY_HIGH_VEM_THRESHOLD) {
                binsAboveVeryHigh++;
            }
        }
        
        features.totalCharge = totalCharge;
        
        // Calculate signal duration
        if (firstHighBin >= 0 && lastHighBin >= 0) {
            features.signalDuration = (lastHighBin - firstHighBin) * NS_PER_BIN;
        } else {
            features.signalDuration = 0;
        }
        
        // Calculate sustained signal fraction
        features.sustainedSignalFraction = (double)binsAboveHigh / trace.size();
        
        // Calculate signal smoothness (RMS of first derivative)
        double sumSqDiff = 0;
        int count = 0;
        
        for (size_t i = 1; i < trace.size(); i++) {
            double diff = (trace[i] - trace[i-1]) / ADC_PER_VEM;
            // Only calculate for significant signal regions
            if ((trace[i] - baseline) / ADC_PER_VEM > 10.0) {
                sumSqDiff += diff * diff;
                count++;
            }
        }
        
        features.signalSmoothness = (count > 0) ? sqrt(sumSqDiff / count) : 999.0;
        
        // Calculate average plateau level (excluding rise/fall)
        double plateauSum = 0;
        int plateauCount = 0;
        
        if (firstHighBin >= 0 && lastHighBin >= 0) {
            int plateauStart = firstHighBin + (lastHighBin - firstHighBin) / 4;
            int plateauEnd = lastHighBin - (lastHighBin - firstHighBin) / 4;
            
            for (int i = plateauStart; i <= plateauEnd; i++) {
                double signal = (trace[i] - baseline) / ADC_PER_VEM;
                if (signal > HIGH_VEM_THRESHOLD) {
                    plateauSum += signal;
                    plateauCount++;
                }
            }
        }
        
        double plateauLevel = (plateauCount > 0) ? plateauSum / plateauCount : 0;
        features.peakToPlateau = (plateauLevel > 0) ? maxSignal / plateauLevel : 999.0;
        
        // Calculate rise time (10% to 90% of plateau)
        if (firstHighBin >= 0 && plateauLevel > 20.0) {
            double threshold10 = baseline + 0.1 * plateauLevel * ADC_PER_VEM;
            double threshold90 = baseline + 0.9 * plateauLevel * ADC_PER_VEM;
            
            int bin10 = -1, bin90 = -1;
            for (size_t i = 0; i < trace.size(); i++) {
                if (bin10 < 0 && trace[i] > threshold10) bin10 = i;
                if (bin90 < 0 && trace[i] > threshold90) bin90 = i;
                if (bin10 >= 0 && bin90 >= 0) break;
            }
            
            features.riseTime = (bin90 > bin10) ? (bin90 - bin10) * NS_PER_BIN : 0;
        } else {
            features.riseTime = 0;
        }
        
        // Calculate monopole score based on multiple criteria
        double score = 0;
        
        // Criterion 1: Sustained high signal (weight: 0.3)
        if (features.signalDuration > 1000) score += 0.15;  // >1 microsecond
        if (features.signalDuration > 2000) score += 0.15;  // >2 microseconds
        
        // Criterion 2: Very high charge (weight: 0.25)
        if (features.totalCharge > 1000) score += 0.125;   // >1000 VEM total
        if (features.totalCharge > 5000) score += 0.125;   // >5000 VEM total
        
        // Criterion 3: Smooth signal (weight: 0.2)
        if (features.signalSmoothness < 5.0) score += 0.1;
        if (features.signalSmoothness < 2.0) score += 0.1;
        
        // Criterion 4: Rectangular shape (weight: 0.15)
        if (features.peakToPlateau < 1.5) score += 0.15;   // Peak close to plateau
        
        // Criterion 5: Many bins above very high threshold (weight: 0.1)
        if (binsAboveVeryHigh > 40) score += 0.05;  // >1 microsecond above 100 VEM
        if (binsAboveVeryHigh > 80) score += 0.05;  // >2 microseconds above 100 VEM
        
        features.monopoleScore = score;
        
        // Classification threshold
        features.isMonopoleCandidate = (score > 0.6);
        
        // Additional strong monopole criteria
        if (features.totalCharge > 10000 &&          // Extremely high charge
            features.signalDuration > 2000 &&        // Very long duration
            features.signalSmoothness < 2.0 &&       // Very smooth
            features.peakToPlateau < 1.3) {          // Very rectangular
            features.isMonopoleCandidate = true;
            features.monopoleScore = std::min(1.0, score + 0.2);
        }
        
        return features;
    }
    
    /**
     * \brief Check for monopole track pattern across multiple stations
     * 
     * Monopoles would trigger stations in a nearly straight line with
     * similar high signals in each station
     */
    static bool CheckTrackPattern(const std::vector<MonopoleFeatures>& stationFeatures,
                                  const std::vector<double>& stationDistances) {
        // Require at least 3 stations with monopole-like signals
        int candidateStations = 0;
        double avgCharge = 0;
        
        for (const auto& features : stationFeatures) {
            if (features.isMonopoleCandidate) {
                candidateStations++;
                avgCharge += features.totalCharge;
            }
        }
        
        if (candidateStations < 3) return false;
        
        avgCharge /= candidateStations;
        
        // Check for consistent high signals (monopoles don't attenuate like showers)
        double chargeVariation = 0;
        for (const auto& features : stationFeatures) {
            if (features.isMonopoleCandidate) {
                double deviation = fabs(features.totalCharge - avgCharge) / avgCharge;
                chargeVariation += deviation;
            }
        }
        chargeVariation /= candidateStations;
        
        // Monopoles should have relatively consistent signals (<50% variation)
        return (chargeVariation < 0.5);
    }
};

#endif // _MonopoleSignatureAnalyzer_h_
