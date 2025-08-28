/**
 * MonopoleTriggerModule.cc
 * Module for identifying magnetic monopole candidates in Surface Detector data
 * 
 * This module analyzes FADC traces looking for signatures consistent with
 * magnetic monopoles: extremely high sustained signals with smooth profiles
 */

#include "MonopoleTriggerModule.h"
#include "MonopoleSignatureAnalyzer.h"
#include <fwk/CentralConfig.h>
#include <utl/ErrorLogger.h>
#include <evt/Event.h>
#include <sevt/SEvent.h>
#include <sevt/Station.h>
#include <sevt/PMT.h>
#include <det/Detector.h>
#include <sdet/SDetector.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TH2D.h>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace fwk;
using namespace evt;
using namespace sevt;

// Constructor
MonopoleTriggerModule::MonopoleTriggerModule() :
    fEventCount(0),
    fMonopoleCandidates(0),
    fStationsAnalyzed(0),
    fOutputFile(nullptr),
    fMonopoleTree(nullptr)
{
}

// Destructor
MonopoleTriggerModule::~MonopoleTriggerModule()
{
}

// Init method
VModule::ResultFlag MonopoleTriggerModule::Init()
{
    INFO("MonopoleTriggerModule::Init() - Initializing magnetic monopole trigger");
    
    INFO("=== Magnetic Monopole Detection Configuration ===");
    INFO("Expected signatures:");
    INFO("  - Sustained signals >50-100 VEM");
    INFO("  - Signal duration >1-2 microseconds");
    INFO("  - Smooth traces (no muon spikes)");
    INFO("  - Multiple stations in straight track");
    INFO("  - Total charge >1000-10000 VEM");
    INFO("================================================");
    
    // Create output file
    fOutputFile = new TFile("monopole_candidates.root", "RECREATE");
    if (!fOutputFile || fOutputFile->IsZombie()) {
        ERROR("Failed to create monopole output file");
        return eFailure;
    }
    
    // Create histograms
    hMonopoleScore = new TH1D("hMonopoleScore", 
        "Monopole Score Distribution;Score;Entries", 100, 0, 1);
    hSustainedSignal = new TH1D("hSustainedSignal", 
        "Sustained Signal Duration;Duration [ns];Entries", 100, 0, 5000);
    hTotalChargeMonopole = new TH1D("hTotalChargeMonopole", 
        "Total Charge for Monopole Candidates;Charge [VEM];Entries", 100, 0, 20000);
    hSignalSmoothness = new TH1D("hSignalSmoothness", 
        "Signal Smoothness (RMS derivative);RMS [VEM];Entries", 100, 0, 20);
    hChargeVsSmoothnessMap = new TH2D("hChargeVsSmoothnessMap",
        "Charge vs Smoothness;Total Charge [VEM];Smoothness", 
        100, 0, 20000, 100, 0, 20);
    hStationsPerEvent = new TH1D("hStationsPerEvent",
        "Stations with Monopole-like Signal per Event;N Stations;Events", 20, 0, 20);
    
    // Create analysis tree
    fMonopoleTree = new TTree("MonopoleTree", "Magnetic Monopole Candidate Analysis");
    fMonopoleTree->Branch("eventId", &fEventId, "eventId/I");
    fMonopoleTree->Branch("stationId", &fStationId, "stationId/I");
    fMonopoleTree->Branch("monopoleScore", &fMonopoleScore, "monopoleScore/D");
    fMonopoleTree->Branch("totalCharge", &fTotalCharge, "totalCharge/D");
    fMonopoleTree->Branch("signalDuration", &fSignalDuration, "signalDuration/D");
    fMonopoleTree->Branch("signalSmoothness", &fSignalSmoothness, "signalSmoothness/D");
    fMonopoleTree->Branch("sustainedFraction", &fSustainedFraction, "sustainedFraction/D");
    fMonopoleTree->Branch("peakToPlateau", &fPeakToPlateau, "peakToPlateau/D");
    fMonopoleTree->Branch("isCandidate", &fIsCandidate, "isCandidate/O");
    fMonopoleTree->Branch("energy", &fEnergy, "energy/D");
    fMonopoleTree->Branch("coreDistance", &fCoreDistance, "coreDistance/D");
    fMonopoleTree->Branch("zenith", &fZenith, "zenith/D");
    
    INFO("MonopoleTriggerModule initialized successfully");
    return eSuccess;
}

// Run method - Process one event
VModule::ResultFlag MonopoleTriggerModule::Run(Event& event)
{
    fEventCount++;
    fEventId = fEventCount;
    
    if (fEventCount % 100 == 0) {
        ostringstream msg;
        msg << "Processing event " << fEventCount 
            << " - Found " << fMonopoleCandidates << " monopole candidates so far";
        INFO(msg.str());
    }
    
    // Get event info
    fEnergy = 0;
    fZenith = 0;
    double coreX = 0, coreY = 0;
    
    if (event.HasSimShower()) {
        const ShowerSimData& shower = event.GetSimShower();
        fEnergy = shower.GetEnergy();
        fZenith = shower.GetGroundParticleCoordinateSystemZenith() * 180.0 / M_PI;
        
        if (shower.GetNSimCores() > 0) {
            const Detector& detector = Detector::GetInstance();
            const CoordinateSystemPtr& siteCS = detector.GetSiteCoordinateSystem();
            Point core = shower.GetSimCore(0);
            coreX = core.GetX(siteCS);
            coreY = core.GetY(siteCS);
        }
    }
    
    // Process stations
    vector<MonopoleSignatureAnalyzer::MonopoleFeatures> eventFeatures;
    vector<double> stationDistances;
    int monopoleLikeStations = 0;
    
    if (event.HasSEvent()) {
        const SEvent& sevent = event.GetSEvent();
        const Detector& detector = Detector::GetInstance();
        const SDetector& sdetector = detector.GetSDetector();
        
        // Analyze each station
        for (SEvent::ConstStationIterator it = sevent.StationsBegin(); 
             it != sevent.StationsEnd(); ++it) {
            
            const Station& station = *it;
            fStationId = station.GetId();
            
            if (!station.HasSimData()) continue;
            
            // Get station position
            double stationX = 0, stationY = 0;
            try {
                const sdet::Station& detStation = sdetector.GetStation(fStationId);
                const CoordinateSystemPtr& siteCS = detector.GetSiteCoordinateSystem();
                stationX = detStation.GetPosition().GetX(siteCS);
                stationY = detStation.GetPosition().GetY(siteCS);
                fCoreDistance = sqrt(pow(stationX - coreX, 2) + pow(stationY - coreY, 2));
            } catch (...) {
                fCoreDistance = -1;
                continue;
            }
            
            stationDistances.push_back(fCoreDistance);
            
            // Process PMTs
            const int firstPMT = sdet::Station::GetFirstPMTId();
            
            for (int p = 0; p < 3; p++) {
                const int pmtId = p + firstPMT;
                
                if (!station.HasPMT(pmtId)) continue;
                
                const PMT& pmt = station.GetPMT(pmtId);
                
                // Get FADC trace
                vector<double> trace_data;
                bool gotTrace = false;
                
                if (pmt.HasFADCTrace()) {
                    try {
                        const auto& trace = pmt.GetFADCTrace(sdet::PMTConstants::eHighGain);
                        for (int i = 0; i < 2048; i++) {
                            trace_data.push_back(trace[i]);
                        }
                        gotTrace = true;
                    } catch (...) {
                        // Try simulation data
                    }
                }
                
                // Fallback to simulation data
                if (!gotTrace && pmt.HasSimData()) {
                    try {
                        const PMTSimData& simData = pmt.GetSimData();
                        if (simData.HasFADCTrace(StationConstants::eTotal)) {
                            const auto& fadcTrace = simData.GetFADCTrace(
                                sdet::PMTConstants::eHighGain,
                                StationConstants::eTotal
                            );
                            for (int i = 0; i < 2048; i++) {
                                trace_data.push_back(fadcTrace[i]);
                            }
                            gotTrace = true;
                        }
                    } catch (...) {}
                }
                
                if (!gotTrace || trace_data.empty()) continue;
                
                // Analyze trace for monopole signatures
                MonopoleSignatureAnalyzer::MonopoleFeatures features = 
                    MonopoleSignatureAnalyzer::AnalyzeTrace(trace_data);
                
                eventFeatures.push_back(features);
                fStationsAnalyzed++;
                
                // Store analysis results
                fMonopoleScore = features.monopoleScore;
                fTotalCharge = features.totalCharge;
                fSignalDuration = features.signalDuration;
                fSignalSmoothness = features.signalSmoothness;
                fSustainedFraction = features.sustainedSignalFraction;
                fPeakToPlateau = features.peakToPlateau;
                fIsCandidate = features.isMonopoleCandidate;
                
                // Fill histograms
                hMonopoleScore->Fill(fMonopoleScore);
                hSustainedSignal->Fill(fSignalDuration);
                hSignalSmoothness->Fill(fSignalSmoothness);
                hChargeVsSmoothnessMap->Fill(fTotalCharge, fSignalSmoothness);
                
                if (fIsCandidate) {
                    monopoleLikeStations++;
                    hTotalChargeMonopole->Fill(fTotalCharge);
                    
                    // Report significant candidates
                    if (fMonopoleScore > 0.8) {
                        ostringstream msg;
                        msg << "*** MONOPOLE CANDIDATE ***\n"
                            << "  Event: " << fEventId << ", Station: " << fStationId << "\n"
                            << "  Score: " << fixed << setprecision(3) << fMonopoleScore << "\n"
                            << "  Total Charge: " << fTotalCharge << " VEM\n"
                            << "  Duration: " << fSignalDuration << " ns\n"
                            << "  Smoothness: " << fSignalSmoothness << "\n"
                            << "  Energy: " << fEnergy/1e18 << " EeV";
                        INFO(msg.str());
                    }
                }
                
                // Fill tree
                fMonopoleTree->Fill();
            }
        }
    }
    
    // Check for monopole track pattern
    if (monopoleLikeStations >= 3) {
        bool hasTrackPattern = MonopoleSignatureAnalyzer::CheckTrackPattern(
            eventFeatures, stationDistances);
        
        if (hasTrackPattern) {
            fMonopoleCandidates++;
            
            ostringstream msg;
            msg << "*** MONOPOLE EVENT CANDIDATE ***\n"
                << "  Event: " << fEventId << "\n"
                << "  Stations with monopole-like signals: " << monopoleLikeStations << "\n"
                << "  Consistent track pattern: YES\n"
                << "  Event energy: " << fEnergy/1e18 << " EeV";
            INFO(msg.str());
            
            // Mark event for special handling
            // In real analysis, this would trigger special data storage
        }
    }
    
    hStationsPerEvent->Fill(monopoleLikeStations);
    
    return eSuccess;
}

// Finish method
VModule::ResultFlag MonopoleTriggerModule::Finish()
{
    INFO("MonopoleTriggerModule::Finish() - Analysis complete");
    
    ostringstream summary;
    summary << "=== Monopole Analysis Summary ===\n"
            << "Events processed: " << fEventCount << "\n"
            << "Stations analyzed: " << fStationsAnalyzed << "\n"
            << "Monopole event candidates: " << fMonopoleCandidates << "\n";
    
    if (fMonopoleCandidates > 0) {
        double rate = (double)fMonopoleCandidates / fEventCount;
        summary << "Candidate rate: " << rate * 100 << "%\n";
        
        // In real analysis, calculate flux limits
        double exposureKmYr = 3000 * (fEventCount / 365.0);  // Simplified
        double fluxLimit = fMonopoleCandidates / exposureKmYr;
        summary << "Estimated flux limit: " << fluxLimit << " km^-2 yr^-1\n";
    }
    
    summary << "=================================";
    INFO(summary.str());
    
    // Save results
    if (fOutputFile) {
        fOutputFile->Write();
        fOutputFile->Close();
        delete fOutputFile;
        INFO("Results saved to monopole_candidates.root");
    }
    
    return eSuccess;
}
