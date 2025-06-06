"""
Spectrum Analysis Engine
Provides comprehensive spectrum analysis capabilities
"""

import math
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Union
from enum import Enum

from spectrum_data_models import SpectrumData, SpectrumMetadata

class PeakSearchMethod(Enum):
    """Peak search algorithms"""
    LOCAL_MAXIMUM = "local_maximum"
    DERIVATIVE = "derivative"
    GAUSSIAN_FIT = "gaussian_fit"

@dataclass(frozen=True)
class Peak:
    """Represents a detected peak"""
    channel: int
    counts: int
    energy: Optional[float] = None
    fwhm: Optional[float] = None  # Full Width at Half Maximum
    area: Optional[int] = None
    significance: Optional[float] = None  # Peak significance over background
    
    @property
    def has_energy(self) -> bool:
        return self.energy is not None

@dataclass(frozen=True)
class SpectrumStatistics:
    """Comprehensive spectrum statistics"""
    total_counts: int
    total_channels: int
    peak_channel: int
    peak_counts: int
    mean_channel: float
    std_deviation: float
    count_rate: Optional[float] = None
    background_level: Optional[float] = None
    noise_level: Optional[float] = None
    signal_to_noise: Optional[float] = None
    
    # Additional statistical measures
    median_count: float = 0.0
    mode_channel: int = 0
    skewness: float = 0.0
    kurtosis: float = 0.0
    
    # Energy-related statistics (if calibrated)
    mean_energy: Optional[float] = None
    energy_range: Optional[Tuple[float, float]] = None

@dataclass
class AnalysisResult:
    """Complete analysis result"""
    spectrum: SpectrumData
    statistics: SpectrumStatistics
    peaks: List[Peak] = field(default_factory=list)
    background_profile: Optional[List[float]] = None
    smoothed_spectrum: Optional[List[int]] = None
    analysis_parameters: Dict[str, any] = field(default_factory=dict)

class BackgroundEstimator:
    """Estimates background in spectrum data"""
    
    @staticmethod
    def linear_interpolation(spectrum: SpectrumData, 
                           background_regions: List[Tuple[int, int]]) -> List[float]:
        """Estimate background using linear interpolation between regions"""
        channels = list(spectrum.channels)
        counts = list(spectrum.counts)
        background = [0.0] * len(channels)
        
        if not background_regions:
            # If no regions specified, use simple percentile method
            return BackgroundEstimator.percentile_method(spectrum, percentile=10)
        
        # Sort background regions
        regions = sorted(background_regions)
        
        # Calculate average counts in each background region
        region_points = []
        for start_ch, end_ch in regions:
            start_idx = next((i for i, ch in enumerate(channels) if ch >= start_ch), 0)
            end_idx = next((i for i, ch in enumerate(channels) if ch > end_ch), len(channels))
            
            if start_idx < end_idx:
                region_counts = counts[start_idx:end_idx]
                avg_count = sum(region_counts) / len(region_counts)
                mid_channel = (start_ch + end_ch) / 2
                region_points.append((mid_channel, avg_count))
        
        # Interpolate between region points
        if len(region_points) < 2:
            # Not enough points for interpolation
            avg_bg = region_points[0][1] if region_points else 0
            return [avg_bg] * len(channels)
        
        for i, channel in enumerate(channels):
            # Find surrounding region points
            if channel <= region_points[0][0]:
                background[i] = region_points[0][1]
            elif channel >= region_points[-1][0]:
                background[i] = region_points[-1][1]
            else:
                # Linear interpolation
                for j in range(len(region_points) - 1):
                    ch1, count1 = region_points[j]
                    ch2, count2 = region_points[j + 1]
                    
                    if ch1 <= channel <= ch2:
                        if ch2 == ch1:
                            background[i] = count1
                        else:
                            ratio = (channel - ch1) / (ch2 - ch1)
                            background[i] = count1 + ratio * (count2 - count1)
                        break
        
        return background
    
    @staticmethod
    def percentile_method(spectrum: SpectrumData, percentile: float = 10) -> List[float]:
        """Estimate background using percentile method"""
        non_zero_counts = [c for c in spectrum.counts if c > 0]
        if not non_zero_counts:
            return [0.0] * len(spectrum.counts)
        
        background_level = sorted(non_zero_counts)[int(len(non_zero_counts) * percentile / 100)]
        return [float(background_level)] * len(spectrum.counts)
    
    @staticmethod
    def moving_minimum(spectrum: SpectrumData, window_size: int = 50) -> List[float]:
        """Estimate background using moving minimum window"""
        counts = list(spectrum.counts)
        background = []
        
        half_window = window_size // 2
        
        for i in range(len(counts)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(counts), i + half_window + 1)
            
            window_counts = counts[start_idx:end_idx]
            background.append(float(min(window_counts)))
        
        return background

class PeakFinder:
    """Advanced peak finding algorithms"""
    
    def __init__(self):
        self.background_estimator = BackgroundEstimator()
    
    def find_peaks_local_maximum(self, spectrum: SpectrumData, 
                                min_height: float = 0, 
                                min_distance: int = 1,
                                background: Optional[List[float]] = None) -> List[Peak]:
        """Find peaks using local maximum method"""
        if background is None:
            background = self.background_estimator.percentile_method(spectrum)
        
        peaks = []
        channels = list(spectrum.channels)
        counts = list(spectrum.counts)
        
        # Subtract background
        net_counts = [max(0, c - bg) for c, bg in zip(counts, background)]
        
        for i in range(1, len(net_counts) - 1):
            if net_counts[i] < min_height:
                continue
            
            # Check if it's a local maximum
            is_peak = True
            
            # Check immediate neighbors
            if net_counts[i] <= net_counts[i-1] or net_counts[i] <= net_counts[i+1]:
                is_peak = False
            
            # Check minimum distance
            if is_peak and min_distance > 1:
                for j in range(max(0, i - min_distance), min(len(net_counts), i + min_distance + 1)):
                    if j != i and net_counts[j] >= net_counts[i]:
                        is_peak = False
                        break
            
            if is_peak:
                channel = channels[i]
                count = counts[i]
                net_count = net_counts[i]
                
                # Calculate energy if calibration is available
                energy = spectrum.metadata.calculate_energy(channel)
                
                # Estimate peak significance
                local_bg = background[i]
                significance = net_count / math.sqrt(max(1, local_bg)) if local_bg > 0 else net_count
                
                peak = Peak(
                    channel=channel,
                    counts=count,
                    energy=energy,
                    significance=significance
                )
                peaks.append(peak)
        
        # Sort by counts (descending)
        peaks.sort(key=lambda p: p.counts, reverse=True)
        return peaks
    
    def find_peaks_derivative(self, spectrum: SpectrumData,
                            threshold: float = 0.1) -> List[Peak]:
        """Find peaks using derivative method"""
        counts = list(spectrum.counts)
        channels = list(spectrum.channels)
        
        if len(counts) < 3:
            return []
        
        # Calculate first derivative
        first_derivative = []
        for i in range(1, len(counts) - 1):
            derivative = (counts[i+1] - counts[i-1]) / 2.0
            first_derivative.append(derivative)
        
        # Find zero crossings where derivative changes from positive to negative
        peaks = []
        for i in range(1, len(first_derivative) - 1):
            if (first_derivative[i-1] > threshold and 
                first_derivative[i+1] < -threshold and
                abs(first_derivative[i]) < threshold):
                
                # Found a potential peak at index i+1 in original data
                peak_idx = i + 1
                channel = channels[peak_idx]
                count = counts[peak_idx]
                
                energy = spectrum.metadata.calculate_energy(channel)
                
                peak = Peak(
                    channel=channel,
                    counts=count,
                    energy=energy
                )
                peaks.append(peak)
        
        return peaks
    
    def estimate_fwhm(self, spectrum: SpectrumData, peak_channel: int) -> Optional[float]:
        """Estimate Full Width at Half Maximum for a peak"""
        channels = list(spectrum.channels)
        counts = list(spectrum.counts)
        
        try:
            peak_idx = channels.index(peak_channel)
        except ValueError:
            return None
        
        peak_count = counts[peak_idx]
        half_max = peak_count / 2.0
        
        # Find left half-maximum point
        left_idx = peak_idx
        while left_idx > 0 and counts[left_idx] > half_max:
            left_idx -= 1
        
        # Find right half-maximum point
        right_idx = peak_idx
        while right_idx < len(counts) - 1 and counts[right_idx] > half_max:
            right_idx += 1
        
        # Interpolate to find precise half-maximum points
        if left_idx < peak_idx and counts[left_idx] < half_max:
            # Linear interpolation
            left_frac = (half_max - counts[left_idx]) / (counts[left_idx + 1] - counts[left_idx])
            left_channel = channels[left_idx] + left_frac
        else:
            left_channel = channels[left_idx]
        
        if right_idx > peak_idx and counts[right_idx] < half_max:
            # Linear interpolation
            right_frac = (half_max - counts[right_idx]) / (counts[right_idx - 1] - counts[right_idx])
            right_channel = channels[right_idx] - right_frac
        else:
            right_channel = channels[right_idx]
        
        return right_channel - left_channel

class SpectrumAnalyzer:
    """Main spectrum analysis service"""
    
    def __init__(self):
        self.peak_finder = PeakFinder()
        self.background_estimator = BackgroundEstimator()
    
    def analyze(self, spectrum: SpectrumData, 
               peak_method: PeakSearchMethod = PeakSearchMethod.LOCAL_MAXIMUM,
               **analysis_params) -> AnalysisResult:
        """Perform comprehensive spectrum analysis"""
        
        # Calculate basic statistics
        statistics = self._calculate_statistics(spectrum)
        
        # Estimate background
        background = self._estimate_background(spectrum, analysis_params)
        
        # Find peaks
        peaks = self._find_peaks(spectrum, peak_method, background, analysis_params)
        
        # Enhance peaks with additional information
        enhanced_peaks = self._enhance_peaks(spectrum, peaks, background)
        
        # Create analysis result
        result = AnalysisResult(
            spectrum=spectrum,
            statistics=statistics,
            peaks=enhanced_peaks,
            background_profile=background,
            analysis_parameters=analysis_params
        )
        
        return result
    
    def _calculate_statistics(self, spectrum: SpectrumData) -> SpectrumStatistics:
        """Calculate comprehensive spectrum statistics"""
        channels = list(spectrum.channels)
        counts = list(spectrum.counts)
        
        # Basic statistics
        total_counts = sum(counts)
        total_channels = len(channels)
        peak_counts = max(counts)
        peak_channel = channels[counts.index(peak_counts)]
        
        # Weighted statistics
        if total_counts > 0:
            mean_channel = sum(ch * count for ch, count in zip(channels, counts)) / total_counts
        else:
            mean_channel = 0.0
        
        # Standard deviation
        if total_counts > 1:
            variance = sum(count * (ch - mean_channel)**2 for ch, count in zip(channels, counts)) / (total_counts - 1)
            std_deviation = math.sqrt(max(0, variance))
        else:
            std_deviation = 0.0
        
        # Additional statistics
        non_zero_counts = [c for c in counts if c > 0]
        median_count = statistics.median(non_zero_counts) if non_zero_counts else 0.0
        
        # Mode (most frequent channel)
        max_count = max(counts)
        mode_channel = channels[counts.index(max_count)]
        
        # Background and noise estimation
        background_level = statistics.median(non_zero_counts) if non_zero_counts else 0.0
        
        # Calculate noise level (RMS of deviations from smoothed curve)
        noise_level = self._estimate_noise_level(counts)
        
        # Signal-to-noise ratio
        signal_to_noise = peak_counts / noise_level if noise_level > 0 else None
        
        # Count rate
        count_rate = spectrum.count_rate
        
        # Energy statistics if calibrated
        mean_energy = None
        energy_range = None
        if spectrum.metadata.has_energy_calibration:
            energies = []
            for ch, count in zip(channels, counts):
                energy = spectrum.metadata.calculate_energy(ch)
                if energy is not None:
                    energies.extend([energy] * count)
            
            if energies:
                mean_energy = sum(energies) / len(energies)
                energy_range = (min(energies), max(energies))
        
        return SpectrumStatistics(
            total_counts=total_counts,
            total_channels=total_channels,
            peak_channel=peak_channel,
            peak_counts=peak_counts,
            mean_channel=mean_channel,
            std_deviation=std_deviation,
            count_rate=count_rate,
            background_level=background_level,
            noise_level=noise_level,
            signal_to_noise=signal_to_noise,
            median_count=median_count,
            mode_channel=mode_channel,
            mean_energy=mean_energy,
            energy_range=energy_range
        )
    
    def _estimate_noise_level(self, counts: List[int]) -> float:
        """Estimate noise level in spectrum"""
        if len(counts) < 3:
            return 0.0
        
        # Calculate differences between adjacent points
        differences = [abs(counts[i+1] - counts[i]) for i in range(len(counts) - 1)]
        
        # RMS of differences
        mean_diff_sq = sum(d**2 for d in differences) / len(differences)
        return math.sqrt(mean_diff_sq)
    
    def _estimate_background(self, spectrum: SpectrumData, 
                           params: Dict[str, any]) -> List[float]:
        """Estimate background using specified method"""
        bg_method = params.get('background_method', 'percentile')
        
        if bg_method == 'percentile':
            percentile = params.get('background_percentile', 10)
            return self.background_estimator.percentile_method(spectrum, percentile)
        elif bg_method == 'linear':
            regions = params.get('background_regions', [])
            return self.background_estimator.linear_interpolation(spectrum, regions)
        elif bg_method == 'moving_minimum':
            window_size = params.get('background_window', 50)
            return self.background_estimator.moving_minimum(spectrum, window_size)
        else:
            return self.background_estimator.percentile_method(spectrum)
    
    def _find_peaks(self, spectrum: SpectrumData, method: PeakSearchMethod,
                   background: List[float], params: Dict[str, any]) -> List[Peak]:
        """Find peaks using specified method"""
        if method == PeakSearchMethod.LOCAL_MAXIMUM:
            min_height = params.get('peak_min_height', 10)
            min_distance = params.get('peak_min_distance', 1)
            return self.peak_finder.find_peaks_local_maximum(
                spectrum, min_height, min_distance, background
            )
        elif method == PeakSearchMethod.DERIVATIVE:
            threshold = params.get('derivative_threshold', 0.1)
            return self.peak_finder.find_peaks_derivative(spectrum, threshold)
        else:
            # Default to local maximum
            return self.peak_finder.find_peaks_local_maximum(spectrum, background=background)
    
    def _enhance_peaks(self, spectrum: SpectrumData, peaks: List[Peak], 
                      background: List[float]) -> List[Peak]:
        """Enhance peaks with additional information"""
        enhanced_peaks = []
        
        for peak in peaks:
            # Estimate FWHM
            fwhm = self.peak_finder.estimate_fwhm(spectrum, peak.channel)
            
            # Estimate peak area (simple integration)
            area = self._estimate_peak_area(spectrum, peak, background)
            
            # Create enhanced peak
            enhanced_peak = Peak(
                channel=peak.channel,
                counts=peak.counts,
                energy=peak.energy,
                fwhm=fwhm,
                area=area,
                significance=peak.significance
            )
            enhanced_peaks.append(enhanced_peak)
        
        return enhanced_peaks
    
    def _estimate_peak_area(self, spectrum: SpectrumData, peak: Peak, 
                           background: List[float]) -> int:
        """Estimate peak area using simple integration"""
        channels = list(spectrum.channels)
        counts = list(spectrum.counts)
        
        try:
            peak_idx = channels.index(peak.channel)
        except ValueError:
            return peak.counts
        
        # Find peak boundaries (simple method)
        half_max = peak.counts / 2.0
        
        # Find left boundary
        left_idx = peak_idx
        while left_idx > 0 and counts[left_idx] > half_max:
            left_idx -= 1
        
        # Find right boundary  
        right_idx = peak_idx
        while right_idx < len(counts) - 1 and counts[right_idx] > half_max:
            right_idx += 1
        
        # Integrate peak area above background
        area = 0
        for i in range(left_idx, right_idx + 1):
            if i < len(background):
                net_count = max(0, counts[i] - background[i])
                area += net_count
            else:
                area += counts[i]
        
        return int(area)
    
    def export_analysis_report(self, result: AnalysisResult, 
                              output_path: Union[str, Path]) -> None:
        """Export detailed analysis report"""
        output_path = Path(output_path)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("SPECTRUM ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # File information
            f.write("FILE INFORMATION:\n")
            f.write("-" * 16 + "\n")
            meta = result.spectrum.metadata
            if meta.source_file:
                f.write(f"Source File: {meta.source_file.name}\n")
            if meta.date:
                f.write(f"Measurement Date: {meta.date}\n")
            if meta.device:
                f.write(f"Device: {meta.device}\n")
            f.write("\n")
            
            # Statistics
            f.write("SPECTRUM STATISTICS:\n")
            f.write("-" * 19 + "\n")
            stats = result.statistics
            f.write(f"Total Channels: {stats.total_channels:,}\n")
            f.write(f"Total Counts: {stats.total_counts:,}\n")
            f.write(f"Peak Channel: {stats.peak_channel}\n")
            f.write(f"Peak Counts: {stats.peak_counts:,}\n")
            f.write(f"Mean Channel: {stats.mean_channel:.2f}\n")
            f.write(f"Standard Deviation: {stats.std_deviation:.2f}\n")
            
            if stats.count_rate:
                f.write(f"Count Rate: {stats.count_rate:.2f} cps\n")
            if stats.signal_to_noise:
                f.write(f"Signal-to-Noise: {stats.signal_to_noise:.2f}\n")
            
            if stats.mean_energy:
                f.write(f"Mean Energy: {stats.mean_energy:.2f} keV\n")
            if stats.energy_range:
                f.write(f"Energy Range: {stats.energy_range[0]:.2f} - {stats.energy_range[1]:.2f} keV\n")
            f.write("\n")
            
            # Peaks
            if result.peaks:
                f.write("DETECTED PEAKS:\n")
                f.write("-" * 14 + "\n")
                f.write("Rank  Channel   Counts    Energy    FWHM     Area    Significance\n")
                f.write("----  -------  --------  --------  -------  ------  ------------\n")
                
                for i, peak in enumerate(result.peaks[:20], 1):  # Top 20 peaks
                    energy_str = f"{peak.energy:.2f}" if peak.energy else "N/A"
                    fwhm_str = f"{peak.fwhm:.2f}" if peak.fwhm else "N/A"
                    area_str = f"{peak.area:,}" if peak.area else "N/A"
                    sig_str = f"{peak.significance:.2f}" if peak.significance else "N/A"
                    
                    f.write(f"{i:4d}  {peak.channel:7d}  {peak.counts:8,}  {energy_str:>8}  "
                           f"{fwhm_str:>7}  {area_str:>6}  {sig_str:>12}\n")
            
            f.write("\n")
            
            # Analysis parameters
            if result.analysis_parameters:
                f.write("ANALYSIS PARAMETERS:\n")
                f.write("-" * 19 + "\n")
                for key, value in result.analysis_parameters.items():
                    f.write(f"{key}: {value}\n")

# Example usage
if __name__ == "__main__":
    # This would be used by the main application
    print("Spectrum Analysis Engine ready")
