"""
Spectrum Plotting Engine
Provides comprehensive plotting capabilities for spectrum data
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any
from enum import Enum
import math

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    patches = None
    Figure = None
    FigureCanvasAgg = None
    np = None

from spectrum_data_models import SpectrumData
from analysis_engine import AnalysisResult, Peak

class PlotType(Enum):
    """Types of spectrum plots"""
    LINEAR = "linear"
    SEMILOG = "semilog"
    LOGLOG = "loglog"
    ENERGY = "energy"
    OVERLAY = "overlay"

class PlotStyle(Enum):
    """Plot visual styles"""
    LINE = "line"
    STEPS = "steps"
    HISTOGRAM = "histogram"
    MARKERS = "markers"

@dataclass
class PlotOptions:
    """Configuration for spectrum plots"""
    # Basic plot settings
    plot_type: PlotType = PlotType.LINEAR
    plot_style: PlotStyle = PlotStyle.LINE
    title: str = "Spectrum Plot"
    x_label: str = "Channel"
    y_label: str = "Counts"
    
    # Scale settings
    x_scale: str = "linear"  # linear, log
    y_scale: str = "linear"  # linear, log
    x_range: Optional[Tuple[float, float]] = None
    y_range: Optional[Tuple[float, float]] = None
    
    # Visual settings
    line_color: str = "blue"
    line_width: float = 1.0
    marker_style: str = "."
    marker_size: float = 4.0
    alpha: float = 1.0
    
    # Grid and annotations
    show_grid: bool = True
    grid_alpha: float = 0.3
    show_peaks: bool = True
    peak_marker: str = "v"
    peak_color: str = "red"
    peak_size: float = 8.0
    
    # Background
    show_background: bool = False
    background_color: str = "orange"
    background_alpha: float = 0.3
    
    # Energy axis
    use_energy_axis: bool = False
    energy_unit: str = "keV"
    
    # Statistics text
    show_statistics: bool = True
    stats_position: str = "upper right"  # matplotlib legend positions
    
    # Figure settings
    figure_size: Tuple[float, float] = (10, 6)
    dpi: int = 100
    tight_layout: bool = True

@dataclass
class PlotData:
    """Data container for plotting"""
    x_values: List[float]
    y_values: List[float]
    label: str = ""
    spectrum: Optional[SpectrumData] = None
    analysis: Optional[AnalysisResult] = None

class PlotEngine:
    """Base plotting engine interface"""
    
    def __init__(self):
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is required for plotting functionality")
        
        self.figure: Optional[Figure] = None
        self.canvas = None
        self._plot_data: List[PlotData] = []
        
    def create_figure(self, options: PlotOptions) -> Figure:
        """Create a new figure with specified options"""
        self.figure = Figure(figsize=options.figure_size, dpi=options.dpi)
        return self.figure
    
    def add_spectrum(self, spectrum: SpectrumData, 
                    analysis: Optional[AnalysisResult] = None,
                    label: str = "") -> None:
        """Add spectrum data to plot"""
        if not spectrum:
            raise ValueError("Spectrum data is required")
        
        plot_data = PlotData(
            x_values=list(spectrum.channels),
            y_values=list(spectrum.counts),
            label=label or "Spectrum",
            spectrum=spectrum,
            analysis=analysis
        )
        self._plot_data.append(plot_data)
    
    def clear(self) -> None:
        """Clear all plot data"""
        self._plot_data.clear()
        if self.figure:
            self.figure.clear()
    
    def plot(self, options: PlotOptions) -> Figure:
        """Create the plot with specified options"""
        if not self._plot_data:
            raise ValueError("No data to plot")
        
        # Create figure if needed
        if not self.figure:
            self.create_figure(options)
        
        # Clear existing axes
        self.figure.clear()
        
        # Create main axes
        ax = self.figure.add_subplot(111)
        
        # Plot each spectrum
        for i, data in enumerate(self._plot_data):
            self._plot_spectrum(ax, data, options, color_index=i)
        
        # Configure axes
        self._configure_axes(ax, options)
        
        # Add annotations
        self._add_annotations(ax, options)
        
        # Apply layout
        if options.tight_layout:
            self.figure.tight_layout()
        
        return self.figure
    
    def _plot_spectrum(self, ax, data: PlotData, options: PlotOptions, 
                      color_index: int = 0) -> None:
        """Plot a single spectrum"""
        # Determine x-axis values
        if options.use_energy_axis and data.spectrum and data.spectrum.metadata.has_energy_calibration:
            x_values = []
            for channel in data.x_values:
                energy = data.spectrum.metadata.calculate_energy(int(channel))
                if energy is not None:
                    x_values.append(energy)
                else:
                    x_values.append(channel)
            x_label = f"Energy ({options.energy_unit})"
        else:
            x_values = data.x_values
            x_label = options.x_label
        
        y_values = data.y_values
        
        # Choose color if not specified
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        color = colors[color_index % len(colors)] if options.line_color == "blue" else options.line_color
        
        # Plot based on style
        if options.plot_style == PlotStyle.LINE:
            ax.plot(x_values, y_values, color=color, linewidth=options.line_width,
                   alpha=options.alpha, label=data.label)
        elif options.plot_style == PlotStyle.STEPS:
            ax.step(x_values, y_values, where='mid', color=color, 
                   linewidth=options.line_width, alpha=options.alpha, label=data.label)
        elif options.plot_style == PlotStyle.HISTOGRAM:
            ax.bar(x_values, y_values, width=1.0, color=color, alpha=options.alpha,
                  edgecolor='none', label=data.label)
        elif options.plot_style == PlotStyle.MARKERS:
            ax.scatter(x_values, y_values, c=color, s=options.marker_size,
                      marker=options.marker_style, alpha=options.alpha, label=data.label)
        
        # Plot background if available and requested
        if (options.show_background and data.analysis and 
            data.analysis.background_profile):
            bg_x = x_values if len(data.analysis.background_profile) == len(x_values) else data.x_values
            ax.fill_between(bg_x, data.analysis.background_profile, 
                           alpha=options.background_alpha, 
                           color=options.background_color,
                           label="Background")
        
        # Plot peaks if available and requested
        if options.show_peaks and data.analysis and data.analysis.peaks:
            self._plot_peaks(ax, data, options, x_values)
    
    def _plot_peaks(self, ax, data: PlotData, options: PlotOptions, 
                   x_values: List[float]) -> None:
        """Plot peak markers"""
        peaks = data.analysis.peaks
        if not peaks:
            return
        
        peak_x = []
        peak_y = []
        
        for peak in peaks:
            if options.use_energy_axis and peak.energy is not None:
                peak_x.append(peak.energy)
            else:
                peak_x.append(peak.channel)
            peak_y.append(peak.counts)
        
        # Plot peak markers
        ax.scatter(peak_x, peak_y, c=options.peak_color, s=options.peak_size**2,
                  marker=options.peak_marker, zorder=5, 
                  label=f"Peaks ({len(peaks)} found)")
        
        # Annotate major peaks
        major_peaks = sorted(peaks, key=lambda p: p.counts, reverse=True)[:5]
        for peak in major_peaks:
            x_pos = peak.energy if (options.use_energy_axis and peak.energy) else peak.channel
            y_pos = peak.counts
            
            # Add peak annotation
            if peak.energy and options.use_energy_axis:
                annotation = f"{peak.energy:.1f} {options.energy_unit}"
            else:
                annotation = f"Ch {peak.channel}"
            
            ax.annotate(annotation, xy=(x_pos, y_pos), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
    
    def _configure_axes(self, ax, options: PlotOptions) -> None:
        """Configure axis properties"""
        # Set scales
        ax.set_xscale(options.x_scale)
        ax.set_yscale(options.y_scale)
        
        # Set labels
        if options.use_energy_axis:
            ax.set_xlabel(f"Energy ({options.energy_unit})")
        else:
            ax.set_xlabel(options.x_label)
        ax.set_ylabel(options.y_label)
        ax.set_title(options.title)
        
        # Set ranges if specified
        if options.x_range:
            ax.set_xlim(options.x_range)
        if options.y_range:
            ax.set_ylim(options.y_range)
        
        # Configure grid
        if options.show_grid:
            ax.grid(True, alpha=options.grid_alpha)
        
        # Add legend if multiple datasets
        if len(self._plot_data) > 1 or any(data.analysis and data.analysis.peaks for data in self._plot_data):
            ax.legend(loc=options.stats_position)
    
    def _add_annotations(self, ax, options: PlotOptions) -> None:
        """Add statistics and other annotations"""
        if not options.show_statistics or not self._plot_data:
            return
        
        # Get statistics from first spectrum
        data = self._plot_data[0]
        if not data.spectrum:
            return
        
        stats_text = self._build_statistics_text(data)
        
        # Position statistics text
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round', 
               facecolor='white', alpha=0.8), fontsize=9)
    
    def _build_statistics_text(self, data: PlotData) -> str:
        """Build statistics text for annotation"""
        spectrum = data.spectrum
        lines = []
        
        lines.append(f"Total: {spectrum.total_counts:,}")
        lines.append(f"Peak: {spectrum.peak_counts:,}")
        lines.append(f"Channels: {spectrum.channel_count:,}")
        
        if spectrum.count_rate:
            lines.append(f"Rate: {spectrum.count_rate:.1f} cps")
        
        if data.analysis:
            stats = data.analysis.statistics
            if stats.signal_to_noise:
                lines.append(f"S/N: {stats.signal_to_noise:.1f}")
            
            if data.analysis.peaks:
                lines.append(f"Peaks: {len(data.analysis.peaks)}")
        
        return "\n".join(lines)
    
    def save_plot(self, filename: Union[str, Path], 
                 format: str = "png", dpi: int = 300) -> None:
        """Save plot to file"""
        if not self.figure:
            raise ValueError("No figure to save")
        
        self.figure.savefig(filename, format=format, dpi=dpi, 
                           bbox_inches='tight', facecolor='white')

class MultiSpectrumPlotter(PlotEngine):
    """Specialized plotter for multiple spectra"""
    
    def plot_overlay(self, spectra: List[SpectrumData], 
                    labels: Optional[List[str]] = None,
                    options: Optional[PlotOptions] = None) -> Figure:
        """Plot multiple spectra as overlay"""
        if not options:
            options = PlotOptions(plot_type=PlotType.OVERLAY)
        
        self.clear()
        
        # Add all spectra
        for i, spectrum in enumerate(spectra):
            label = labels[i] if labels and i < len(labels) else f"Spectrum {i+1}"
            self.add_spectrum(spectrum, label=label)
        
        return self.plot(options)
    
    def plot_comparison(self, spectrum1: SpectrumData, spectrum2: SpectrumData,
                       labels: Optional[Tuple[str, str]] = None,
                       options: Optional[PlotOptions] = None) -> Figure:
        """Plot two spectra for comparison"""
        if not options:
            options = PlotOptions(title="Spectrum Comparison")
        
        labels = labels or ("Spectrum 1", "Spectrum 2")
        
        return self.plot_overlay([spectrum1, spectrum2], list(labels), options)

class SpectrumPlotter:
    """Main plotting interface"""
    
    def __init__(self):
        if not MATPLOTLIB_AVAILABLE:
            self.available = False
            self._engine = None
        else:
            self.available = True
            self._engine = PlotEngine()
            self._multi_engine = MultiSpectrumPlotter()
    
    def is_available(self) -> bool:
        """Check if plotting is available"""
        return self.available
    
    def create_simple_plot(self, spectrum: SpectrumData, 
                          title: str = "Spectrum") -> Optional[Figure]:
        """Create a simple spectrum plot"""
        if not self.available:
            return None
        
        options = PlotOptions(title=title)
        
        self._engine.clear()
        self._engine.add_spectrum(spectrum)
        return self._engine.plot(options)
    
    def create_analysis_plot(self, analysis_result: AnalysisResult,
                           title: str = "Spectrum Analysis") -> Optional[Figure]:
        """Create plot with analysis annotations"""
        if not self.available:
            return None
        
        options = PlotOptions(
            title=title,
            show_peaks=True,
            show_background=True,
            show_statistics=True
        )
        
        self._engine.clear()
        self._engine.add_spectrum(analysis_result.spectrum, analysis_result)
        return self._engine.plot(options)
    
    def create_energy_plot(self, spectrum: SpectrumData,
                          title: str = "Energy Spectrum") -> Optional[Figure]:
        """Create energy-calibrated plot"""
        if not self.available:
            return None
        
        if not spectrum.metadata.has_energy_calibration:
            raise ValueError("Spectrum must have energy calibration for energy plots")
        
        options = PlotOptions(
            title=title,
            use_energy_axis=True,
            x_label="Energy (keV)",
            show_statistics=True
        )
        
        self._engine.clear()
        self._engine.add_spectrum(spectrum)
        return self._engine.plot(options)
    
    def create_log_plot(self, spectrum: SpectrumData,
                       title: str = "Spectrum (Log Scale)") -> Optional[Figure]:
        """Create logarithmic plot"""
        if not self.available:
            return None
        
        options = PlotOptions(
            title=title,
            y_scale="log",
            show_statistics=True
        )
        
        self._engine.clear()
        self._engine.add_spectrum(spectrum)
        return self._engine.plot(options)
    
    def create_comparison_plot(self, spectra: List[SpectrumData],
                             labels: Optional[List[str]] = None,
                             title: str = "Spectrum Comparison") -> Optional[Figure]:
        """Create comparison plot of multiple spectra"""
        if not self.available:
            return None
        
        options = PlotOptions(title=title, show_statistics=False)
        return self._multi_engine.plot_overlay(spectra, labels, options)
    
    def create_custom_plot(self, spectrum: SpectrumData,
                          analysis: Optional[AnalysisResult] = None,
                          options: Optional[PlotOptions] = None) -> Optional[Figure]:
        """Create custom plot with full options"""
        if not self.available:
            return None
        
        if not options:
            options = PlotOptions()
        
        self._engine.clear()
        self._engine.add_spectrum(spectrum, analysis)
        return self._engine.plot(options)
    
    def save_plot(self, figure: Figure, filename: Union[str, Path],
                 format: str = "png", dpi: int = 300) -> None:
        """Save plot to file"""
        if not self.available or not figure:
            raise ValueError("Plotting not available or no figure provided")
        
        figure.savefig(filename, format=format, dpi=dpi,
                      bbox_inches='tight', facecolor='white')

# Utility functions for quick plotting
def quick_plot(spectrum: SpectrumData, title: str = "Quick Plot") -> Optional[Figure]:
    """Quick plotting function for testing"""
    plotter = SpectrumPlotter()
    if plotter.is_available():
        return plotter.create_simple_plot(spectrum, title)
    return None

def plot_with_peaks(analysis_result: AnalysisResult, 
                   title: str = "Analysis Plot") -> Optional[Figure]:
    """Quick analysis plot with peaks"""
    plotter = SpectrumPlotter()
    if plotter.is_available():
        return plotter.create_analysis_plot(analysis_result, title)
    return None

# Example usage and testing
if __name__ == "__main__":
    if MATPLOTLIB_AVAILABLE:
        print("Plotting engine available")
        print("Features:")
        print("  - Linear and logarithmic plots")
        print("  - Energy-calibrated plots")
        print("  - Peak marking and annotation")
        print("  - Background visualization")
        print("  - Multiple spectrum overlays")
        print("  - Statistical annotations")
        print("  - High-resolution export")
    else:
        print("Matplotlib not available - plotting disabled")
        print("Install matplotlib and numpy to enable plotting features")
