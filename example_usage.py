"""
Integration Examples and Usage Scripts
Demonstrates complete workflows and advanced usage patterns
"""

# examples/basic_usage.py
"""
Basic usage examples for Spectrum Converter Pro
"""

from pathlib import Path
from spectrum_converter import (
    SpectrumFileReader, ConversionEngine, SpectrumAnalyzer, 
    SpectrumPlotter, BatchConversionManager
)

def example_file_reading():
    """Example: Reading different spectrum file formats"""
    print("=== File Reading Example ===")
    
    reader = SpectrumFileReader()
    
    # Check what formats are supported
    formats = reader.get_supported_formats()
    print("Supported formats:")
    for name, extensions in formats.items():
        print(f"  {name}: {', '.join(extensions)}")
    
    # Read a spectrum file
    try:
        spectrum = reader.read_file("example.spe")
        print(f"\nLoaded spectrum:")
        print(f"  Channels: {spectrum.channel_count:,}")
        print(f"  Total counts: {spectrum.total_counts:,}")
        print(f"  Peak channel: {spectrum.peak_channel}")
        
        # Access metadata
        meta = spectrum.metadata
        if meta.spec_id:
            print(f"  Spectrum ID: {meta.spec_id}")
        if meta.live_time:
            print(f"  Live time: {meta.live_time:.1f} s")
        if meta.has_energy_calibration:
            print("  Energy calibration available")
            
    except FileNotFoundError:
        print("Example file not found - create example.spe first")
    except Exception as e:
        print(f"Error reading file: {e}")

def example_conversion():
    """Example: Converting spectrum files"""
    print("\n=== Conversion Example ===")
    
    engine = ConversionEngine()
    
    # Show available output formats
    formats = engine.get_available_formats()
    print("Available output formats:")
    for key, name in formats.items():
        print(f"  {key}: {name}")
    
    # Convert single file
    try:
        # Create conversion job
        job = engine.create_job(
            source_file="example.spe",
            target_format="txt_full",
            include_energy=True,
            include_statistics=True,
            include_peaks=True
        )
        
        # Progress callback
        def progress_callback(progress, message):
            print(f"  Progress: {progress:.1f}% - {message}")
        
        # Perform conversion
        result = engine.convert_single(job, progress_callback)
        
        if result.success:
            print(f"Conversion successful!")
            print(f"  Output: {result.output_file}")
            print(f"  Time: {result.processing_time:.2f} seconds")
        else:
            print(f"Conversion failed: {result.error_message}")
            
    except Exception as e:
        print(f"Conversion error: {e}")

def example_batch_conversion():
    """Example: Batch conversion of multiple files"""
    print("\n=== Batch Conversion Example ===")
    
    engine = ConversionEngine()
    batch_manager = BatchConversionManager(engine, max_workers=4)
    
    # Find files in directory
    data_dir = Path("./sample_data")
    if data_dir.exists():
        files = batch_manager.find_convertible_files(data_dir, "txt_full")
        print(f"Found {len(files)} convertible files")
        
        if files:
            # Create batch jobs
            jobs = batch_manager.create_batch_jobs(
                files=files,
                target_format="txt_full",
                create_subfolders=True,
                include_statistics=True
            )
            
            # Progress tracking
            def batch_progress(progress, message):
                print(f"  Batch: {progress:.1f}% - {message}")
            
            # Execute batch conversion
            results = batch_manager.convert_batch(jobs, batch_progress)
            
            # Report results
            successful = sum(1 for r in results if r.success)
            print(f"Batch conversion complete: {successful}/{len(results)} files")
            
            if successful < len(results):
                failed = [r.job.source_file.name for r in results if not r.success]
                print(f"Failed files: {', '.join(failed)}")
    else:
        print("Sample data directory not found")

def example_analysis():
    """Example: Comprehensive spectrum analysis"""
    print("\n=== Analysis Example ===")
    
    reader = SpectrumFileReader()
    analyzer = SpectrumAnalyzer()
    
    try:
        # Load spectrum
        spectrum = reader.read_file("example.spe")
        
        # Perform analysis with custom parameters
        analysis = analyzer.analyze(
            spectrum,
            peak_min_height=50,
            peak_min_distance=3,
            background_percentile=10,
            background_method="percentile"
        )
        
        # Display results
        stats = analysis.statistics
        print("Analysis Results:")
        print(f"  Total counts: {stats.total_counts:,}")
        print(f"  Mean channel: {stats.mean_channel:.2f}")
        print(f"  Standard deviation: {stats.std_deviation:.2f}")
        
        if stats.count_rate:
            print(f"  Count rate: {stats.count_rate:.2f} cps")
        if stats.signal_to_noise:
            print(f"  Signal/Noise: {stats.signal_to_noise:.2f}")
        
        # Peak information
        peaks = analysis.peaks
        print(f"\nPeaks found: {len(peaks)}")
        for i, peak in enumerate(peaks[:5], 1):  # Top 5 peaks
            print(f"  Peak {i}: Channel {peak.channel}, {peak.counts:,} counts")
            if peak.energy:
                print(f"    Energy: {peak.energy:.2f} keV")
            if peak.fwhm:
                print(f"    FWHM: {peak.fwhm:.2f}")
        
        # Export detailed report
        analyzer.export_analysis_report(analysis, "analysis_report.txt")
        print("\nDetailed report exported to analysis_report.txt")
        
    except Exception as e:
        print(f"Analysis error: {e}")

def example_plotting():
    """Example: Creating plots and visualizations"""
    print("\n=== Plotting Example ===")
    
    plotter = SpectrumPlotter()
    
    if not plotter.is_available():
        print("Plotting not available - matplotlib not installed")
        return
    
    reader = SpectrumFileReader()
    analyzer = SpectrumAnalyzer()
    
    try:
        # Load and analyze spectrum
        spectrum = reader.read_file("example.spe")
        analysis = analyzer.analyze(spectrum, peak_min_height=20)
        
        # Create different types of plots
        print("Creating plots...")
        
        # Simple linear plot
        fig1 = plotter.create_simple_plot(spectrum, "Simple Spectrum Plot")
        if fig1:
            plotter.save_plot(fig1, "simple_plot.png", dpi=150)
            print("  Saved: simple_plot.png")
        
        # Analysis plot with peaks
        fig2 = plotter.create_analysis_plot(analysis, "Spectrum Analysis")
        if fig2:
            plotter.save_plot(fig2, "analysis_plot.png", dpi=150)
            print("  Saved: analysis_plot.png")
        
        # Energy-calibrated plot (if calibration available)
        if spectrum.metadata.has_energy_calibration:
            fig3 = plotter.create_energy_plot(spectrum, "Energy Spectrum")
            if fig3:
                plotter.save_plot(fig3, "energy_plot.png", dpi=150)
                print("  Saved: energy_plot.png")
        
        # Logarithmic plot
        fig4 = plotter.create_log_plot(spectrum, "Log Scale Spectrum")
        if fig4:
            plotter.save_plot(fig4, "log_plot.png", dpi=150)
            print("  Saved: log_plot.png")
        
        # Custom plot with advanced options
        from spectrum_converter.plotting_engine import PlotOptions
        
        options = PlotOptions(
            title="Publication Quality Plot",
            use_energy_axis=spectrum.metadata.has_energy_calibration,
            show_peaks=True,
            show_background=True,
            show_statistics=True,
            figure_size=(12, 8),
            dpi=300
        )
        
        fig5 = plotter.create_custom_plot(spectrum, analysis, options)
        if fig5:
            plotter.save_plot(fig5, "publication_plot.pdf", format="pdf", dpi=300)
            print("  Saved: publication_plot.pdf")
        
    except Exception as e:
        print(f"Plotting error: {e}")

def example_complete_workflow():
    """Example: Complete workflow from file to analysis to visualization"""
    print("\n=== Complete Workflow Example ===")
    
    # Initialize all components
    reader = SpectrumFileReader()
    engine = ConversionEngine()
    analyzer = SpectrumAnalyzer()
    plotter = SpectrumPlotter()
    
    input_file = "example.spe"
    
    try:
        # Step 1: Read the spectrum file
        print("Step 1: Reading spectrum file...")
        spectrum = reader.read_file(input_file)
        print(f"  Loaded {spectrum.channel_count:,} channels, {spectrum.total_counts:,} total counts")
        
        # Step 2: Convert to different formats
        print("\nStep 2: Converting to multiple formats...")
        
        formats = ["txt_full", "txt_simple", "z1d"]
        for fmt in formats:
            job = engine.create_job(input_file, fmt)
            result = engine.convert_single(job)
            if result.success:
                print(f"  ✓ Converted to {fmt}: {result.output_file}")
            else:
                print(f"  ✗ Failed to convert to {fmt}: {result.error_message}")
        
        # Step 3: Perform comprehensive analysis
        print("\nStep 3: Analyzing spectrum...")
        analysis = analyzer.analyze(
            spectrum,
            peak_min_height=30,
            peak_min_distance=2,
            background_percentile=10
        )
        
        stats = analysis.statistics
        print(f"  Analysis complete: {len(analysis.peaks)} peaks found")
        print(f"  S/N ratio: {stats.signal_to_noise:.2f}" if stats.signal_to_noise else "  S/N ratio: N/A")
        
        # Step 4: Create visualizations
        print("\nStep 4: Creating visualizations...")
        if plotter.is_available():
            # Analysis plot
            fig = plotter.create_analysis_plot(analysis, "Complete Workflow Analysis")
            if fig:
                plotter.save_plot(fig, "workflow_analysis.png", dpi=200)
                print("  ✓ Created analysis plot: workflow_analysis.png")
        else:
            print("  Plotting not available")
        
        # Step 5: Export comprehensive report
        print("\nStep 5: Exporting reports...")
        analyzer.export_analysis_report(analysis, "workflow_report.txt")
        print("  ✓ Exported analysis report: workflow_report.txt")
        
        # Create summary report
        with open("workflow_summary.txt", "w") as f:
            f.write("SPECTRUM ANALYSIS WORKFLOW SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Input file: {input_file}\n")
            f.write(f"Total counts: {spectrum.total_counts:,}\n")
            f.write(f"Peak counts: {spectrum.peak_counts:,}\n")
            f.write(f"Channels: {spectrum.channel_count:,}\n")
            f.write(f"Peaks detected: {len(analysis.peaks)}\n")
            if stats.count_rate:
                f.write(f"Count rate: {stats.count_rate:.2f} cps\n")
            
            f.write(f"\nTop 5 peaks:\n")
            for i, peak in enumerate(analysis.peaks[:5], 1):
                f.write(f"  {i}. Channel {peak.channel}: {peak.counts:,} counts")
                if peak.energy:
                    f.write(f" ({peak.energy:.2f} keV)")
                f.write("\n")
        
        print("  ✓ Created workflow summary: workflow_summary.txt")
        print("\nWorkflow completed successfully!")
        
    except Exception as e:
        print(f"Workflow error: {e}")

def main():
    """Run all examples"""
    print("Spectrum Converter Pro - Usage Examples")
    print("=" * 50)
    
    # Create sample data if it doesn't exist
    create_sample_data()
    
    # Run examples
    example_file_reading()
    example_conversion()
    example_batch_conversion()
    example_analysis()
    example_plotting()
    example_complete_workflow()
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("Check the generated files for results.")

def create_sample_data():
    """Create sample SPE file for examples"""
    sample_spe_content = """$SPEC_ID:
EXAMPLE_SPECTRUM_001
$DATE_MEA:
06/06/2025 15:30:00
$MEAS_TIM:
1000.0 1050.0
$DEVICE_ID:
EXAMPLE_DETECTOR
$SPEC_CAL:
0.5 1.0 0.001
$DATA:
0 4095
10 15 20 25 30 35 40 45 50 55
60 65 70 75 80 85 90 95 100 120
150 200 180 160 140 120 100 80 60 40
30 25 20 18 16 14 12 10 8 6
5 4 3 3 2 2 2 1 1 1
1 1 0 0 0 0 0 0 0 0
""" + "0 " * 4000  # Add more zeros to reach full spectrum
    
    # Create example.spe if it doesn't exist
    if not Path("example.spe").exists():
        with open("example.spe", "w") as f:
            f.write(sample_spe_content)
        print("Created example.spe for demonstrations")
    
    # Create sample data directory
    sample_dir = Path("sample_data")
    if not sample_dir.exists():
        sample_dir.mkdir()
        
        # Create a few sample files
        for i in range(3):
            sample_file = sample_dir / f"sample_{i+1}.spe"
            if not sample_file.exists():
                # Modify the content slightly for each file
                modified_content = sample_spe_content.replace(
                    "EXAMPLE_SPECTRUM_001",
                    f"SAMPLE_SPECTRUM_{i+1:03d}"
                )
                sample_file.write_text(modified_content)
        
        print(f"Created sample data directory with 3 files")

if __name__ == "__main__":
    main()

# examples/advanced_usage.py
"""
Advanced usage patterns and customization examples
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any
import json

from spectrum_converter import *

class CustomAnalysisWorkflow:
    """Example of creating custom analysis workflows"""
    
    def __init__(self):
        self.reader = SpectrumFileReader()
        self.analyzer = SpectrumAnalyzer()
        self.plotter = SpectrumPlotter()
        
    def multi_file_comparison(self, file_paths: List[str], 
                             output_dir: str = "comparison_results"):
        """Compare multiple spectrum files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        spectra = []
        analyses = []
        
        print("Loading and analyzing spectra...")
        for file_path in file_paths:
            try:
                spectrum = self.reader.read_file(file_path)
                analysis = self.analyzer.analyze(spectrum)
                
                spectra.append(spectrum)
                analyses.append(analysis)
                
                print(f"  ✓ Loaded {Path(file_path).name}")
            except Exception as e:
                print(f"  ✗ Failed to load {file_path}: {e}")
        
        if not spectra:
            print("No spectra loaded successfully")
            return
        
        # Create comparison plot
        if self.plotter.is_available() and len(spectra) > 1:
            labels = [f"Spectrum {i+1}" for i in range(len(spectra))]
            fig = self.plotter.create_comparison_plot(spectra, labels, 
                                                    "Multi-Spectrum Comparison")
            if fig:
                self.plotter.save_plot(fig, output_path / "comparison_plot.png", dpi=200)
                print(f"Saved comparison plot")
        
        # Create comparison report
        self._create_comparison_report(spectra, analyses, output_path / "comparison_report.json")
        
    def _create_comparison_report(self, spectra: List[SpectrumData], 
                                analyses: List[Any], output_file: Path):
        """Create JSON comparison report"""
        report = {
            "comparison_date": "2025-06-06",
            "total_spectra": len(spectra),
            "summary": {},
            "detailed_comparison": []
        }
        
        # Summary statistics
        total_counts = [s.total_counts for s in spectra]
        peak_counts = [s.peak_counts for s in spectra]
        peak_nums = [len(a.peaks) for a in analyses]
        
        report["summary"] = {
            "total_counts": {
                "min": min(total_counts),
                "max": max(total_counts),
                "mean": sum(total_counts) / len(total_counts)
            },
            "peak_counts": {
                "min": min(peak_counts),
                "max": max(peak_counts),
                "mean": sum(peak_counts) / len(peak_counts)
            },
            "peaks_detected": {
                "min": min(peak_nums),
                "max": max(peak_nums),
                "mean": sum(peak_nums) / len(peak_nums)
            }
        }
        
        # Detailed comparison
        for i, (spectrum, analysis) in enumerate(zip(spectra, analyses)):
            entry = {
                "spectrum_index": i + 1,
                "source_file": str(spectrum.metadata.source_file) if spectrum.metadata.source_file else None,
                "total_counts": spectrum.total_counts,
                "peak_counts": spectrum.peak_counts,
                "peak_channel": spectrum.peak_channel,
                "peaks_detected": len(analysis.peaks),
                "count_rate": spectrum.count_rate,
                "signal_to_noise": analysis.statistics.signal_to_noise
            }
            report["detailed_comparison"].append(entry)
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Saved comparison report: {output_file}")

class CustomFormatHandler(FileFormatHandler):
    """Example custom format handler for CSV files"""
    
    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.csv'
    
    def parse(self, file_path: Path) -> SpectrumData:
        import csv
        
        channels = []
        counts = []
        
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            
            # Skip header if present
            first_row = next(reader, None)
            if first_row and not first_row[0].isdigit():
                pass  # Skip header
            else:
                # Process first row as data
                if first_row:
                    channels.append(int(first_row[0]))
                    counts.append(int(first_row[1]))
            
            # Process remaining rows
            for row in reader:
                if len(row) >= 2:
                    channels.append(int(row[0]))
                    counts.append(int(row[1]))
        
        metadata = SpectrumMetadata(
            source_file=file_path,
            file_format=FileFormat.TXT  # Use TXT as closest equivalent
        )
        
        return SpectrumData(tuple(channels), tuple(counts), metadata)
    
    def get_supported_extensions(self) -> List[str]:
        return ['.csv']
    
    def get_format_name(self) -> str:
        return "CSV Spectrum File"

def example_custom_format():
    """Example of adding custom format support"""
    print("=== Custom Format Example ===")
    
    # Create sample CSV file
    csv_content = "channel,counts\n"
    for i in range(100):
        count = max(1, int(50 * np.exp(-((i-50)**2)/200) + np.random.poisson(5)))
        csv_content += f"{i},{count}\n"
    
    csv_file = Path("sample.csv")
    csv_file.write_text(csv_content)
    
    # Add custom handler
    reader = SpectrumFileReader()
    reader.add_handler(CustomFormatHandler())
    
    # Test reading CSV file
    try:
        spectrum = reader.read_file(csv_file)
        print(f"Successfully read CSV file:")
        print(f"  Channels: {spectrum.channel_count}")
        print(f"  Total counts: {spectrum.total_counts}")
        
        # Clean up
        csv_file.unlink()
        
    except Exception as e:
        print(f"Error reading CSV: {e}")

def example_performance_monitoring():
    """Example of monitoring performance during operations"""
    import time
    import psutil
    
    print("=== Performance Monitoring Example ===")
    
    process = psutil.Process()
    
    def monitor_performance(operation_name: str):
        start_time = time.time()
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        def end_monitoring():
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            
            print(f"  {operation_name}:")
            print(f"    Duration: {duration:.2f} seconds")
            print(f"    Memory usage: {end_memory:.1f} MB (Δ{memory_delta:+.1f} MB)")
        
        return end_monitoring
    
    # Monitor file reading
    end_monitor = monitor_performance("File Reading")
    reader = SpectrumFileReader()
    try:
        spectrum = reader.read_file("example.spe")
    except:
        pass
    end_monitor()
    
    # Monitor analysis
    if 'spectrum' in locals():
        end_monitor = monitor_performance("Spectrum Analysis")
        analyzer = SpectrumAnalyzer()
        analysis = analyzer.analyze(spectrum)
        end_monitor()
    
    # Monitor batch conversion
    end_monitor = monitor_performance("Batch Setup")
    engine = ConversionEngine()
    batch_manager = BatchConversionManager(engine, max_workers=2)
    end_monitor()

def main_advanced():
    """Run advanced examples"""
    print("Spectrum Converter Pro - Advanced Usage Examples")
    print("=" * 60)
    
    # Custom format example
    example_custom_format()
    
    # Performance monitoring
    example_performance_monitoring()
    
    # Multi-file comparison workflow
    workflow = CustomAnalysisWorkflow()
    
    # Create some sample files for comparison
    sample_files = ["example.spe"]
    if Path("sample_data").exists():
        sample_files.extend([str(f) for f in Path("sample_data").glob("*.spe")])
    
    if len(sample_files) > 1:
        print("\n=== Multi-File Comparison ===")
        workflow.multi_file_comparison(sample_files[:3])  # Compare up to 3 files
    
    print("\nAdvanced examples completed!")

if __name__ == "__main__":
    main_advanced()

# examples/migration_helper.py
"""
Migration helper for transitioning from old monolithic code
"""

class LegacyConverter:
    """Compatibility wrapper for old-style usage patterns"""
    
    def __init__(self):
        from spectrum_converter import (
            SpectrumFileReader, ConversionEngine, 
            SpectrumAnalyzer, SpectrumPlotter
        )
        
        self.file_reader = SpectrumFileReader()
        self.conversion_engine = ConversionEngine()
        self.analyzer = SpectrumAnalyzer()
        self.plotter = SpectrumPlotter()
        
        self.current_spectrum = None
        self.current_analysis = None
    
    def load_file(self, file_path: str) -> bool:
        """Legacy-style file loading"""
        try:
            self.current_spectrum = self.file_reader.read_file(file_path)
            return True
        except Exception as e:
            print(f"Error loading file: {e}")
            return False
    
    def convert_file(self, output_format: str, output_path: str = None) -> bool:
        """Legacy-style conversion"""
        if not self.current_spectrum:
            print("No file loaded")
            return False
        
        try:
            source_path = self.current_spectrum.metadata.source_file
            job = self.conversion_engine.create_job(
                source_file=source_path,
                target_format=output_format,
                output_path=output_path
            )
            
            result = self.conversion_engine.convert_single(job)
            return result.success
        except Exception as e:
            print(f"Conversion error: {e}")
            return False
    
    def analyze_spectrum(self) -> bool:
        """Legacy-style analysis"""
        if not self.current_spectrum:
            print("No file loaded")
            return False
        
        try:
            self.current_analysis = self.analyzer.analyze(self.current_spectrum)
            return True
        except Exception as e:
            print(f"Analysis error: {e}")
            return False
    
    def get_peak_count(self) -> int:
        """Legacy-style peak count access"""
        if self.current_analysis:
            return len(self.current_analysis.peaks)
        return 0
    
    def plot_spectrum(self, output_path: str = "spectrum_plot.png") -> bool:
        """Legacy-style plotting"""
        if not self.current_spectrum:
            print("No file loaded")
            return False
        
        if not self.plotter.is_available():
            print("Plotting not available")
            return False
        
        try:
            if self.current_analysis:
                figure = self.plotter.create_analysis_plot(
                    self.current_analysis, "Spectrum Plot"
                )
            else:
                figure = self.plotter.create_simple_plot(
                    self.current_spectrum, "Spectrum Plot"
                )
            
            if figure:
                self.plotter.save_plot(figure, output_path)
                return True
            return False
        except Exception as e:
            print(f"Plotting error: {e}")
            return False

def migration_example():
    """Example showing migration from old to new patterns"""
    print("=== Migration Example ===")
    
    # Old-style usage (compatibility wrapper)
    print("Using legacy compatibility wrapper:")
    legacy = LegacyConverter()
    
    if legacy.load_file("example.spe"):
        print("  ✓ File loaded")
        
        if legacy.convert_file("txt_full"):
            print("  ✓ File converted")
        
        if legacy.analyze_spectrum():
            peaks = legacy.get_peak_count()
            print(f"  ✓ Analysis complete: {peaks} peaks found")
        
        if legacy.plot_spectrum("legacy_plot.png"):
            print("  ✓ Plot created")
    
    # New-style usage (recommended)
    print("\nUsing new modular approach:")
    try:
        from spectrum_converter import (
            SpectrumFileReader, ConversionEngine, 
            SpectrumAnalyzer, SpectrumPlotter
        )
        
        # Load file
        reader = SpectrumFileReader()
        spectrum = reader.read_file("example.spe")
        print("  ✓ File loaded with reader")
        
        # Convert file
        engine = ConversionEngine()
        job = engine.create_job("example.spe", "txt_full")
        result = engine.convert_single(job)
        if result.success:
            print("  ✓ File converted with engine")
        
        # Analyze spectrum
        analyzer = SpectrumAnalyzer()
        analysis = analyzer.analyze(spectrum)
        print(f"  ✓ Analysis complete: {len(analysis.peaks)} peaks found")
        
        # Create plot
        plotter = SpectrumPlotter()
        if plotter.is_available():
            figure = plotter.create_analysis_plot(analysis, "Modern Plot")
            if figure:
                plotter.save_plot(figure, "modern_plot.png")
                print("  ✓ Plot created with plotter")
        
    except Exception as e:
        print(f"  Error in new approach: {e}")

if __name__ == "__main__":
    migration_example()
