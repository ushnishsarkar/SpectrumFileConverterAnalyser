"""
Command Line Interface for Spectrum Converter Pro
Provides automation-friendly CLI for scripting and batch operations
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from spectrum_data_models import SpectrumFileReader, FileFormatError
from conversion_engine import ConversionEngine, BatchConversionManager
from analysis_engine import SpectrumAnalyzer, PeakSearchMethod
from plotting_engine import SpectrumPlotter, PlotOptions, PlotType

class CLIProgressReporter:
    """Simple progress reporter for CLI operations"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.last_progress = -1
    
    def report(self, progress: float, message: str = ""):
        """Report progress with simple text output"""
        if not self.verbose:
            return
        
        # Only update on significant progress changes
        current_progress = int(progress)
        if current_progress != self.last_progress and current_progress % 10 == 0:
            print(f"Progress: {current_progress}% - {message}")
            self.last_progress = current_progress

class SpectrumConverterCLI:
    """Main CLI application class"""
    
    def __init__(self):
        self.file_reader = SpectrumFileReader()
        self.conversion_engine = ConversionEngine()
        self.analyzer = SpectrumAnalyzer()
        self.plotter = SpectrumPlotter()
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser"""
        parser = argparse.ArgumentParser(
            prog="spectrum-cli",
            description="Spectrum Converter Pro - Command Line Interface",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Convert single file
  spectrum-cli convert input.spe --format txt_full --output result.txt
  
  # Batch conversion
  spectrum-cli batch /path/to/spectra --format z1d --workers 8
  
  # Analyze spectrum
  spectrum-cli analyze input.spe --peaks --export analysis.txt
  
  # Create plot
  spectrum-cli plot input.spe --output plot.png --energy-axis
  
  # Get file information
  spectrum-cli info input.spe --json
            """
        )
        
        parser.add_argument("--verbose", "-v", action="store_true",
                          help="Enable verbose output")
        parser.add_argument("--version", action="version", version="2.0.0")
        
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # Convert command
        self._add_convert_parser(subparsers)
        
        # Batch command
        self._add_batch_parser(subparsers)
        
        # Analyze command
        self._add_analyze_parser(subparsers)
        
        # Plot command
        self._add_plot_parser(subparsers)
        
        # Info command
        self._add_info_parser(subparsers)
        
        # List-formats command
        self._add_list_formats_parser(subparsers)
        
        return parser
    
    def _add_convert_parser(self, subparsers):
        """Add convert command parser"""
        parser = subparsers.add_parser("convert", help="Convert single spectrum file")
        parser.add_argument("input", help="Input spectrum file")
        parser.add_argument("--format", "-f", required=True,
                          choices=["txt_full", "txt_simple", "z1d"],
                          help="Output format")
        parser.add_argument("--output", "-o", help="Output file path")
        parser.add_argument("--array-size", type=int, default=16384,
                          help="Array size for Z1D format (default: 16384)")
        parser.add_argument("--force", action="store_true",
                          help="Overwrite existing output file")
    
    def _add_batch_parser(self, subparsers):
        """Add batch command parser"""
        parser = subparsers.add_parser("batch", help="Batch convert multiple files")
        parser.add_argument("input_dir", help="Directory containing spectrum files")
        parser.add_argument("--format", "-f", required=True,
                          choices=["txt_full", "txt_simple", "z1d"],
                          help="Output format")
        parser.add_argument("--output-dir", "-o", help="Output directory")
        parser.add_argument("--subfolders", action="store_true", default=True,
                          help="Create organized subfolders (default: True)")
        parser.add_argument("--no-subfolders", dest="subfolders", action="store_false",
                          help="Don't create subfolders")
        parser.add_argument("--workers", "-w", type=int, default=4,
                          help="Number of parallel workers (default: 4)")
        parser.add_argument("--array-size", type=int, default=16384,
                          help="Array size for Z1D format (default: 16384)")
        parser.add_argument("--filter", help="File pattern filter (e.g., '*.spe')")
    
    def _add_analyze_parser(self, subparsers):
        """Add analyze command parser"""
        parser = subparsers.add_parser("analyze", help="Analyze spectrum")
        parser.add_argument("input", help="Input spectrum file")
        parser.add_argument("--peaks", action="store_true",
                          help="Find peaks in spectrum")
        parser.add_argument("--peak-method", choices=["local_maximum", "derivative"],
                          default="local_maximum", help="Peak finding method")
        parser.add_argument("--min-height", type=float, default=10,
                          help="Minimum peak height (default: 10)")
        parser.add_argument("--min-distance", type=int, default=1,
                          help="Minimum peak distance (default: 1)")
        parser.add_argument("--background-percentile", type=float, default=10,
                          help="Background percentile (default: 10)")
        parser.add_argument("--export", "-e", help="Export analysis report to file")
        parser.add_argument("--json", action="store_true",
                          help="Output results in JSON format")
    
    def _add_plot_parser(self, subparsers):
        """Add plot command parser"""
        parser = subparsers.add_parser("plot", help="Create spectrum plot")
        parser.add_argument("input", help="Input spectrum file")
        parser.add_argument("--output", "-o", required=True,
                          help="Output plot file")
        parser.add_argument("--format", choices=["png", "pdf", "svg"],
                          default="png", help="Plot format (default: png)")
        parser.add_argument("--title", help="Plot title")
        parser.add_argument("--energy-axis", action="store_true",
                          help="Use energy axis (requires calibration)")
        parser.add_argument("--log-scale", action="store_true",
                          help="Use logarithmic y-axis")
        parser.add_argument("--show-peaks", action="store_true",
                          help="Mark peaks on plot")
        parser.add_argument("--show-grid", action="store_true", default=True,
                          help="Show grid (default: True)")
        parser.add_argument("--no-grid", dest="show_grid", action="store_false",
                          help="Don't show grid")
        parser.add_argument("--dpi", type=int, default=300,
                          help="Plot resolution (default: 300)")
        parser.add_argument("--size", nargs=2, type=float, default=[10, 6],
                          metavar=("WIDTH", "HEIGHT"),
                          help="Figure size in inches (default: 10 6)")
    
    def _add_info_parser(self, subparsers):
        """Add info command parser"""
        parser = subparsers.add_parser("info", help="Display file information")
        parser.add_argument("input", help="Input spectrum file")
        parser.add_argument("--json", action="store_true",
                          help="Output in JSON format")
        parser.add_argument("--stats", action="store_true",
                          help="Include detailed statistics")
    
    def _add_list_formats_parser(self, subparsers):
        """Add list-formats command parser"""
        parser = subparsers.add_parser("list-formats", 
                                     help="List supported file formats")
        parser.add_argument("--json", action="store_true",
                          help="Output in JSON format")
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """Run CLI with given arguments"""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)
        
        if not parsed_args.command:
            parser.print_help()
            return 1
        
        try:
            # Dispatch to appropriate handler
            if parsed_args.command == "convert":
                return self._handle_convert(parsed_args)
            elif parsed_args.command == "batch":
                return self._handle_batch(parsed_args)
            elif parsed_args.command == "analyze":
                return self._handle_analyze(parsed_args)
            elif parsed_args.command == "plot":
                return self._handle_plot(parsed_args)
            elif parsed_args.command == "info":
                return self._handle_info(parsed_args)
            elif parsed_args.command == "list-formats":
                return self._handle_list_formats(parsed_args)
            else:
                print(f"Unknown command: {parsed_args.command}", file=sys.stderr)
                return 1
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user", file=sys.stderr)
            return 130
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            if parsed_args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def _handle_convert(self, args) -> int:
        """Handle convert command"""
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Input file not found: {input_path}", file=sys.stderr)
            return 1
        
        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            format_handler = self.conversion_engine.output_formats[args.format]
            output_path = input_path.with_suffix(format_handler.get_extension())
        
        # Check if output exists
        if output_path.exists() and not args.force:
            print(f"Output file exists: {output_path}", file=sys.stderr)
            print("Use --force to overwrite", file=sys.stderr)
            return 1
        
        # Create conversion job
        job = self.conversion_engine.create_job(
            source_file=input_path,
            target_format=args.format,
            output_path=output_path,
            array_size=args.array_size
        )
        
        # Set up progress reporting
        progress_reporter = CLIProgressReporter(args.verbose)
        
        if args.verbose:
            print(f"Converting {input_path} to {args.format} format...")
        
        # Perform conversion
        result = self.conversion_engine.convert_single(job, progress_reporter.report)
        
        if result.success:
            print(f"Conversion completed: {result.output_file}")
            if args.verbose:
                print(f"Processing time: {result.processing_time:.2f} seconds")
            return 0
        else:
            print(f"Conversion failed: {result.error_message}", file=sys.stderr)
            return 1
    
    def _handle_batch(self, args) -> int:
        """Handle batch command"""
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            print(f"Input directory not found: {input_dir}", file=sys.stderr)
            return 1
        
        # Create batch manager
        batch_manager = BatchConversionManager(
            self.conversion_engine,
            max_workers=args.workers
        )
        
        # Find files
        if args.verbose:
            print(f"Scanning directory: {input_dir}")
        
        files = batch_manager.find_convertible_files(input_dir, args.format)
        
        # Apply filter if specified
        if args.filter:
            import fnmatch
            files = [f for f in files if fnmatch.fnmatch(f.name, args.filter)]
        
        if not files:
            print("No compatible files found", file=sys.stderr)
            return 1
        
        if args.verbose:
            print(f"Found {len(files)} compatible files")
        
        # Create jobs
        output_dir = Path(args.output_dir) if args.output_dir else input_dir
        jobs = batch_manager.create_batch_jobs(
            files=files,
            target_format=args.format,
            output_dir=output_dir,
            create_subfolders=args.subfolders,
            array_size=args.array_size
        )
        
        # Set up progress reporting
        progress_reporter = CLIProgressReporter(args.verbose)
        
        if args.verbose:
            print(f"Starting batch conversion with {args.workers} workers...")
        
        # Perform batch conversion
        results = batch_manager.convert_batch(jobs, progress_reporter.report)
        
        # Report results
        successful = sum(1 for r in results if r.success)
        total = len(results)
        
        print(f"Batch conversion completed: {successful}/{total} files")
        
        if successful < total:
            failed_files = [r.job.source_file.name for r in results if not r.success]
            print(f"Failed files: {', '.join(failed_files)}", file=sys.stderr)
            return 1
        
        return 0
    
    def _handle_analyze(self, args) -> int:
        """Handle analyze command"""
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Input file not found: {input_path}", file=sys.stderr)
            return 1
        
        # Load spectrum
        if args.verbose:
            print(f"Loading spectrum: {input_path}")
        
        try:
            spectrum = self.file_reader.read_file(input_path)
        except FileFormatError as e:
            print(f"Failed to read spectrum: {e}", file=sys.stderr)
            return 1
        
        # Perform analysis
        analysis_params = {
            'peak_min_height': args.min_height,
            'peak_min_distance': args.min_distance,
            'background_percentile': args.background_percentile
        }
        
        method = PeakSearchMethod(args.peak_method)
        analysis = self.analyzer.analyze(spectrum, method, **analysis_params)
        
        # Output results
        if args.json:
            self._output_analysis_json(analysis)
        else:
            self._output_analysis_text(analysis, args.verbose)
        
        # Export report if requested
        if args.export:
            export_path = Path(args.export)
            self.analyzer.export_analysis_report(analysis, export_path)
            if args.verbose:
                print(f"Analysis report exported to: {export_path}")
        
        return 0
    
    def _handle_plot(self, args) -> int:
        """Handle plot command"""
        if not self.plotter.is_available():
            print("Plotting not available - matplotlib not installed", file=sys.stderr)
            return 1
        
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Input file not found: {input_path}", file=sys.stderr)
            return 1
        
        # Load spectrum
        if args.verbose:
            print(f"Loading spectrum: {input_path}")
        
        try:
            spectrum = self.file_reader.read_file(input_path)
        except FileFormatError as e:
            print(f"Failed to read spectrum: {e}", file=sys.stderr)
            return 1
        
        # Create plot options
        title = args.title or f"Spectrum: {input_path.name}"
        
        options = PlotOptions(
            title=title,
            use_energy_axis=args.energy_axis,
            y_scale="log" if args.log_scale else "linear",
            show_peaks=args.show_peaks,
            show_grid=args.show_grid,
            figure_size=tuple(args.size),
            dpi=args.dpi
        )
        
        # Create plot
        if args.verbose:
            print("Creating plot...")
        
        figure = self.plotter.create_custom_plot(spectrum, options=options)
        
        if not figure:
            print("Failed to create plot", file=sys.stderr)
            return 1
        
        # Save plot
        output_path = Path(args.output)
        self.plotter.save_plot(figure, output_path, format=args.format, dpi=args.dpi)
        
        print(f"Plot saved: {output_path}")
        return 0
    
    def _handle_info(self, args) -> int:
        """Handle info command"""
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Input file not found: {input_path}", file=sys.stderr)
            return 1
        
        # Load spectrum
        try:
            spectrum = self.file_reader.read_file(input_path)
        except FileFormatError as e:
            print(f"Failed to read spectrum: {e}", file=sys.stderr)
            return 1
        
        # Get analysis if requested
        analysis = None
        if args.stats:
            analysis = self.analyzer.analyze(spectrum)
        
        # Output information
        if args.json:
            self._output_info_json(spectrum, analysis)
        else:
            self._output_info_text(spectrum, analysis)
        
        return 0
    
    def _handle_list_formats(self, args) -> int:
        """Handle list-formats command"""
        input_formats = self.file_reader.get_supported_formats()
        output_formats = self.conversion_engine.get_available_formats()
        
        if args.json:
            data = {
                "input_formats": input_formats,
                "output_formats": output_formats
            }
            print(json.dumps(data, indent=2))
        else:
            print("SUPPORTED FILE FORMATS")
            print("=" * 40)
            print("\nInput Formats:")
            for name, extensions in input_formats.items():
                print(f"  {name}: {', '.join(extensions)}")
            
            print("\nOutput Formats:")
            for key, name in output_formats.items():
                print(f"  {key}: {name}")
        
        return 0
    
    def _output_analysis_json(self, analysis):
        """Output analysis results in JSON format"""
        data = {
            "file_info": {
                "source_file": str(analysis.spectrum.metadata.source_file),
                "format": analysis.spectrum.metadata.file_format.value if analysis.spectrum.metadata.file_format else None,
                "channels": analysis.spectrum.channel_count,
                "total_counts": analysis.spectrum.total_counts
            },
            "statistics": {
                "peak_channel": analysis.statistics.peak_channel,
                "peak_counts": analysis.statistics.peak_counts,
                "mean_channel": analysis.statistics.mean_channel,
                "std_deviation": analysis.statistics.std_deviation,
                "background_level": analysis.statistics.background_level,
                "noise_level": analysis.statistics.noise_level,
                "signal_to_noise": analysis.statistics.signal_to_noise,
                "count_rate": analysis.statistics.count_rate
            },
            "peaks": [
                {
                    "channel": peak.channel,
                    "counts": peak.counts,
                    "energy": peak.energy,
                    "fwhm": peak.fwhm,
                    "area": peak.area,
                    "significance": peak.significance
                }
                for peak in analysis.peaks
            ]
        }
        print(json.dumps(data, indent=2))
    
    def _output_analysis_text(self, analysis, verbose: bool):
        """Output analysis results in text format"""
        spectrum = analysis.spectrum
        stats = analysis.statistics
        
        print("SPECTRUM ANALYSIS")
        print("=" * 40)
        print(f"File: {spectrum.metadata.source_file.name if spectrum.metadata.source_file else 'Unknown'}")
        print(f"Total Counts: {spectrum.total_counts:,}")
        print(f"Peak Counts: {spectrum.peak_counts:,}")
        print(f"Channels: {spectrum.channel_count:,}")
        
        if verbose:
            print(f"Mean Channel: {stats.mean_channel:.2f}")
            print(f"Std Deviation: {stats.std_deviation:.2f}")
            if stats.count_rate:
                print(f"Count Rate: {stats.count_rate:.2f} cps")
            if stats.signal_to_noise:
                print(f"Signal/Noise: {stats.signal_to_noise:.2f}")
        
        if analysis.peaks:
            print(f"\nPEAKS FOUND: {len(analysis.peaks)}")
            print("-" * 20)
            print("Channel   Counts    Energy    FWHM")
            print("-" * 35)
            for peak in analysis.peaks[:10]:  # Show top 10
                energy_str = f"{peak.energy:.1f}" if peak.energy else "N/A"
                fwhm_str = f"{peak.fwhm:.1f}" if peak.fwhm else "N/A"
                print(f"{peak.channel:7d}  {peak.counts:8,}  {energy_str:>7}  {fwhm_str:>6}")
    
    def _output_info_json(self, spectrum, analysis=None):
        """Output file info in JSON format"""
        meta = spectrum.metadata
        data = {
            "file_info": {
                "source_file": str(meta.source_file) if meta.source_file else None,
                "format": meta.file_format.value if meta.file_format else None,
                "spec_id": meta.spec_id,
                "date": meta.date,
                "device": meta.device,
                "live_time": meta.live_time,
                "real_time": meta.real_time,
                "dead_time_percent": meta.dead_time_percent,
                "has_energy_calibration": meta.has_energy_calibration,
                "calibration_params": list(meta.calibration_params) if meta.calibration_params else None
            },
            "spectrum_data": {
                "total_counts": spectrum.total_counts,
                "peak_counts": spectrum.peak_counts,
                "peak_channel": spectrum.peak_channel,
                "channel_count": spectrum.channel_count,
                "count_rate": spectrum.count_rate
            }
        }
        
        if analysis:
            data["statistics"] = {
                "mean_channel": analysis.statistics.mean_channel,
                "std_deviation": analysis.statistics.std_deviation,
                "background_level": analysis.statistics.background_level,
                "noise_level": analysis.statistics.noise_level,
                "signal_to_noise": analysis.statistics.signal_to_noise
            }
            data["peaks_found"] = len(analysis.peaks)
        
        print(json.dumps(data, indent=2))
    
    def _output_info_text(self, spectrum, analysis=None):
        """Output file info in text format"""
        meta = spectrum.metadata
        
        print("FILE INFORMATION")
        print("=" * 30)
        
        if meta.source_file:
            print(f"File: {meta.source_file.name}")
            print(f"Path: {meta.source_file}")
        
        if meta.file_format:
            print(f"Format: {meta.file_format.value.upper()}")
        
        if meta.spec_id:
            print(f"Spectrum ID: {meta.spec_id}")
        
        if meta.date:
            print(f"Date: {meta.date}")
        
        if meta.device:
            print(f"Device: {meta.device}")
        
        if meta.live_time and meta.real_time:
            print(f"Live Time: {meta.live_time:.1f} s")
            print(f"Real Time: {meta.real_time:.1f} s")
            if meta.dead_time_percent:
                print(f"Dead Time: {meta.dead_time_percent:.2f}%")
        
        print(f"\nSPECTRUM DATA")
        print("-" * 20)
        print(f"Total Counts: {spectrum.total_counts:,}")
        print(f"Peak Counts: {spectrum.peak_counts:,}")
        print(f"Peak Channel: {spectrum.peak_channel}")
        print(f"Channels: {spectrum.channel_count:,}")
        
        if spectrum.count_rate:
            print(f"Count Rate: {spectrum.count_rate:.2f} cps")
        
        if meta.has_energy_calibration:
            print(f"\nENERGY CALIBRATION")
            print("-" * 18)
            params = meta.calibration_params
            if len(params) == 2:
                print(f"E = {params[0]:.6f} + {params[1]:.6f} * Ch")
            elif len(params) >= 3:
                print(f"E = {params[0]:.6f} + {params[1]:.6f} * Ch + {params[2]:.6f} * Ch²")
        
        if analysis:
            stats = analysis.statistics
            print(f"\nSTATISTICS")
            print("-" * 10)
            print(f"Mean Channel: {stats.mean_channel:.2f}")
            print(f"Std Deviation: {stats.std_deviation:.2f}")
            if stats.signal_to_noise:
                print(f"Signal/Noise: {stats.signal_to_noise:.2f}")
            if analysis.peaks:
                print(f"Peaks Found: {len(analysis.peaks)}")

def main():
    """Main CLI entry point"""
    cli = SpectrumConverterCLI()
    sys.exit(cli.run())

if __name__ == "__main__":
    main()
