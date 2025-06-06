# SpectrumFileConverterAnalyser
Conversion from SPE-->TXT, CNF-->TXT, SPE-->Z1D . Analyse SPE files for spectrum analysis and spectrum plotting 
# Spectrum Converter Pro - API Documentation

## Overview

Spectrum Converter Pro is a comprehensive Python library for gamma-ray spectrum analysis, conversion, and visualization. It provides a clean, modular architecture with separate engines for different tasks.

## Architecture

The application is built with a modular architecture consisting of five main components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │ Conversion      │    │   Analysis      │
│                 │    │   Engine        │    │   Engine        │
│ SpectrumData    │◄───┤                 │    │                 │
│ FileReader      │    │ OutputFormats   │    │ PeakFinder      │
│ FormatHandlers  │    │ BatchManager    │    │ Statistics      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                  │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Plotting Engine │    │   GUI Layer     │    │   CLI Layer     │
│                 │    │                 │    │                 │
│ PlotOptions     │◄───┤ MainApplication │    │ ArgumentParser  │
│ SpectrumPlotter │    │ TabInterfaces   │    │ CommandHandlers │
│ Visualization   │    │ ProgressDialogs │    │ OutputFormatters│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

### Installation

```bash
# Basic installation
pip install spectrum-converter-pro

# With plotting support
pip install spectrum-converter-pro[plotting]

# Development installation
git clone https://github.com/spectrum-analysis/spectrum-converter-pro.git
cd spectrum-converter-pro
pip install -e ".[dev,plotting]"
```

### Basic Usage

```python
from spectrum_converter import (
    SpectrumFileReader, ConversionEngine, 
    SpectrumAnalyzer, SpectrumPlotter
)

# Read a spectrum file
reader = SpectrumFileReader()
spectrum = reader.read_file("example.spe")

# Convert to different format
engine = ConversionEngine()
job = engine.create_job("example.spe", "txt_full")
result = engine.convert_single(job)

# Analyze spectrum
analyzer = SpectrumAnalyzer()
analysis = analyzer.analyze(spectrum)
print(f"Found {len(analysis.peaks)} peaks")

# Create plot
plotter = SpectrumPlotter()
figure = plotter.create_simple_plot(spectrum)
```

## Core Data Models

### SpectrumData

Immutable container for spectrum data with validation and computed properties.

```python
@dataclass(frozen=True)
class SpectrumData:
    channels: Tuple[int, ...]
    counts: Tuple[int, ...]
    metadata: SpectrumMetadata
    
    @property
    def total_counts(self) -> int
    @property
    def peak_channel(self) -> int
    @property
    def peak_counts(self) -> int
    @property
    def count_rate(self) -> Optional[float]
```

**Example:**
```python
metadata = SpectrumMetadata(spec_id="TEST001", live_time=100.0)
spectrum = SpectrumData((0, 1, 2), (10, 20, 15), metadata)

print(f"Total counts: {spectrum.total_counts}")  # 45
print(f"Peak channel: {spectrum.peak_channel}")  # 1
print(f"Count rate: {spectrum.count_rate}")      # 0.45 cps
```

### SpectrumMetadata

Contains measurement information and calibration data.

```python
@dataclass(frozen=True)
class SpectrumMetadata:
    spec_id: Optional[str] = None
    date: Optional[str] = None
    live_time: Optional[float] = None
    real_time: Optional[float] = None
    device: Optional[str] = None
    calibration_params: Optional[Tuple[float, ...]] = None
    
    @property
    def dead_time_percent(self) -> Optional[float]
    @property
    def has_energy_calibration(self) -> bool
    
    def calculate_energy(self, channel: int) -> Optional[float]
```

**Example:**
```python
metadata = SpectrumMetadata(
    spec_id="SAMPLE001",
    live_time=100.0,
    real_time=105.0,
    calibration_params=(0.5, 1.0, 0.001)  # a + b*ch + c*ch²
)

energy = metadata.calculate_energy(100)  # 110.5 keV
dead_time = metadata.dead_time_percent   # 4.76%
```

## File I/O Operations

### SpectrumFileReader

Main interface for reading spectrum files with automatic format detection.

```python
class SpectrumFileReader:
    def read_file(self, file_path: Union[str, Path]) -> SpectrumData
    def can_read_file(self, file_path: Union[str, Path]) -> bool
    def get_supported_formats(self) -> Dict[str, List[str]]
    def detect_format(self, file_path: Union[str, Path]) -> Optional[FileFormatHandler]
```

**Example:**
```python
reader = SpectrumFileReader()

# Check if file is supported
if reader.can_read_file("data.spe"):
    spectrum = reader.read_file("data.spe")
    print(f"Loaded {spectrum.channel_count} channels")

# Get supported formats
formats = reader.get_supported_formats()
for name, extensions in formats.items():
    print(f"{name}: {', '.join(extensions)}")
```

### Custom Format Handlers

Extend support for new file formats by implementing `FileFormatHandler`.

```python
class CustomFormatHandler(FileFormatHandler):
    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.custom'
    
    def parse(self, file_path: Path) -> SpectrumData:
        # Implementation for custom format
        pass
    
    def get_supported_extensions(self) -> List[str]:
        return ['.custom']
    
    def get_format_name(self) -> str:
        return "Custom Format"

# Register custom handler
reader = SpectrumFileReader()
reader.add_handler(CustomFormatHandler())
```

## Format Conversion

### ConversionEngine

Handles file format conversion with job management and progress reporting.

```python
class ConversionEngine:
    def create_job(self, source_file: Union[str, Path], 
                   target_format: str, **parameters) -> ConversionJob
    def convert_single(self, job: ConversionJob, 
                      progress_callback: Optional[Callable] = None) -> ConversionResult
    def get_available_formats(self) -> Dict[str, str]
```

**Example:**
```python
engine = ConversionEngine()

# Single file conversion
job = engine.create_job("input.spe", "txt_full", include_energy=True)

def progress_callback(progress: float, message: str):
    print(f"{progress:.1f}% - {message}")

result = engine.convert_single(job, progress_callback)

if result.success:
    print(f"Converted to: {result.output_file}")
else:
    print(f"Conversion failed: {result.error_message}")
```

### Output Formats

Built-in output formats with extensible architecture:

- **TXT Full**: Complete metadata and analysis
- **TXT Simple**: Count data only
- **Z1D Binary**: Binary format for analysis software

```python
# Custom output format
class CSVFormat(OutputFormat):
    def get_name(self) -> str:
        return "CSV Format"
    
    def get_extension(self) -> str:
        return ".csv"
    
    def convert(self, spectrum: SpectrumData, output_path: Path, 
                progress: ProgressCallback, **params) -> None:
        with open(output_path, 'w') as f:
            f.write("Channel,Counts\n")
            for ch, count in zip(spectrum.channels, spectrum.counts):
                f.write(f"{ch},{count}\n")

# Register custom format
engine.add_format("csv", CSVFormat())
```

### Batch Processing

Process multiple files with parallel execution and organized output.

```python
batch_manager = BatchConversionManager(engine, max_workers=4)

# Find compatible files
files = batch_manager.find_convertible_files(Path("./data"), "txt_full")

# Create batch jobs
jobs = batch_manager.create_batch_jobs(
    files=files,
    target_format="txt_full",
    create_subfolders=True,
    include_statistics=True
)

# Execute batch conversion
results = batch_manager.convert_batch(jobs, progress_callback)

successful = sum(1 for r in results if r.success)
print(f"Converted {successful}/{len(results)} files")
```

## Spectrum Analysis

### SpectrumAnalyzer

Comprehensive analysis including peak finding and statistics calculation.

```python
class SpectrumAnalyzer:
    def analyze(self, spectrum: SpectrumData, 
               peak_method: PeakSearchMethod = PeakSearchMethod.LOCAL_MAXIMUM,
               **analysis_params) -> AnalysisResult
```

**Example:**
```python
analyzer = SpectrumAnalyzer()

# Basic analysis
analysis = analyzer.analyze(spectrum)

print(f"Total counts: {analysis.statistics.total_counts}")
print(f"Signal/Noise: {analysis.statistics.signal_to_noise:.2f}")
print(f"Peaks found: {len(analysis.peaks)}")

# Advanced analysis with parameters
analysis = analyzer.analyze(
    spectrum,
    peak_method=PeakSearchMethod.LOCAL_MAXIMUM,
    peak_min_height=50,
    peak_min_distance=5,
    background_percentile=10
)

# Export detailed report
analyzer.export_analysis_report(analysis, "analysis_report.txt")
```

### Peak Detection

Multiple algorithms available with configurable parameters:

```python
# Local maximum method
peaks = peak_finder.find_peaks_local_maximum(
    spectrum, 
    min_height=10,
    min_distance=2,
    background=background_profile
)

# Derivative method
peaks = peak_finder.find_peaks_derivative(
    spectrum,
    threshold=0.1
)

# Access peak properties
for peak in peaks[:5]:  # Top 5 peaks
    print(f"Channel {peak.channel}: {peak.counts} counts")
    if peak.energy:
        print(f"  Energy: {peak.energy:.2f} keV")
    if peak.fwhm:
        print(f"  FWHM: {peak.fwhm:.2f}")
```

### Background Estimation

Several background estimation methods:

```python
background_estimator = BackgroundEstimator()

# Percentile method
bg_percentile = background_estimator.percentile_method(spectrum, 10)

# Linear interpolation between regions
regions = [(10, 50), (200, 250), (450, 500)]
bg_linear = background_estimator.linear_interpolation(spectrum, regions)

# Moving minimum window
bg_moving = background_estimator.moving_minimum(spectrum, window_size=50)
```

## Plotting and Visualization

### SpectrumPlotter

High-quality plotting with extensive customization options.

```python
class SpectrumPlotter:
    def create_simple_plot(self, spectrum: SpectrumData, title: str) -> Figure
    def create_analysis_plot(self, analysis: AnalysisResult, title: str) -> Figure
    def create_energy_plot(self, spectrum: SpectrumData, title: str) -> Figure
    def create_log_plot(self, spectrum: SpectrumData, title: str) -> Figure
    def create_comparison_plot(self, spectra: List[SpectrumData], 
                              labels: List[str], title: str) -> Figure
```

**Example:**
```python
plotter = SpectrumPlotter()

# Simple linear plot
figure = plotter.create_simple_plot(spectrum, "Sample Spectrum")

# Analysis plot with peaks and background
figure = plotter.create_analysis_plot(analysis, "Peak Analysis")

# Energy-calibrated plot
if spectrum.metadata.has_energy_calibration:
    figure = plotter.create_energy_plot(spectrum, "Energy Spectrum")

# Custom plot with full options
options = PlotOptions(
    title="Custom Plot",
    use_energy_axis=True,
    y_scale="log",
    show_peaks=True,
    show_background=True,
    show_grid=True,
    figure_size=(12, 8),
    dpi=300
)

figure = plotter.create_custom_plot(spectrum, analysis, options)

# Save high-resolution plot
plotter.save_plot(figure, "spectrum_plot.png", dpi=300)
```

### Plot Customization

Extensive customization through `PlotOptions`:

```python
options = PlotOptions(
    # Basic settings
    title="My Spectrum",
    x_label="Energy (keV)",
    y_label="Counts",
    
    # Scale and range
    x_scale="linear",       # linear, log
    y_scale="log",          # linear, log
    x_range=(0, 1000),      # Energy range
    y_range=(1, 10000),     # Count range
    
    # Visual settings
    line_color="blue",
    line_width=1.5,
    show_grid=True,
    grid_alpha=0.3,
    
    # Peak marking
    show_peaks=True,
    peak_marker="v",
    peak_color="red",
    peak_size=8.0,
    
    # Background
    show_background=True,
    background_color="orange",
    background_alpha=0.3,
    
    # Energy axis
    use_energy_axis=True,
    energy_unit="keV",
    
    # Figure settings
    figure_size=(10, 6),
    dpi=100,
    tight_layout=True
)
```

## Command Line Interface

### Basic Commands

```bash
# Convert single file
spectrum-cli convert input.spe --format txt_full --output result.txt

# Batch conversion
spectrum-cli batch /path/to/spectra --format z1d --workers 8

# Analyze spectrum
spectrum-cli analyze input.spe --peaks --export analysis.txt

# Create plot
spectrum-cli plot input.spe --output plot.png --energy-axis --log-scale

# Get file information
spectrum-cli info input.spe --stats --json
```

### Advanced Usage

```bash
# Conversion with custom parameters
spectrum-cli convert input.spe \
    --format z1d \
    --array-size 32768 \
    --output custom.z1d \
    --force

# Batch conversion with filtering
spectrum-cli batch /data/spectra \
    --format txt_full \
    --filter "*.spe" \
    --output-dir /output \
    --workers 16 \
    --no-subfolders

# Analysis with custom parameters
spectrum-cli analyze spectrum.spe \
    --peaks \
    --peak-method local_maximum \
    --min-height 50 \
    --min-distance 5 \
    --background-percentile 5 \
    --export detailed_analysis.txt \
    --json > analysis.json

# High-quality plot generation
spectrum-cli plot spectrum.spe \
    --output publication_plot.pdf \
    --format pdf \
    --title "Sample Analysis" \
    --energy-axis \
    --show-peaks \
    --dpi 600 \
    --size 12 8
```

## Error Handling

The library uses specific exception types for different error conditions:

```python
try:
    spectrum = reader.read_file("data.spe")
except FileNotFoundError:
    print("File not found")
except FileFormatError as e:
    print(f"Invalid file format: {e}")
except SecurityError as e:
    print(f"Security validation failed: {e}")
except SpectrumError as e:
    print(f"General spectrum processing error: {e}")
```

### Exception Hierarchy

```
SpectrumError
├── FileFormatError
├── ConversionError
└── SecurityError
```

## Configuration Management

Application settings are managed through a configuration system:

```python
from main_gui_application import ConfigManager

config = ConfigManager()

# Get configuration values
max_workers = config.get("batch_max_workers", 4)
array_size = config.get("z1d_array_size", 16384)
plot_dpi = config.get("plotting.dpi", 100)

# Set configuration values
config.set("batch_max_workers", 8)
config.set("plotting.default_style", "line")

# Add recent file
config.add_recent_file("/path/to/spectrum.spe")

# Save configuration
config.save_config()
```

## Performance Considerations

### Memory Management

- Use streaming for large files
- Immutable data structures prevent accidental modification
- Automatic cleanup of temporary files

### Parallel Processing

```python
# Batch conversion with controlled parallelism
batch_manager = BatchConversionManager(engine, max_workers=cpu_count())

# Monitor memory usage during large batch operations
import psutil
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
```

### Optimization Tips

1. **Batch Processing**: Use batch operations for multiple files
2. **Array Size**: Choose appropriate Z1D array sizes (powers of 2)
3. **Peak Finding**: Adjust parameters to balance accuracy vs speed
4. **Plotting**: Use appropriate DPI for intended output

## Testing

The library includes comprehensive tests:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=spectrum_converter --cov-report=html

# Run specific test categories
pytest tests/ -m "not slow"          # Skip slow tests
pytest tests/ -k "test_conversion"   # Only conversion tests
pytest tests/ -x                     # Stop on first failure
```

### Writing Tests

```python
import pytest
from spectrum_converter import SpectrumData, SpectrumMetadata

def test_spectrum_creation():
    metadata = SpectrumMetadata(spec_id="TEST")
    spectrum = SpectrumData((0, 1, 2), (10, 20, 15), metadata)
    
    assert spectrum.total_counts == 45
    assert spectrum.peak_channel == 1

@pytest.fixture
def sample_spectrum():
    metadata = SpectrumMetadata(spec_id="SAMPLE")
    return SpectrumData((0, 1, 2), (5, 10, 8), metadata)

def test_analysis(sample_spectrum):
    # Test using fixture
    assert sample_spectrum.peak_counts == 10
```

## Deployment

### Docker Deployment

```bash
# Build image
docker build -t spectrum-converter-pro .

# Run GUI application
docker run -it --rm \
    -v $(pwd)/data:/app/data \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    spectrum-converter-pro

# Run CLI commands
docker run -it --rm \
    -v $(pwd)/data:/app/data \
    spectrum-converter-pro \
    spectrum-cli convert /app/data/sample.spe --format txt_full
```

### Production Deployment

```bash
# Install production version
pip install spectrum-converter-pro[plotting]

# Set up system service (systemd example)
sudo cp spectrum-converter.service /etc/systemd/system/
sudo systemctl enable spectrum-converter
sudo systemctl start spectrum-converter
```

## API Reference Summary

### Core Classes

| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `SpectrumData` | Immutable spectrum container | `total_counts`, `peak_channel`, `get_energy_spectrum()` |
| `SpectrumMetadata` | Measurement metadata | `calculate_energy()`, `dead_time_percent` |
| `SpectrumFileReader` | File I/O interface | `read_file()`, `can_read_file()`, `get_supported_formats()` |
| `ConversionEngine` | Format conversion | `create_job()`, `convert_single()`, `get_available_formats()` |
| `SpectrumAnalyzer` | Analysis and statistics | `analyze()`, `export_analysis_report()` |
| `SpectrumPlotter` | Plotting and visualization | `create_simple_plot()`, `create_analysis_plot()`, `save_plot()` |

### Data Types

| Type | Description |
|------|-------------|
| `FileFormat` | Enum of supported file formats |
| `ConversionJob` | Single conversion task definition |
| `ConversionResult` | Conversion operation result |
| `Peak` | Detected peak information |
| `SpectrumStatistics` | Comprehensive spectrum statistics |
| `PlotOptions` | Plot customization settings |

### Exception Types

| Exception | When Raised |
|-----------|-------------|
| `SpectrumError` | Base exception for all spectrum operations |
| `FileFormatError` | Invalid or unsupported file format |
| `ConversionError` | Conversion process failure |
| `SecurityError` | Security validation failure |

## Migration Guide

### From Original Version

The refactored version provides backward-compatible functionality with improved architecture:

```python
# Old approach (monolithic)
# converter = UnifiedSpectrumConverter(root)
# converter.convert_file()

# New approach (modular)
reader = SpectrumFileReader()
engine = ConversionEngine()

spectrum = reader.read_file("input.spe")
job = engine.create_job("input.spe", "txt_full")
result = engine.convert_single(job)
```

### Key Improvements

1. **Separation of Concerns**: Each engine handles specific functionality
2. **Type Safety**: Comprehensive type hints and validation
3. **Error Handling**: Specific exception types with detailed messages
4. **Testing**: Full test coverage with fixtures and parameterized tests
5. **Configuration**: Externalized settings with persistence
6. **Security**: Input validation and path sanitization
7. **Performance**: Parallel processing and efficient memory usage

## Support and Contributing

- **Documentation**: https://spectrum-converter-pro.readthedocs.io
- **Issues**: https://github.com/spectrum-analysis/spectrum-converter-pro/issues
- **Discussions**: https://github.com/spectrum-analysis/spectrum-converter-pro/discussions
- **Contributing**: See CONTRIBUTING.md for development guidelines

## License

MIT License - see LICENSE file for details.
