"""
Spectrum Conversion Engine
Handles all format conversions with proper error handling and progress reporting
"""

import struct
import time
import concurrent.futures
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Union
from enum import Enum

from spectrum_data_models import (
    SpectrumData, SpectrumFileReader, ConversionError, FileFormat
)

class ConversionStatus(Enum):
    """Status of conversion operations"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ConversionJob:
    """Represents a single conversion task"""
    job_id: str
    source_file: Path
    target_format: str
    output_path: Optional[Path] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # Higher number = higher priority
    
    def __post_init__(self):
        if not isinstance(self.source_file, Path):
            self.source_file = Path(self.source_file)

@dataclass
class ConversionResult:
    """Result of a conversion operation"""
    job: ConversionJob
    status: ConversionStatus
    output_file: Optional[Path] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    @property
    def success(self) -> bool:
        return self.status == ConversionStatus.COMPLETED

class ProgressCallback:
    """Thread-safe progress reporting"""
    
    def __init__(self, callback: Optional[Callable[[float, str], None]] = None):
        self.callback = callback
        self._cancelled = False
    
    def report(self, progress: float, message: str = ""):
        """Report progress (0-100)"""
        if self.callback and not self._cancelled:
            self.callback(min(100.0, max(0.0, progress)), message)
    
    def cancel(self):
        """Mark as cancelled"""
        self._cancelled = True
    
    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

class OutputFormat(ABC):
    """Abstract base for output format handlers"""
    
    @abstractmethod
    def get_name(self) -> str:
        """Human-readable format name"""
        pass
    
    @abstractmethod
    def get_extension(self) -> str:
        """File extension including dot"""
        pass
    
    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """Default parameters for this format"""
        pass
    
    @abstractmethod
    def validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize parameters"""
        pass
    
    @abstractmethod
    def convert(self, spectrum: SpectrumData, output_path: Path, 
                progress: ProgressCallback, **params) -> None:
        """Perform the conversion"""
        pass

class TXTFullFormat(OutputFormat):
    """Convert to TXT with full metadata and analysis"""
    
    def get_name(self) -> str:
        return "TXT Full Metadata"
    
    def get_extension(self) -> str:
        return ".txt"
    
    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'include_energy': True,
            'include_statistics': True,
            'include_peaks': False,
            'peak_threshold': 3.0
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        validated = self.get_default_parameters()
        validated.update(params)
        
        # Validate peak threshold
        if validated['peak_threshold'] <= 0:
            validated['peak_threshold'] = 3.0
        
        return validated
    
    def convert(self, spectrum: SpectrumData, output_path: Path, 
                progress: ProgressCallback, **params) -> None:
        """Write spectrum to TXT with full metadata"""
        params = self.validate_parameters(params)
        
        progress.report(10, "Writing file header")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            self._write_header(f, spectrum)
            
            progress.report(25, "Writing metadata")
            self._write_metadata(f, spectrum, params)
            
            progress.report(50, "Writing spectrum data")
            self._write_data(f, spectrum, params)
            
            progress.report(75, "Calculating statistics")
            self._write_statistics(f, spectrum, params)
            
            if params['include_peaks']:
                progress.report(90, "Finding peaks")
                self._write_peaks(f, spectrum, params)
        
        progress.report(100, "Conversion complete")
    
    def _write_header(self, f, spectrum: SpectrumData):
        """Write file header"""
        f.write("GAMMA RAY SPECTRUM DATA CONVERSION\n")
        f.write("=" * 55 + "\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source: {spectrum.metadata.source_file.name if spectrum.metadata.source_file else 'Unknown'}\n")
        f.write(f"Format: {spectrum.metadata.file_format.value if spectrum.metadata.file_format else 'Unknown'}\n\n")
    
    def _write_metadata(self, f, spectrum: SpectrumData, params: Dict[str, Any]):
        """Write measurement metadata"""
        meta = spectrum.metadata
        
        f.write("MEASUREMENT INFORMATION:\n")
        f.write("-" * 25 + "\n")
        
        if meta.spec_id:
            f.write(f"Spectrum ID: {meta.spec_id}\n")
        if meta.date:
            f.write(f"Date/Time: {meta.date}\n")
        if meta.device:
            f.write(f"Device: {meta.device}\n")
        
        if meta.live_time and meta.real_time:
            f.write(f"Live Time: {meta.live_time:.1f} seconds\n")
            f.write(f"Real Time: {meta.real_time:.1f} seconds\n")
            if meta.dead_time_percent:
                f.write(f"Dead Time: {meta.dead_time_percent:.2f}%\n")
        
        if meta.has_energy_calibration and params['include_energy']:
            f.write(f"\nENERGY CALIBRATION:\n")
            f.write("-" * 18 + "\n")
            params_cal = meta.calibration_params
            if len(params_cal) == 2:
                f.write(f"E = {params_cal[0]:.6f} + {params_cal[1]:.6f} * Ch\n")
            elif len(params_cal) >= 3:
                f.write(f"E = {params_cal[0]:.6f} + {params_cal[1]:.6f} * Ch + {params_cal[2]:.6f} * Ch²\n")
        
        f.write("\n")
    
    def _write_data(self, f, spectrum: SpectrumData, params: Dict[str, Any]):
        """Write spectrum data"""
        f.write("SPECTRUM DATA:\n")
        f.write("=" * 15 + "\n")
        
        if params['include_energy'] and spectrum.metadata.has_energy_calibration:
            f.write("Channel\tCounts\tEnergy (keV)\n")
            f.write("-------\t------\t------------\n")
            
            for channel, count in zip(spectrum.channels, spectrum.counts):
                energy = spectrum.metadata.calculate_energy(channel)
                f.write(f"{channel}\t{count}\t{energy:.3f}\n")
        else:
            f.write("Channel\tCounts\n")
            f.write("-------\t------\n")
            
            for channel, count in zip(spectrum.channels, spectrum.counts):
                f.write(f"{channel}\t{count}\n")
        
        f.write("\n")
    
    def _write_statistics(self, f, spectrum: SpectrumData, params: Dict[str, Any]):
        """Write spectrum statistics"""
        if not params['include_statistics']:
            return
        
        f.write("SPECTRUM STATISTICS:\n")
        f.write("-" * 19 + "\n")
        f.write(f"Total Channels: {spectrum.channel_count:,}\n")
        f.write(f"Total Counts: {spectrum.total_counts:,}\n")
        f.write(f"Peak Channel: {spectrum.peak_channel}\n")
        f.write(f"Peak Counts: {spectrum.peak_counts:,}\n")
        
        if spectrum.count_rate:
            f.write(f"Count Rate: {spectrum.count_rate:.2f} cps\n")
        
        # Calculate additional statistics
        non_zero_counts = [c for c in spectrum.counts if c > 0]
        if non_zero_counts:
            avg_count = sum(non_zero_counts) / len(non_zero_counts)
            f.write(f"Average Count (non-zero): {avg_count:.2f}\n")
        
        f.write("\n")
    
    def _write_peaks(self, f, spectrum: SpectrumData, params: Dict[str, Any]):
        """Find and write significant peaks"""
        peaks = self._find_peaks(spectrum, params['peak_threshold'])
        
        if peaks:
            f.write("SIGNIFICANT PEAKS:\n")
            f.write("-" * 17 + "\n")
            f.write("Rank\tChannel\tCounts\tEnergy (keV)\n")
            f.write("----\t-------\t------\t------------\n")
            
            for i, (channel, count) in enumerate(peaks[:10], 1):
                energy_str = "N/A"
                if spectrum.metadata.has_energy_calibration:
                    energy = spectrum.metadata.calculate_energy(channel)
                    if energy:
                        energy_str = f"{energy:.3f}"
                
                f.write(f"{i}\t{channel}\t{count:,}\t{energy_str}\n")
            
            f.write("\n")
    
    def _find_peaks(self, spectrum: SpectrumData, threshold: float) -> List[Tuple[int, int]]:
        """Find significant peaks in spectrum"""
        if len(spectrum.counts) < 5:
            return []
        
        # Calculate background level
        sorted_counts = sorted([c for c in spectrum.counts if c > 0])
        if not sorted_counts:
            return []
        
        background = sorted_counts[len(sorted_counts) // 10]  # 10th percentile
        min_height = max(1, background * threshold)
        
        peaks = []
        counts_dict = dict(zip(spectrum.channels, spectrum.counts))
        
        # Find local maxima above threshold
        for i, channel in enumerate(spectrum.channels[2:-2], 2):
            count = spectrum.counts[i]
            
            if count < min_height:
                continue
            
            # Check if it's a local maximum
            is_peak = True
            for offset in [-2, -1, 1, 2]:
                neighbor_idx = i + offset
                if neighbor_idx >= 0 and neighbor_idx < len(spectrum.counts):
                    if spectrum.counts[neighbor_idx] >= count:
                        is_peak = False
                        break
            
            if is_peak:
                peaks.append((channel, count))
        
        # Sort by count (descending)
        peaks.sort(key=lambda x: x[1], reverse=True)
        return peaks

class TXTSimpleFormat(OutputFormat):
    """Convert to simple single-column TXT (counts only)"""
    
    def get_name(self) -> str:
        return "TXT Simple (Counts Only)"
    
    def get_extension(self) -> str:
        return ".txt"
    
    def get_default_parameters(self) -> Dict[str, Any]:
        return {'start_channel': 0}
    
    def validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        validated = self.get_default_parameters()
        validated.update(params)
        
        if validated['start_channel'] < 0:
            validated['start_channel'] = 0
        
        return validated
    
    def convert(self, spectrum: SpectrumData, output_path: Path, 
                progress: ProgressCallback, **params) -> None:
        """Write spectrum as simple count list"""
        params = self.validate_parameters(params)
        
        progress.report(25, "Writing spectrum data")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, count in enumerate(spectrum.counts):
                f.write(f"{count}\n")
                
                # Report progress periodically
                if i % 1000 == 0:
                    progress_pct = 25 + (i / len(spectrum.counts)) * 75
                    progress.report(progress_pct, f"Writing channel {i}")
        
        progress.report(100, "Simple TXT conversion complete")

class Z1DFormat(OutputFormat):
    """Convert to Z1D binary format"""
    
    def get_name(self) -> str:
        return "Z1D Binary"
    
    def get_extension(self) -> str:
        return ".z1d"
    
    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'array_size': 16384,
            'pad_value': 0,
            'integer_size': 4  # 32-bit integers
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        validated = self.get_default_parameters()
        validated.update(params)
        
        # Validate array size
        if validated['array_size'] <= 0:
            validated['array_size'] = 16384
        
        # Validate integer size
        if validated['integer_size'] not in [2, 4, 8]:
            validated['integer_size'] = 4
        
        return validated
    
    def convert(self, spectrum: SpectrumData, output_path: Path, 
                progress: ProgressCallback, **params) -> None:
        """Write spectrum as Z1D binary file"""
        params = self.validate_parameters(params)
        
        array_size = params['array_size']
        pad_value = params['pad_value']
        int_size = params['integer_size']
        
        progress.report(20, "Preparing data array")
        
        # Prepare data array
        counts = list(spectrum.counts)
        
        # Pad or truncate to exact array size
        if len(counts) < array_size:
            counts.extend([pad_value] * (array_size - len(counts)))
        elif len(counts) > array_size:
            counts = counts[:array_size]
        
        progress.report(50, "Writing binary data")
        
        # Determine struct format
        format_map = {2: 'h', 4: 'i', 8: 'q'}  # short, int, long long
        struct_format = format_map[int_size]
        
        # Write binary file
        with open(output_path, 'wb') as f:
            for i, count in enumerate(counts):
                try:
                    f.write(struct.pack(struct_format, count))
                except struct.error:
                    # Handle overflow by clamping
                    max_val = 2**(int_size * 8 - 1) - 1
                    min_val = -(2**(int_size * 8 - 1))
                    clamped_count = max(min_val, min(max_val, count))
                    f.write(struct.pack(struct_format, clamped_count))
                
                # Report progress periodically
                if i % 1000 == 0:
                    progress_pct = 50 + (i / len(counts)) * 50
                    progress.report(progress_pct, f"Writing element {i}")
        
        progress.report(100, f"Z1D binary conversion complete ({array_size} elements)")

class ConversionEngine:
    """Main conversion service with job management"""
    
    def __init__(self):
        self.file_reader = SpectrumFileReader()
        self.output_formats: Dict[str, OutputFormat] = {
            'txt_full': TXTFullFormat(),
            'txt_simple': TXTSimpleFormat(),
            'z1d': Z1DFormat(),
        }
        self._job_counter = 0
    
    def get_available_formats(self) -> Dict[str, str]:
        """Get available output formats"""
        return {key: fmt.get_name() for key, fmt in self.output_formats.items()}
    
    def add_format(self, key: str, format_handler: OutputFormat):
        """Add custom output format"""
        self.output_formats[key] = format_handler
    
    def create_job(self, source_file: Union[str, Path], target_format: str, 
                   output_path: Optional[Path] = None, **parameters) -> ConversionJob:
        """Create a new conversion job"""
        self._job_counter += 1
        job_id = f"job_{self._job_counter:04d}"
        
        return ConversionJob(
            job_id=job_id,
            source_file=Path(source_file),
            target_format=target_format,
            output_path=output_path,
            parameters=parameters
        )
    
    def convert_single(self, job: ConversionJob, 
                      progress_callback: Optional[Callable[[float, str], None]] = None) -> ConversionResult:
        """Convert a single file"""
        start_time = time.time()
        progress = ProgressCallback(progress_callback)
        
        result = ConversionResult(
            job=job,
            status=ConversionStatus.IN_PROGRESS,
            start_time=start_time
        )
        
        try:
            progress.report(0, f"Starting conversion of {job.source_file.name}")
            
            # Validate target format
            if job.target_format not in self.output_formats:
                raise ConversionError(f"Unknown target format: {job.target_format}")
            
            format_handler = self.output_formats[job.target_format]
            
            # Read source file
            progress.report(10, "Reading source file")
            spectrum = self.file_reader.read_file(job.source_file)
            
            # Determine output path
            if job.output_path:
                output_path = job.output_path
            else:
                output_path = job.source_file.with_suffix(format_handler.get_extension())
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Perform conversion
            progress.report(20, "Converting format")
            format_handler.convert(spectrum, output_path, progress, **job.parameters)
            
            # Update result
            end_time = time.time()
            result.status = ConversionStatus.COMPLETED
            result.output_file = output_path
            result.end_time = end_time
            result.processing_time = end_time - start_time
            
            progress.report(100, f"Conversion completed: {output_path.name}")
            
        except Exception as e:
            end_time = time.time()
            result.status = ConversionStatus.FAILED
            result.error_message = str(e)
            result.end_time = end_time
            result.processing_time = end_time - start_time
            
            progress.report(0, f"Conversion failed: {e}")
        
        return result

class BatchConversionManager:
    """Manages batch conversion operations with parallel processing"""
    
    def __init__(self, conversion_engine: ConversionEngine, max_workers: int = 4):
        self.conversion_engine = conversion_engine
        self.max_workers = max_workers
        self._cancelled = False
    
    def find_convertible_files(self, folder: Path, target_format: str) -> List[Path]:
        """Find files in folder that can be converted"""
        if not folder.is_dir():
            raise ValueError(f"Not a directory: {folder}")
        
        convertible_files = []
        
        for file_path in folder.iterdir():
            if file_path.is_file():
                try:
                    if self.conversion_engine.file_reader.can_read_file(file_path):
                        convertible_files.append(file_path)
                except Exception:
                    continue  # Skip files that can't be read
        
        return convertible_files
    
    def create_batch_jobs(self, files: List[Path], target_format: str, 
                         output_dir: Optional[Path] = None, 
                         create_subfolders: bool = True, **parameters) -> List[ConversionJob]:
        """Create conversion jobs for batch processing"""
        if not files:
            return []
        
        jobs = []
        
        # Determine base output directory
        if output_dir is None:
            output_dir = files[0].parent
        
        # Create format-specific subdirectory if requested
        if create_subfolders:
            format_handler = self.conversion_engine.output_formats[target_format]
            subdir_name = f"{target_format}_output"
            actual_output_dir = output_dir / subdir_name
        else:
            actual_output_dir = output_dir
        
        # Create jobs
        for file_path in files:
            job = self.conversion_engine.create_job(
                source_file=file_path,
                target_format=target_format,
                output_path=actual_output_dir / f"{file_path.stem}{self.conversion_engine.output_formats[target_format].get_extension()}",
                **parameters
            )
            jobs.append(job)
        
        return jobs
    
    def convert_batch(self, jobs: List[ConversionJob], 
                     progress_callback: Optional[Callable[[float, str], None]] = None) -> List[ConversionResult]:
        """Convert multiple files in parallel"""
        if not jobs:
            return []
        
        results = []
        self._cancelled = False
        
        if progress_callback:
            progress_callback(0, f"Starting batch conversion of {len(jobs)} files")
        
        # Sort jobs by priority (highest first)
        sorted_jobs = sorted(jobs, key=lambda j: j.priority, reverse=True)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_job = {}
            for job in sorted_jobs:
                if self._cancelled:
                    break
                future = executor.submit(self.conversion_engine.convert_single, job, None)
                future_to_job[future] = job
            
            # Collect results as they complete
            completed = 0
            for future in concurrent.futures.as_completed(future_to_job):
                if self._cancelled:
                    break
                
                result = future.result()
                results.append(result)
                completed += 1
                
                # Update progress
                progress = (completed / len(jobs)) * 100
                if progress_callback:
                    status = "succeeded" if result.success else "failed"
                    progress_callback(
                        progress,
                        f"Processed {completed}/{len(jobs)} files "
                        f"({result.job.source_file.name} {status})"
                    )
        
        return results
    
    def cancel(self):
        """Cancel ongoing batch conversion"""
        self._cancelled = True

# Example usage
if __name__ == "__main__":
    # Test the conversion engine
    engine = ConversionEngine()
    
    print("Available formats:")
    for key, name in engine.get_available_formats().items():
        print(f"  {key}: {name}")
    
    # Example job creation
    try:
        job = engine.create_job(
            source_file="test.spe",
            target_format="txt_full",
            include_energy=True,
            include_statistics=True
        )
        print(f"Created job: {job.job_id}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("Conversion engine ready")
