"""
Spectrum Data Models and File Handlers
Core data layer with proper error handling and security
"""

import os
import re
import struct
import subprocess
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpectrumError(Exception):
    """Base exception for spectrum processing errors"""
    pass

class FileFormatError(SpectrumError):
    """Raised when file format is invalid or unsupported"""
    pass

class ConversionError(SpectrumError):
    """Raised when conversion between formats fails"""
    pass

class SecurityError(SpectrumError):
    """Raised when security validation fails"""
    pass

class FileFormat(Enum):
    """Supported file formats"""
    SPE = "spe"
    CNF = "cnf"
    TXT = "txt"
    Z1D = "z1d"

@dataclass(frozen=True)
class SpectrumMetadata:
    """Immutable spectrum metadata container"""
    spec_id: Optional[str] = None
    date: Optional[str] = None
    live_time: Optional[float] = None
    real_time: Optional[float] = None
    device: Optional[str] = None
    calibration_params: Optional[Tuple[float, ...]] = None
    source_file: Optional[Path] = None
    file_format: Optional[FileFormat] = None
    
    @property
    def dead_time_percent(self) -> Optional[float]:
        """Calculate dead time percentage"""
        if self.live_time and self.real_time and self.real_time > 0:
            return ((self.real_time - self.live_time) / self.real_time) * 100
        return None
    
    @property
    def has_energy_calibration(self) -> bool:
        """Check if energy calibration is available"""
        return self.calibration_params is not None and len(self.calibration_params) >= 2
    
    def calculate_energy(self, channel: int) -> Optional[float]:
        """Calculate energy for given channel using calibration"""
        if not self.has_energy_calibration:
            return None
        
        params = self.calibration_params
        a = params[0]
        b = params[1] 
        c = params[2] if len(params) > 2 else 0
        
        return a + b * channel + c * channel**2

@dataclass(frozen=True)
class SpectrumData:
    """Immutable spectrum data container"""
    channels: Tuple[int, ...]
    counts: Tuple[int, ...]
    metadata: SpectrumMetadata
    
    def __post_init__(self):
        # Validation
        if len(self.channels) != len(self.counts):
            raise ValueError("Channels and counts must have same length")
        
        if not all(isinstance(c, int) and c >= 0 for c in self.counts):
            raise ValueError("Counts must be non-negative integers")
        
        if not all(isinstance(ch, int) and ch >= 0 for ch in self.channels):
            raise ValueError("Channels must be non-negative integers")
    
    @property
    def channel_count(self) -> int:
        """Number of channels"""
        return len(self.channels)
    
    @property
    def total_counts(self) -> int:
        """Total counts across all channels"""
        return sum(self.counts)
    
    @property
    def peak_channel(self) -> int:
        """Channel with maximum counts"""
        max_count = max(self.counts)
        return self.channels[self.counts.index(max_count)]
    
    @property
    def peak_counts(self) -> int:
        """Maximum counts in any channel"""
        return max(self.counts)
    
    @property
    def count_rate(self) -> Optional[float]:
        """Calculate count rate if live time is available"""
        if self.metadata.live_time and self.metadata.live_time > 0:
            return self.total_counts / self.metadata.live_time
        return None
    
    def get_energy_spectrum(self) -> Optional[Tuple[Tuple[float, ...], Tuple[int, ...]]]:
        """Get energy-calibrated spectrum if calibration is available"""
        if not self.metadata.has_energy_calibration:
            return None
        
        energies = []
        for channel in self.channels:
            energy = self.metadata.calculate_energy(channel)
            if energy is not None:
                energies.append(energy)
            else:
                return None
        
        return (tuple(energies), self.counts)

class FileValidator:
    """Validates file paths and content for security"""
    
    @staticmethod
    def validate_file_path(file_path: Union[str, Path]) -> Path:
        """Validate and normalize file path"""
        path = Path(file_path).resolve()
        
        # Security checks
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if not path.is_file():
            raise SecurityError(f"Path is not a file: {path}")
        
        # Check for path traversal attempts
        if '..' in str(path):
            raise SecurityError(f"Path traversal detected: {path}")
        
        # Check file size (prevent DoS attacks)
        file_size = path.stat().st_size
        max_size = 100 * 1024 * 1024  # 100MB limit
        if file_size > max_size:
            raise SecurityError(f"File too large: {file_size} bytes (max: {max_size})")
        
        return path
    
    @staticmethod
    def validate_executable(exe_path: str) -> bool:
        """Validate external executable availability"""
        try:
            result = subprocess.run(
                [exe_path, "--version"],
                capture_output=True,
                timeout=5,
                check=False
            )
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

class FileFormatHandler(ABC):
    """Abstract base for file format handlers"""
    
    @abstractmethod
    def can_handle(self, file_path: Path) -> bool:
        """Check if this handler can process the file"""
        pass
    
    @abstractmethod
    def parse(self, file_path: Path) -> SpectrumData:
        """Parse file and return spectrum data"""
        pass
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions"""
        pass
    
    @abstractmethod
    def get_format_name(self) -> str:
        """Return human-readable format name"""
        pass

class SPEFormatHandler(FileFormatHandler):
    """Handler for SPE format files"""
    
    def get_format_name(self) -> str:
        return "SPE Spectrum File"
    
    def get_supported_extensions(self) -> List[str]:
        return ['.spe']
    
    def can_handle(self, file_path: Path) -> bool:
        """Check if file is a valid SPE file"""
        if file_path.suffix.lower() != '.spe':
            return False
        
        try:
            # Quick validation - check for SPE markers
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                header = f.read(1000)
                return '$SPEC_ID:' in header or '$DATA:' in header
        except Exception:
            return False
    
    def parse(self, file_path: Path) -> SpectrumData:
        """Parse SPE file with comprehensive error handling"""
        validated_path = FileValidator.validate_file_path(file_path)
        
        try:
            with open(validated_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            raise FileFormatError(f"Cannot read SPE file {validated_path}: {e}")
        
        metadata = self._parse_metadata(content, validated_path)
        channels, counts = self._parse_spectrum_data(content)
        
        return SpectrumData(
            channels=tuple(channels),
            counts=tuple(counts),
            metadata=metadata
        )
    
    def _parse_metadata(self, content: str, file_path: Path) -> SpectrumMetadata:
        """Extract metadata from SPE content"""
        metadata_dict = {
            'source_file': file_path,
            'file_format': FileFormat.SPE
        }
        
        # Extract spec ID
        if match := re.search(r'\$SPEC_ID:\s*(.+)', content):
            metadata_dict['spec_id'] = match.group(1).strip()
        
        # Extract date
        if match := re.search(r'\$DATE_MEA:\s*(.+)', content):
            metadata_dict['date'] = match.group(1).strip()
        
        # Extract measurement times
        if match := re.search(r'\$MEAS_TIM:\s*(.+)', content):
            times = match.group(1).strip().split()
            if len(times) >= 2:
                try:
                    metadata_dict['live_time'] = float(times[0])
                    metadata_dict['real_time'] = float(times[1])
                except ValueError:
                    logger.warning(f"Invalid time values in {file_path}")
        
        # Extract device ID
        if match := re.search(r'\$DEVICE_ID:\s*(.+)', content):
            metadata_dict['device'] = match.group(1).strip()
        
        # Extract calibration parameters
        if match := re.search(r'\$SPEC_CAL:\s*(.+)', content):
            try:
                cal_values = [float(x) for x in match.group(1).strip().split()]
                metadata_dict['calibration_params'] = tuple(cal_values)
            except ValueError:
                logger.warning(f"Invalid calibration parameters in {file_path}")
        
        return SpectrumMetadata(**metadata_dict)
    
    def _parse_spectrum_data(self, content: str) -> Tuple[List[int], List[int]]:
        """Extract spectrum data from SPE content"""
        data_match = re.search(r'\$DATA:\s*\n\s*(\d+)\s+(\d+)\s*\n(.*)', content, re.DOTALL)
        if not data_match:
            raise FileFormatError("No valid data section found in SPE file")
        
        try:
            start_channel = int(data_match.group(1))
            end_channel = int(data_match.group(2))
            data_section = data_match.group(3)
        except ValueError as e:
            raise FileFormatError(f"Invalid data section header: {e}")
        
        # Parse data values
        data_lines = data_section.strip().split('\n')
        all_values = []
        
        for line_num, line in enumerate(data_lines):
            line = line.strip()
            if not line:
                continue
                
            try:
                values = [int(x) for x in line.split()]
                all_values.extend(values)
            except ValueError:
                logger.warning(f"Skipping invalid data line {line_num}: {line}")
                continue
        
        # Validate data length
        expected_points = end_channel - start_channel + 1
        if len(all_values) < expected_points:
            logger.warning(f"Data section has fewer points than expected "
                         f"({len(all_values)} vs {expected_points})")
        
        # Generate channel/count pairs
        actual_points = min(len(all_values), expected_points)
        channels = list(range(start_channel, start_channel + actual_points))
        counts = all_values[:actual_points]
        
        return channels, counts

class CNFFormatHandler(FileFormatHandler):
    """Handler for CNF format files using external converter"""
    
    def __init__(self, converter_path: str = "cnf2txt.exe"):
        self.converter_path = converter_path
        self._converter_available = FileValidator.validate_executable(converter_path)
        
        if not self._converter_available:
            logger.warning(f"CNF converter not available: {converter_path}")
    
    def get_format_name(self) -> str:
        return "CNF Spectrum File"
    
    def get_supported_extensions(self) -> List[str]:
        return ['.cnf']
    
    def can_handle(self, file_path: Path) -> bool:
        """Check if file is a CNF file and converter is available"""
        return (file_path.suffix.lower() == '.cnf' and self._converter_available)
    
    def parse(self, file_path: Path) -> SpectrumData:
        """Parse CNF file using external converter"""
        if not self._converter_available:
            raise FileFormatError(f"CNF converter not available: {self.converter_path}")
        
        validated_path = FileValidator.validate_file_path(file_path)
        
        # Convert to temporary TXT file
        temp_txt = self._convert_to_txt(validated_path)
        
        try:
            channels, counts = self._parse_txt_output(temp_txt)
            
            metadata = SpectrumMetadata(
                source_file=validated_path,
                file_format=FileFormat.CNF
            )
            
            return SpectrumData(
                channels=tuple(channels),
                counts=tuple(counts),
                metadata=metadata
            )
        finally:
            # Clean up temporary file
            if temp_txt.exists():
                try:
                    temp_txt.unlink()
                except Exception:
                    logger.warning(f"Could not remove temporary file: {temp_txt}")
    
    def _convert_to_txt(self, cnf_path: Path) -> Path:
        """Convert CNF to TXT using external tool"""
        try:
            result = subprocess.run(
                [self.converter_path, str(cnf_path)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=cnf_path.parent,
                check=True
            )
        except subprocess.TimeoutExpired:
            raise ConversionError(f"CNF conversion timed out for {cnf_path}")
        except subprocess.CalledProcessError as e:
            raise ConversionError(f"CNF conversion failed: {e.stderr}")
        except FileNotFoundError:
            raise ConversionError(f"CNF converter not found: {self.converter_path}")
        
        # Check for output file
        txt_path = cnf_path.with_suffix('.txt')
        if not txt_path.exists():
            raise ConversionError(f"CNF converter did not create output file: {txt_path}")
        
        return txt_path
    
    def _parse_txt_output(self, txt_path: Path) -> Tuple[List[int], List[int]]:
        """Parse the TXT output from CNF converter"""
        try:
            with open(txt_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            raise FileFormatError(f"Cannot read converted TXT file: {e}")
        
        channels = []
        counts = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            try:
                count = int(float(line))
                if count >= 0:
                    channels.append(i)
                    counts.append(count)
            except ValueError:
                continue
        
        if not channels:
            raise FileFormatError("No valid data found in converted TXT file")
        
        return channels, counts

class SpectrumFileReader:
    """Main interface for reading spectrum files"""
    
    def __init__(self):
        self.handlers: List[FileFormatHandler] = [
            SPEFormatHandler(),
            CNFFormatHandler(),
        ]
        logger.info(f"Initialized with {len(self.handlers)} format handlers")
    
    def add_handler(self, handler: FileFormatHandler):
        """Add a custom format handler"""
        self.handlers.append(handler)
        logger.info(f"Added custom handler: {handler.get_format_name()}")
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get mapping of format names to supported extensions"""
        formats = {}
        for handler in self.handlers:
            formats[handler.get_format_name()] = handler.get_supported_extensions()
        return formats
    
    def detect_format(self, file_path: Union[str, Path]) -> Optional[FileFormatHandler]:
        """Detect the appropriate handler for a file"""
        validated_path = FileValidator.validate_file_path(file_path)
        
        for handler in self.handlers:
            if handler.can_handle(validated_path):
                return handler
        
        return None
    
    def read_file(self, file_path: Union[str, Path]) -> SpectrumData:
        """Read spectrum file using appropriate handler"""
        validated_path = FileValidator.validate_file_path(file_path)
        
        # Find appropriate handler
        handler = self.detect_format(validated_path)
        if handler is None:
            supported_formats = self.get_supported_formats()
            raise FileFormatError(
                f"Unsupported file format: {validated_path.suffix}. "
                f"Supported formats: {supported_formats}"
            )
        
        logger.info(f"Using {handler.get_format_name()} for {validated_path.name}")
        return handler.parse(validated_path)
    
    def can_read_file(self, file_path: Union[str, Path]) -> bool:
        """Check if file can be read without actually reading it"""
        try:
            return self.detect_format(file_path) is not None
        except Exception:
            return False

# Example usage and basic testing
if __name__ == "__main__":
    # Test the data layer
    reader = SpectrumFileReader()
    
    print("Supported formats:")
    for format_name, extensions in reader.get_supported_formats().items():
        print(f"  {format_name}: {', '.join(extensions)}")
    
    # This would be used by higher-level components
    try:
        # spectrum = reader.read_file("test.spe")
        # print(f"Loaded spectrum: {spectrum.total_counts} total counts")
        print("Data layer ready for integration")
    except Exception as e:
        print(f"Error: {e}")
