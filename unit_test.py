"""
Comprehensive Unit Test Suite for Spectrum Converter Pro
Tests all major components with good coverage
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import struct
import json

# Import the modules to test
from spectrum_data_models import (
    SpectrumData, SpectrumMetadata, FileValidator, 
    SPEFormatHandler, CNFFormatHandler, SpectrumFileReader,
    FileFormat, SpectrumError, FileFormatError, SecurityError
)
from conversion_engine import (
    ConversionEngine, ConversionJob, ConversionResult, ConversionStatus,
    TXTFullFormat, TXTSimpleFormat, Z1DFormat, BatchConversionManager
)
from analysis_engine import (
    SpectrumAnalyzer, Peak, SpectrumStatistics, PeakSearchMethod,
    BackgroundEstimator, PeakFinder
)
from plotting_engine import SpectrumPlotter, PlotOptions, PlotType


class TestSpectrumData:
    """Test SpectrumData and SpectrumMetadata classes"""
    
    def test_spectrum_metadata_creation(self):
        """Test basic metadata creation"""
        metadata = SpectrumMetadata(
            spec_id="TEST001",
            live_time=100.0,
            real_time=110.0,
            calibration_params=(0.5, 1.0, 0.001)
        )
        
        assert metadata.spec_id == "TEST001"
        assert metadata.dead_time_percent == pytest.approx(9.09, rel=1e-2)
        assert metadata.has_energy_calibration is True
        
        # Test energy calculation
        energy = metadata.calculate_energy(100)
        expected = 0.5 + 1.0 * 100 + 0.001 * 100**2
        assert energy == pytest.approx(expected)
    
    def test_spectrum_data_creation(self):
        """Test SpectrumData creation and validation"""
        metadata = SpectrumMetadata(spec_id="TEST")
        channels = (0, 1, 2, 3, 4)
        counts = (10, 20, 50, 30, 5)
        
        spectrum = SpectrumData(channels, counts, metadata)
        
        assert spectrum.channel_count == 5
        assert spectrum.total_counts == 115
        assert spectrum.peak_channel == 2
        assert spectrum.peak_counts == 50
    
    def test_spectrum_data_validation(self):
        """Test SpectrumData validation"""
        metadata = SpectrumMetadata()
        
        # Test mismatched lengths
        with pytest.raises(ValueError, match="Channels and counts must have same length"):
            SpectrumData((0, 1, 2), (10, 20), metadata)
        
        # Test negative counts
        with pytest.raises(ValueError, match="Counts must be non-negative"):
            SpectrumData((0, 1), (10, -5), metadata)
    
    def test_energy_spectrum(self):
        """Test energy spectrum calculation"""
        metadata = SpectrumMetadata(calibration_params=(0.0, 1.0))
        channels = (0, 1, 2)
        counts = (10, 20, 30)
        
        spectrum = SpectrumData(channels, counts, metadata)
        energy_spectrum = spectrum.get_energy_spectrum()
        
        assert energy_spectrum is not None
        energies, spectrum_counts = energy_spectrum
        assert energies == (0.0, 1.0, 2.0)
        assert spectrum_counts == counts


class TestFileValidator:
    """Test FileValidator security functions"""
    
    def test_validate_file_path_success(self):
        """Test successful file path validation"""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            validated = FileValidator.validate_file_path(tmp_path)
            assert validated.exists()
            assert validated.is_absolute()
        finally:
            tmp_path.unlink()
    
    def test_validate_file_path_not_found(self):
        """Test validation of non-existent file"""
        with pytest.raises(FileNotFoundError):
            FileValidator.validate_file_path("/nonexistent/file.txt")
    
    def test_validate_file_path_directory(self):
        """Test validation rejects directories"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with pytest.raises(SecurityError, match="Path is not a file"):
                FileValidator.validate_file_path(tmp_dir)
    
    def test_validate_file_path_too_large(self):
        """Test validation rejects oversized files"""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            # Write > 100MB of data
            tmp.write(b"x" * (101 * 1024 * 1024))
            tmp_path = Path(tmp.name)
        
        try:
            with pytest.raises(SecurityError, match="File too large"):
                FileValidator.validate_file_path(tmp_path)
        finally:
            tmp_path.unlink()


class TestSPEFormatHandler:
    """Test SPE format handler"""
    
    def create_sample_spe_file(self, tmp_path: Path) -> Path:
        """Create a sample SPE file for testing"""
        spe_content = """$SPEC_ID:
TEST_SPECTRUM_001
$DATE_MEA:
02/15/2024 14:30:15
$MEAS_TIM:
100.0 110.0
$DEVICE_ID:
TEST_DETECTOR
$SPEC_CAL:
0.5 1.0 0.001
$DATA:
0 4095
10 20 50 100 80 60 40 30 25 20
15 12 10 8 6 5 4 3 2 1
$ROI:
"""
        spe_file = tmp_path / "test.spe"
        spe_file.write_text(spe_content)
        return spe_file
    
    def test_can_handle_spe_file(self):
        """Test SPE file detection"""
        handler = SPEFormatHandler()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            spe_file = self.create_sample_spe_file(tmp_path)
            
            assert handler.can_handle(spe_file) is True
            
            # Test non-SPE file
            txt_file = tmp_path / "test.txt"
            txt_file.write_text("This is not an SPE file")
            assert handler.can_handle(txt_file) is False
    
    def test_parse_spe_file(self):
        """Test SPE file parsing"""
        handler = SPEFormatHandler()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            spe_file = self.create_sample_spe_file(tmp_path)
            
            spectrum = handler.parse(spe_file)
            
            # Check metadata
            assert spectrum.metadata.spec_id == "TEST_SPECTRUM_001"
            assert spectrum.metadata.date == "02/15/2024 14:30:15"
            assert spectrum.metadata.live_time == 100.0
            assert spectrum.metadata.real_time == 110.0
            assert spectrum.metadata.device == "TEST_DETECTOR"
            assert spectrum.metadata.calibration_params == (0.5, 1.0, 0.001)
            
            # Check data
            assert len(spectrum.channels) == 20
            assert spectrum.channels[0] == 0
            assert spectrum.counts[0] == 10
            assert spectrum.peak_counts == 100
    
    def test_parse_invalid_spe_file(self):
        """Test parsing invalid SPE file"""
        handler = SPEFormatHandler()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            bad_spe = tmp_path / "bad.spe"
            bad_spe.write_text("This is not a valid SPE file")
            
            with pytest.raises(FileFormatError, match="No valid data section"):
                handler.parse(bad_spe)


class TestCNFFormatHandler:
    """Test CNF format handler"""
    
    @patch('subprocess.run')
    def test_cnf_handler_with_mock_converter(self, mock_run):
        """Test CNF handler with mocked external converter"""
        # Mock successful converter check
        mock_run.return_value = Mock(returncode=0)
        
        handler = CNFFormatHandler("mock_cnf2txt.exe")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cnf_file = tmp_path / "test.cnf"
            cnf_file.write_bytes(b"Mock CNF data")
            
            txt_file = tmp_path / "test.txt"
            txt_file.write_text("10\n20\n30\n40\n50\n")
            
            # Mock converter execution
            mock_run.return_value = Mock(returncode=0, stderr="")
            
            assert handler.can_handle(cnf_file) is True
            
            spectrum = handler.parse(cnf_file)
            assert spectrum.metadata.file_format == FileFormat.CNF
            assert len(spectrum.channels) == 5
            assert spectrum.counts == (10, 20, 30, 40, 50)
    
    def test_cnf_handler_converter_not_found(self):
        """Test CNF handler when converter is not available"""
        with pytest.raises(FileFormatError, match="CNF converter not available"):
            CNFFormatHandler("nonexistent_converter.exe")


class TestSpectrumFileReader:
    """Test main file reader interface"""
    
    def test_get_supported_formats(self):
        """Test getting supported formats"""
        reader = SpectrumFileReader()
        formats = reader.get_supported_formats()
        
        assert "SPE Spectrum File" in formats
        assert ".spe" in formats["SPE Spectrum File"]
    
    def test_read_file_with_spe(self):
        """Test reading SPE file through main interface"""
        reader = SpectrumFileReader()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create sample SPE file
            spe_content = """$SPEC_ID:
TEST
$DATA:
0 2
100 200 150
"""
            spe_file = tmp_path / "test.spe"
            spe_file.write_text(spe_content)
            
            spectrum = reader.read_file(spe_file)
            assert spectrum.metadata.spec_id == "TEST"
            assert len(spectrum.channels) == 3
    
    def test_read_unsupported_file(self):
        """Test reading unsupported file format"""
        reader = SpectrumFileReader()
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            unknown_file = tmp_path / "test.xyz"
            unknown_file.write_text("Unknown format")
            
            with pytest.raises(FileFormatError, match="Unsupported file format"):
                reader.read_file(unknown_file)


class TestConversionEngine:
    """Test conversion engine and formats"""
    
    def test_conversion_job_creation(self):
        """Test conversion job creation"""
        engine = ConversionEngine()
        
        job = engine.create_job(
            source_file="test.spe",
            target_format="txt_full",
            include_energy=True
        )
        
        assert job.source_file == Path("test.spe")
        assert job.target_format == "txt_full"
        assert job.parameters["include_energy"] is True
    
    def test_get_available_formats(self):
        """Test getting available output formats"""
        engine = ConversionEngine()
        formats = engine.get_available_formats()
        
        assert "txt_full" in formats
        assert "txt_simple" in formats
        assert "z1d" in formats
    
    def test_txt_full_format(self):
        """Test TXT full format conversion"""
        format_handler = TXTFullFormat()
        
        # Create test spectrum
        metadata = SpectrumMetadata(
            spec_id="TEST",
            live_time=100.0,
            calibration_params=(0.0, 1.0)
        )
        spectrum = SpectrumData((0, 1, 2), (10, 20, 15), metadata)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "output.txt"
            progress = Mock()
            
            format_handler.convert(spectrum, output_path, progress)
            
            # Check file was created and has content
            assert output_path.exists()
            content = output_path.read_text()
            assert "GAMMA RAY SPECTRUM DATA CONVERSION" in content
            assert "TEST" in content
            assert "10\t20" in content  # Data section
    
    def test_z1d_format(self):
        """Test Z1D binary format conversion"""
        format_handler = Z1DFormat()
        
        metadata = SpectrumMetadata()
        spectrum = SpectrumData((0, 1, 2), (100, 200, 150), metadata)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "output.z1d"
            progress = Mock()
            
            format_handler.convert(spectrum, output_path, progress, array_size=10)
            
            # Check file was created with correct size
            assert output_path.exists()
            assert output_path.stat().st_size == 10 * 4  # 10 integers * 4 bytes
            
            # Check content
            with open(output_path, 'rb') as f:
                data = f.read()
                values = struct.unpack('10i', data)
                assert values[:3] == (100, 200, 150)
                assert values[3:] == (0,) * 7  # Padded with zeros


class TestBatchConversionManager:
    """Test batch conversion functionality"""
    
    def test_find_convertible_files(self):
        """Test finding convertible files in directory"""
        engine = ConversionEngine()
        batch_manager = BatchConversionManager(engine)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create test files
            spe_file = tmp_path / "test.spe"
            spe_file.write_text("$SPEC_ID:\nTEST\n$DATA:\n0 1\n100 200\n")
            
            txt_file = tmp_path / "readme.txt"
            txt_file.write_text("Not a spectrum file")
            
            files = batch_manager.find_convertible_files(tmp_path, "txt_full")
            
            assert len(files) == 1
            assert files[0].name == "test.spe"
    
    def test_create_batch_jobs(self):
        """Test creating batch conversion jobs"""
        engine = ConversionEngine()
        batch_manager = BatchConversionManager(engine)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            files = [tmp_path / "file1.spe", tmp_path / "file2.spe"]
            for f in files:
                f.touch()
            
            jobs = batch_manager.create_batch_jobs(
                files=files,
                target_format="txt_full",
                create_subfolders=True
            )
            
            assert len(jobs) == 2
            assert all(job.target_format == "txt_full" for job in jobs)
            assert "txt_full_output" in str(jobs[0].output_path)


class TestSpectrumAnalyzer:
    """Test spectrum analysis functionality"""
    
    def create_test_spectrum(self) -> SpectrumData:
        """Create a test spectrum with known characteristics"""
        # Create spectrum with clear peaks at channels 10, 50, 100
        channels = list(range(200))
        counts = [5] * 200  # Background level
        
        # Add peaks
        counts[10] = 100
        counts[50] = 200
        counts[100] = 150
        
        # Add some noise around peaks
        for i in range(9, 12):
            if i != 10:
                counts[i] = 20
        
        metadata = SpectrumMetadata(
            live_time=100.0,
            calibration_params=(0.0, 1.0)
        )
        
        return SpectrumData(tuple(channels), tuple(counts), metadata)
    
    def test_analyze_spectrum(self):
        """Test basic spectrum analysis"""
        analyzer = SpectrumAnalyzer()
        spectrum = self.create_test_spectrum()
        
        analysis = analyzer.analyze(spectrum, peak_min_height=50)
        
        # Check statistics
        assert analysis.statistics.total_counts > 0
        assert analysis.statistics.peak_channel == 50  # Highest peak
        assert analysis.statistics.peak_counts == 200
        
        # Check peaks found
        assert len(analysis.peaks) == 3
        peak_channels = [p.channel for p in analysis.peaks]
        assert 10 in peak_channels
        assert 50 in peak_channels
        assert 100 in peak_channels
    
    def test_background_estimation(self):
        """Test background estimation methods"""
        analyzer = SpectrumAnalyzer()
        spectrum = self.create_test_spectrum()
        
        # Test percentile method
        bg_percentile = analyzer.background_estimator.percentile_method(spectrum, 10)
        assert len(bg_percentile) == len(spectrum.counts)
        assert all(bg >= 0 for bg in bg_percentile)
        
        # Test moving minimum
        bg_moving = analyzer.background_estimator.moving_minimum(spectrum, 20)
        assert len(bg_moving) == len(spectrum.counts)
    
    def test_peak_finder(self):
        """Test peak finding algorithms"""
        peak_finder = PeakFinder()
        spectrum = self.create_test_spectrum()
        
        # Test local maximum method
        peaks = peak_finder.find_peaks_local_maximum(
            spectrum, min_height=50, min_distance=5
        )
        
        assert len(peaks) >= 2  # Should find main peaks
        assert all(p.counts >= 50 for p in peaks)
        
        # Test FWHM estimation
        fwhm = peak_finder.estimate_fwhm(spectrum, 50)
        assert fwhm is not None
        assert fwhm > 0
    
    def test_export_analysis_report(self):
        """Test exporting analysis report"""
        analyzer = SpectrumAnalyzer()
        spectrum = self.create_test_spectrum()
        analysis = analyzer.analyze(spectrum)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "report.txt"
            analyzer.export_analysis_report(analysis, report_path)
            
            assert report_path.exists()
            content = report_path.read_text()
            assert "SPECTRUM ANALYSIS REPORT" in content
            assert "DETECTED PEAKS" in content


class TestPlottingEngine:
    """Test plotting functionality"""
    
    def test_plotter_availability(self):
        """Test plotter availability check"""
        plotter = SpectrumPlotter()
        # Note: This may be False in CI environments without matplotlib
        availability = plotter.is_available()
        assert isinstance(availability, bool)
    
    @pytest.mark.skipif(not SpectrumPlotter().is_available(), 
                       reason="Matplotlib not available")
    def test_create_simple_plot(self):
        """Test creating a simple plot"""
        plotter = SpectrumPlotter()
        
        metadata = SpectrumMetadata()
        spectrum = SpectrumData((0, 1, 2, 3), (10, 20, 30, 15), metadata)
        
        figure = plotter.create_simple_plot(spectrum, "Test Plot")
        assert figure is not None
    
    def test_plot_options(self):
        """Test plot options configuration"""
        options = PlotOptions(
            title="Test Plot",
            use_energy_axis=True,
            show_peaks=True,
            y_scale="log"
        )
        
        assert options.title == "Test Plot"
        assert options.use_energy_axis is True
        assert options.show_peaks is True
        assert options.y_scale == "log"


class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_complete_conversion_workflow(self):
        """Test complete file conversion workflow"""
        # Create test SPE file
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            spe_content = """$SPEC_ID:
INTEGRATION_TEST
$DATA:
0 9
100 200 150 120 90 80 70 60 50 40
"""
            spe_file = tmp_path / "test.spe"
            spe_file.write_text(spe_content)
            
            # Read file
            reader = SpectrumFileReader()
            spectrum = reader.read_file(spe_file)
            
            # Convert to different formats
            engine = ConversionEngine()
            
            # Test TXT conversion
            txt_job = engine.create_job(spe_file, "txt_full")
            txt_result = engine.convert_single(txt_job)
            assert txt_result.success
            assert txt_result.output_file.exists()
            
            # Test Z1D conversion
            z1d_job = engine.create_job(spe_file, "z1d", array_size=16)
            z1d_result = engine.convert_single(z1d_job)
            assert z1d_result.success
            assert z1d_result.output_file.exists()
            assert z1d_result.output_file.stat().st_size == 16 * 4
    
    def test_analysis_and_plotting_workflow(self):
        """Test analysis and plotting workflow"""
        # Create test spectrum with peaks
        metadata = SpectrumMetadata(
            spec_id="ANALYSIS_TEST",
            calibration_params=(0.0, 0.5)
        )
        
        # Create spectrum with clear peaks
        channels = list(range(100))
        counts = [2] * 100  # Background
        counts[20] = 50   # Peak 1
        counts[40] = 100  # Peak 2
        counts[60] = 75   # Peak 3
        
        spectrum = SpectrumData(tuple(channels), tuple(counts), metadata)
        
        # Analyze spectrum
        analyzer = SpectrumAnalyzer()
        analysis = analyzer.analyze(spectrum, peak_min_height=20)
        
        assert len(analysis.peaks) >= 3
        assert analysis.statistics.total_counts > 0
        
        # Test plotting (if available)
        plotter = SpectrumPlotter()
        if plotter.is_available():
            figure = plotter.create_analysis_plot(analysis)
            assert figure is not None


# Fixtures for common test data
@pytest.fixture
def sample_spectrum():
    """Fixture providing a sample spectrum for testing"""
    metadata = SpectrumMetadata(
        spec_id="SAMPLE",
        live_time=100.0,
        real_time=105.0,
        calibration_params=(0.0, 1.0, 0.001)
    )
    
    channels = tuple(range(10))
    counts = (5, 10, 15, 25, 50, 40, 30, 20, 10, 5)
    
    return SpectrumData(channels, counts, metadata)


@pytest.fixture
def temp_directory():
    """Fixture providing a temporary directory"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


# Configuration for pytest
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


# Performance tests
class TestPerformance:
    """Performance tests for critical operations"""
    
    @pytest.mark.slow
    def test_large_spectrum_processing(self):
        """Test processing of large spectrum (performance test)"""
        # Create large spectrum (16K channels)
        channels = tuple(range(16384))
        counts = tuple([i % 100 for i in range(16384)])
        
        metadata = SpectrumMetadata()
        spectrum = SpectrumData(channels, counts, metadata)
        
        # Test analysis performance
        analyzer = SpectrumAnalyzer()
        import time
        
        start_time = time.time()
        analysis = analyzer.analyze(spectrum)
        end_time = time.time()
        
        # Should complete in reasonable time (< 5 seconds)
        assert end_time - start_time < 5.0
        assert analysis.statistics.total_counts > 0
    
    @pytest.mark.slow
    def test_batch_conversion_performance(self):
        """Test batch conversion performance"""
        engine = ConversionEngine()
        batch_manager = BatchConversionManager(engine, max_workers=2)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Create multiple test files
            files = []
            for i in range(10):
                spe_file = tmp_path / f"test_{i}.spe"
                spe_content = f"""$SPEC_ID:
TEST_{i}
$DATA:
0 99
""" + " ".join(str(j) for j in range(100))
                spe_file.write_text(spe_content)
                files.append(spe_file)
            
            # Test batch conversion
            jobs = batch_manager.create_batch_jobs(
                files=files,
                target_format="txt_simple",
                create_subfolders=False
            )
            
            import time
            start_time = time.time()
            results = batch_manager.convert_batch(jobs)
            end_time = time.time()
            
            # All should succeed
            assert all(r.success for r in results)
            # Should complete reasonably quickly with parallel processing
            assert end_time - start_time < 10.0


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
