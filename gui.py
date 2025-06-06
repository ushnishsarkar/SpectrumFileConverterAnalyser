"""
Main GUI Application
Clean interface that delegates to the various engines
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

# Import our engines
from spectrum_data_models import SpectrumFileReader, SpectrumData, FileFormatError
from conversion_engine import ConversionEngine, BatchConversionManager, ConversionJob
from analysis_engine import SpectrumAnalyzer, PeakSearchMethod
from plotting_engine import SpectrumPlotter, PlotOptions, PlotType

# Try to import matplotlib backend for tkinter
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    MATPLOTLIB_GUI_AVAILABLE = True
except ImportError:
    MATPLOTLIB_GUI_AVAILABLE = False

class ConfigManager:
    """Manages application configuration"""
    
    def __init__(self, config_file: str = "spectrum_converter_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_default_config()
        self.load_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration"""
        return {
            "recent_files": [],
            "max_recent_files": 10,
            "default_output_format": "txt_full",
            "batch_max_workers": 4,
            "z1d_array_size": 16384,
            "create_subfolders": True,
            "window_geometry": "1000x800",
            "peak_finding": {
                "method": "local_maximum",
                "min_height": 10,
                "min_distance": 1,
                "background_percentile": 10
            },
            "plotting": {
                "default_style": "line",
                "show_peaks": True,
                "show_grid": True,
                "figure_size": [10, 6],
                "dpi": 100
            }
        }
    
    def load_config(self) -> None:
        """Load configuration from file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    saved_config = json.load(f)
                    self.config.update(saved_config)
            except Exception as e:
                print(f"Warning: Could not load config: {e}")
    
    def save_config(self) -> None:
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save config: {e}")
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def add_recent_file(self, file_path: str) -> None:
        """Add file to recent files list"""
        recent = self.config["recent_files"]
        if file_path in recent:
            recent.remove(file_path)
        recent.insert(0, file_path)
        
        max_recent = self.config["max_recent_files"]
        self.config["recent_files"] = recent[:max_recent]
        self.save_config()

class ProgressDialog:
    """Modal progress dialog with cancellation support"""
    
    def __init__(self, parent, title: str = "Processing"):
        self.parent = parent
        self.cancelled = False
        
        # Create dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.geometry("400x150")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Center on parent
        self.dialog.geometry(f"+{parent.winfo_rootx() + 50}+{parent.winfo_rooty() + 50}")
        
        # Create widgets
        self._create_widgets()
        
        # Handle close
        self.dialog.protocol("WM_DELETE_WINDOW", self.cancel)
    
    def _create_widgets(self):
        """Create progress dialog widgets"""
        main_frame = ttk.Frame(self.dialog, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="Initializing...")
        self.status_label.pack(pady=(0, 10))
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(main_frame, mode='determinate', length=350)
        self.progress_bar.pack(pady=(0, 10))
        
        # Progress percentage
        self.percent_label = ttk.Label(main_frame, text="0%")
        self.percent_label.pack(pady=(0, 10))
        
        # Cancel button
        self.cancel_button = ttk.Button(main_frame, text="Cancel", command=self.cancel)
        self.cancel_button.pack()
    
    def update_progress(self, progress: float, message: str = ""):
        """Update progress (0-100)"""
        if not self.cancelled:
            self.progress_bar['value'] = progress
            self.percent_label.config(text=f"{int(progress)}%")
            if message:
                self.status_label.config(text=message)
            self.dialog.update_idletasks()
    
    def cancel(self):
        """Cancel operation"""
        self.cancelled = True
        self.dialog.destroy()
    
    def close(self):
        """Close dialog"""
        if not self.cancelled:
            self.dialog.destroy()

class ConversionTab:
    """Handles the conversion tab interface"""
    
    def __init__(self, parent_notebook, config: ConfigManager, 
                 file_reader: SpectrumFileReader, 
                 conversion_engine: ConversionEngine):
        self.config = config
        self.file_reader = file_reader
        self.conversion_engine = conversion_engine
        
        # Create tab
        self.frame = ttk.Frame(parent_notebook)
        parent_notebook.add(self.frame, text="File Conversion")
        
        # Variables
        self.selected_file = tk.StringVar()
        self.selected_folder = tk.StringVar()
        self.conversion_mode = tk.StringVar(value="single")
        self.output_format = tk.StringVar(value=config.get("default_output_format"))
        self.batch_format = tk.StringVar(value="z1d")
        self.z1d_array_size = tk.StringVar(value=str(config.get("z1d_array_size")))
        self.create_subfolders = tk.BooleanVar(value=config.get("create_subfolders"))
        
        # Create interface
        self._create_widgets()
        self._load_recent_files()
        
        # Bind events
        self.conversion_mode.trace_add("write", self._on_mode_change)
    
    def _create_widgets(self):
        """Create conversion tab widgets"""
        main_frame = ttk.Frame(self.frame, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Spectrum File Converter", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Mode selection
        mode_frame = ttk.LabelFrame(main_frame, text="Conversion Mode", padding="10")
        mode_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Radiobutton(mode_frame, text="Single File", 
                       variable=self.conversion_mode, value="single").pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(mode_frame, text="Batch Folder", 
                       variable=self.conversion_mode, value="batch").pack(side=tk.LEFT)
        
        # File selection frame
        self.file_frame = ttk.LabelFrame(main_frame, text="File Selection", padding="10")
        self.file_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Single file widgets
        self.file_button = ttk.Button(self.file_frame, text="Select File", 
                                     command=self._select_file)
        self.file_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.file_label = ttk.Label(self.file_frame, text="No file selected", 
                                   foreground="gray")
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Batch folder widgets (initially hidden)
        self.folder_button = ttk.Button(self.file_frame, text="Select Folder", 
                                       command=self._select_folder)
        self.folder_label = ttk.Label(self.file_frame, text="No folder selected", 
                                     foreground="gray")
        
        # Format selection
        self.format_frame = ttk.LabelFrame(main_frame, text="Output Format", padding="10")
        self.format_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Get available formats
        formats = self.conversion_engine.get_available_formats()
        for key, name in formats.items():
            ttk.Radiobutton(self.format_frame, text=name, 
                           variable=self.output_format, value=key).pack(side=tk.LEFT, padx=(0, 15))
        
        # Batch options (initially hidden)
        self.batch_frame = ttk.LabelFrame(main_frame, text="Batch Options", padding="10")
        
        # Z1D array size
        array_frame = ttk.Frame(self.batch_frame)
        array_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(array_frame, text="Z1D Array Size:").pack(side=tk.LEFT)
        ttk.Entry(array_frame, textvariable=self.z1d_array_size, width=10).pack(side=tk.LEFT, padx=(5, 0))
        
        # Subfolder option
        ttk.Checkbutton(self.batch_frame, text="Create organized output subfolders", 
                       variable=self.create_subfolders).pack(anchor=tk.W)
        
        # Recent files
        recent_frame = ttk.LabelFrame(main_frame, text="Recent Files", padding="10")
        recent_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.recent_combo = ttk.Combobox(recent_frame, state="readonly", width=80)
        self.recent_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.recent_combo.bind("<<ComboboxSelected>>", self._on_recent_selected)
        
        ttk.Button(recent_frame, text="Clear History", 
                  command=self._clear_recent).pack(side=tk.RIGHT)
        
        # Convert button
        self.convert_button = ttk.Button(main_frame, text="CONVERT", 
                                        command=self._start_conversion, 
                                        state="disabled",
                                        style="Accent.TButton")
        self.convert_button.pack(pady=(20, 0))
        
        # Status
        self.status_label = ttk.Label(main_frame, text="Ready", foreground="green")
        self.status_label.pack(pady=(10, 0))
    
    def _load_recent_files(self):
        """Load recent files into combobox"""
        recent_files = self.config.get("recent_files", [])
        if recent_files:
            # Show just filenames but store full paths
            display_names = [f"{Path(f).name} ({f})" for f in recent_files if Path(f).exists()]
            self.recent_combo['values'] = display_names
            if display_names:
                self.recent_combo.set(display_names[0])
        else:
            self.recent_combo['values'] = ["No recent files"]
    
    def _on_mode_change(self, *args):
        """Handle conversion mode change"""
        mode = self.conversion_mode.get()
        
        if mode == "single":
            # Show single file widgets
            self.file_button.pack(side=tk.LEFT, padx=(0, 10))
            self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Hide batch widgets
            self.folder_button.pack_forget()
            self.folder_label.pack_forget()
            self.batch_frame.pack_forget()
            
            # Show format frame
            self.format_frame.pack(fill=tk.X, pady=(0, 15), before=self.file_frame)
        else:
            # Hide single file widgets
            self.file_button.pack_forget()
            self.file_label.pack_forget()
            self.format_frame.pack_forget()
            
            # Show batch widgets
            self.folder_button.pack(side=tk.LEFT, padx=(0, 10))
            self.folder_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.batch_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Reset selection and button state
        self.selected_file.set("")
        self.selected_folder.set("")
        self.convert_button.config(state="disabled")
        self._update_status("Ready")
    
    def _select_file(self):
        """Select single file for conversion"""
        file_path = filedialog.askopenfilename(
            title="Select Spectrum File",
            filetypes=[
                ("Spectrum files", "*.spe *.cnf"),
                ("SPE files", "*.spe"),
                ("CNF files", "*.cnf"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.selected_file.set(file_path)
            filename = Path(file_path).name
            self.file_label.config(text=filename, foreground="black")
            self.convert_button.config(state="normal")
            self._update_status(f"Selected: {filename}")
            
            # Add to recent files
            self.config.add_recent_file(file_path)
            self._load_recent_files()
    
    def _select_folder(self):
        """Select folder for batch conversion"""
        folder_path = filedialog.askdirectory(title="Select Folder Containing Spectrum Files")
        
        if folder_path:
            # Count compatible files
            folder = Path(folder_path)
            file_count = 0
            for file_path in folder.iterdir():
                if file_path.is_file() and self.file_reader.can_read_file(file_path):
                    file_count += 1
            
            if file_count > 0:
                self.selected_folder.set(folder_path)
                self.folder_label.config(text=f"{folder.name} ({file_count} files)", 
                                       foreground="black")
                self.convert_button.config(state="normal")
                self._update_status(f"Selected: {file_count} compatible files")
            else:
                self._update_status("No compatible files found", "red")
    
    def _on_recent_selected(self, event):
        """Handle recent file selection"""
        selection = self.recent_combo.get()
        if selection and selection != "No recent files":
            # Extract file path from display string
            file_path = selection.split(" (")[1].rstrip(")")
            
            if Path(file_path).exists():
                self.selected_file.set(file_path)
                filename = Path(file_path).name
                self.file_label.config(text=filename, foreground="black")
                self.convert_button.config(state="normal")
                self._update_status(f"Selected from recent: {filename}")
            else:
                self._update_status("File no longer exists", "red")
    
    def _clear_recent(self):
        """Clear recent files history"""
        if messagebox.askyesno("Clear History", "Clear all recent files?"):
            self.config.set("recent_files", [])
            self.config.save_config()
            self.recent_combo['values'] = ["No recent files"]
            self.recent_combo.set("No recent files")
            self._update_status("Recent files cleared")
    
    def _start_conversion(self):
        """Start conversion process"""
        if self.conversion_mode.get() == "single":
            self._convert_single_file()
        else:
            self._convert_batch_files()
    
    def _convert_single_file(self):
        """Convert single file"""
        file_path = self.selected_file.get()
        if not file_path:
            return
        
        # Create conversion job
        job = self.conversion_engine.create_job(
            source_file=file_path,
            target_format=self.output_format.get(),
            array_size=int(self.z1d_array_size.get()) if self.z1d_array_size.get().isdigit() else 16384
        )
        
        # Create progress dialog
        progress_dialog = ProgressDialog(self.frame.winfo_toplevel(), "Converting File")
        
        def progress_callback(progress: float, message: str):
            if not progress_dialog.cancelled:
                progress_dialog.update_progress(progress, message)
        
        def conversion_thread():
            try:
                result = self.conversion_engine.convert_single(job, progress_callback)
                
                if not progress_dialog.cancelled:
                    # Update UI in main thread
                    self.frame.after(0, lambda: self._conversion_complete(result, progress_dialog))
            except Exception as e:
                if not progress_dialog.cancelled:
                    self.frame.after(0, lambda: self._conversion_error(str(e), progress_dialog))
        
        # Start conversion in background
        thread = threading.Thread(target=conversion_thread, daemon=True)
        thread.start()
    
    def _convert_batch_files(self):
        """Convert batch files"""
        folder_path = self.selected_folder.get()
        if not folder_path:
            return
        
        # Create batch manager
        batch_manager = BatchConversionManager(
            self.conversion_engine, 
            max_workers=self.config.get("batch_max_workers", 4)
        )
        
        # Find files and create jobs
        folder = Path(folder_path)
        files = batch_manager.find_convertible_files(folder, self.batch_format.get())
        
        if not files:
            self._update_status("No convertible files found", "red")
            return
        
        jobs = batch_manager.create_batch_jobs(
            files=files,
            target_format=self.batch_format.get(),
            create_subfolders=self.create_subfolders.get(),
            array_size=int(self.z1d_array_size.get()) if self.z1d_array_size.get().isdigit() else 16384
        )
        
        # Create progress dialog
        progress_dialog = ProgressDialog(self.frame.winfo_toplevel(), "Batch Conversion")
        
        def progress_callback(progress: float, message: str):
            if not progress_dialog.cancelled:
                progress_dialog.update_progress(progress, message)
        
        def conversion_thread():
            try:
                results = batch_manager.convert_batch(jobs, progress_callback)
                
                if not progress_dialog.cancelled:
                    self.frame.after(0, lambda: self._batch_conversion_complete(results, progress_dialog))
            except Exception as e:
                if not progress_dialog.cancelled:
                    self.frame.after(0, lambda: self._conversion_error(str(e), progress_dialog))
        
        # Start conversion in background
        thread = threading.Thread(target=conversion_thread, daemon=True)
        thread.start()
    
    def _conversion_complete(self, result, progress_dialog: ProgressDialog):
        """Handle single conversion completion"""
        progress_dialog.close()
        
        if result.success:
            self._update_status(f"Conversion completed: {result.output_file.name}", "green")
            messagebox.showinfo("Success", 
                              f"File converted successfully!\n\n"
                              f"Output: {result.output_file.name}\n"
                              f"Time: {result.processing_time:.2f} seconds")
        else:
            self._update_status("Conversion failed", "red")
            messagebox.showerror("Error", f"Conversion failed:\n\n{result.error_message}")
    
    def _batch_conversion_complete(self, results, progress_dialog: ProgressDialog):
        """Handle batch conversion completion"""
        progress_dialog.close()
        
        successful = sum(1 for r in results if r.success)
        total = len(results)
        
        self._update_status(f"Batch complete: {successful}/{total} files", "green")
        
        # Show detailed results
        if successful == total:
            messagebox.showinfo("Batch Complete", 
                              f"All {total} files converted successfully!")
        else:
            failed_files = [r.job.source_file.name for r in results if not r.success]
            messagebox.showwarning("Batch Complete", 
                                 f"Converted {successful} of {total} files.\n\n"
                                 f"Failed files:\n" + "\n".join(failed_files[:5]) +
                                 (f"\n... and {len(failed_files) - 5} more" if len(failed_files) > 5 else ""))
    
    def _conversion_error(self, error_message: str, progress_dialog: ProgressDialog):
        """Handle conversion error"""
        progress_dialog.close()
        self._update_status("Conversion failed", "red")
        messagebox.showerror("Error", f"Conversion failed:\n\n{error_message}")
    
    def _update_status(self, message: str, color: str = "black"):
        """Update status label"""
        self.status_label.config(text=message, foreground=color)

class AnalysisTab:
    """Handles the analysis tab interface"""
    
    def __init__(self, parent_notebook, config: ConfigManager,
                 file_reader: SpectrumFileReader,
                 analyzer: SpectrumAnalyzer):
        self.config = config
        self.file_reader = file_reader
        self.analyzer = analyzer
        self.current_spectrum: Optional[SpectrumData] = None
        self.current_analysis = None
        
        # Create tab
        self.frame = ttk.Frame(parent_notebook)
        parent_notebook.add(self.frame, text="Spectrum Analysis")
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create analysis tab widgets"""
        main_frame = ttk.Frame(self.frame, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Spectrum Analysis", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Button(control_frame, text="Load Spectrum", 
                  command=self._load_spectrum).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="Analyze", 
                  command=self._analyze_spectrum).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(control_frame, text="Export Report", 
                  command=self._export_report).pack(side=tk.LEFT, padx=(0, 10))
        
        # Create paned window
        paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Information
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        
        # File info
        info_frame = ttk.LabelFrame(left_frame, text="File Information", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.info_text = tk.Text(info_frame, height=10, wrap=tk.WORD)
        info_scroll = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.info_text.yview)
        self.info_text.configure(yscrollcommand=info_scroll.set)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        info_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Statistics
        stats_frame = ttk.LabelFrame(left_frame, text="Statistics", padding="10")
        stats_frame.pack(fill=tk.BOTH, expand=True)
        
        self.stats_text = tk.Text(stats_frame, height=10, wrap=tk.WORD)
        stats_scroll = ttk.Scrollbar(stats_frame, orient=tk.VERTICAL, command=self.stats_text.yview)
        self.stats_text.configure(yscrollcommand=stats_scroll.set)
        self.stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        stats_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Right panel - Peak list
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=1)
        
        peaks_frame = ttk.LabelFrame(right_frame, text="Detected Peaks", padding="10")
        peaks_frame.pack(fill=tk.BOTH, expand=True)
        
        # Peaks treeview
        columns = ("Rank", "Channel", "Counts", "Energy", "FWHM", "Significance")
        self.peaks_tree = ttk.Treeview(peaks_frame, columns=columns, show="headings", height=20)
        
        for col in columns:
            self.peaks_tree.heading(col, text=col)
            self.peaks_tree.column(col, width=80, anchor=tk.CENTER)
        
        peaks_scroll = ttk.Scrollbar(peaks_frame, orient=tk.VERTICAL, command=self.peaks_tree.yview)
        self.peaks_tree.configure(yscrollcommand=peaks_scroll.set)
        
        self.peaks_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        peaks_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Initialize with empty message
        self._clear_analysis()
    
    def _load_spectrum(self):
        """Load spectrum file for analysis"""
        file_path = filedialog.askopenfilename(
            title="Select Spectrum File for Analysis",
            filetypes=[
                ("Spectrum files", "*.spe *.cnf"),
                ("SPE files", "*.spe"),
                ("CNF files", "*.cnf"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.current_spectrum = self.file_reader.read_file(file_path)
                self._display_file_info()
                self.config.add_recent_file(file_path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load spectrum:\n\n{str(e)}")
    
    def _display_file_info(self):
        """Display loaded file information"""
        if not self.current_spectrum:
            return
        
        self.info_text.delete(1.0, tk.END)
        
        # File information
        meta = self.current_spectrum.metadata
        self.info_text.insert(tk.END, "FILE INFORMATION\n")
        self.info_text.insert(tk.END, "=" * 40 + "\n\n")
        
        if meta.source_file:
            self.info_text.insert(tk.END, f"File: {meta.source_file.name}\n")
            self.info_text.insert(tk.END, f"Path: {meta.source_file}\n")
        
        if meta.file_format:
            self.info_text.insert(tk.END, f"Format: {meta.file_format.value.upper()}\n")
        
        if meta.spec_id:
            self.info_text.insert(tk.END, f"Spectrum ID: {meta.spec_id}\n")
        
        if meta.date:
            self.info_text.insert(tk.END, f"Date: {meta.date}\n")
        
        if meta.device:
            self.info_text.insert(tk.END, f"Device: {meta.device}\n")
        
        if meta.live_time and meta.real_time:
            self.info_text.insert(tk.END, f"Live Time: {meta.live_time:.1f} s\n")
            self.info_text.insert(tk.END, f"Real Time: {meta.real_time:.1f} s\n")
            if meta.dead_time_percent:
                self.info_text.insert(tk.END, f"Dead Time: {meta.dead_time_percent:.2f}%\n")
        
        if meta.has_energy_calibration:
            self.info_text.insert(tk.END, f"\nENERGY CALIBRATION:\n")
            params = meta.calibration_params
            if len(params) == 2:
                self.info_text.insert(tk.END, f"E = {params[0]:.6f} + {params[1]:.6f} * Ch\n")
            elif len(params) >= 3:
                self.info_text.insert(tk.END, f"E = {params[0]:.6f} + {params[1]:.6f} * Ch + {params[2]:.6f} * Ch²\n")
        
        # Basic spectrum info
        self.info_text.insert(tk.END, f"\nSPECTRUM DATA:\n")
        self.info_text.insert(tk.END, f"Channels: {self.current_spectrum.channel_count:,}\n")
        self.info_text.insert(tk.END, f"Total Counts: {self.current_spectrum.total_counts:,}\n")
        self.info_text.insert(tk.END, f"Peak Channel: {self.current_spectrum.peak_channel}\n")
        self.info_text.insert(tk.END, f"Peak Counts: {self.current_spectrum.peak_counts:,}\n")
        
        if self.current_spectrum.count_rate:
            self.info_text.insert(tk.END, f"Count Rate: {self.current_spectrum.count_rate:.2f} cps\n")
    
    def _analyze_spectrum(self):
        """Perform spectrum analysis"""
        if not self.current_spectrum:
            messagebox.showwarning("No Data", "Please load a spectrum file first.")
            return
        
        try:
            # Get analysis parameters from config
            params = {
                'peak_min_height': self.config.get('peak_finding.min_height', 10),
                'peak_min_distance': self.config.get('peak_finding.min_distance', 1),
                'background_percentile': self.config.get('peak_finding.background_percentile', 10)
            }
            
            # Perform analysis
            method = PeakSearchMethod.LOCAL_MAXIMUM
            self.current_analysis = self.analyzer.analyze(self.current_spectrum, method, **params)
            
            self._display_analysis_results()
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Failed to analyze spectrum:\n\n{str(e)}")
    
    def _display_analysis_results(self):
        """Display analysis results"""
        if not self.current_analysis:
            return
        
        # Display statistics
        self.stats_text.delete(1.0, tk.END)
        
        stats = self.current_analysis.statistics
        self.stats_text.insert(tk.END, "ANALYSIS RESULTS\n")
        self.stats_text.insert(tk.END, "=" * 30 + "\n\n")
        
        self.stats_text.insert(tk.END, f"Total Counts: {stats.total_counts:,}\n")
        self.stats_text.insert(tk.END, f"Peak Counts: {stats.peak_counts:,}\n")
        self.stats_text.insert(tk.END, f"Mean Channel: {stats.mean_channel:.2f}\n")
        self.stats_text.insert(tk.END, f"Std Deviation: {stats.std_deviation:.2f}\n")
        
        if stats.background_level:
            self.stats_text.insert(tk.END, f"Background Level: {stats.background_level:.2f}\n")
        
        if stats.noise_level:
            self.stats_text.insert(tk.END, f"Noise Level: {stats.noise_level:.2f}\n")
        
        if stats.signal_to_noise:
            self.stats_text.insert(tk.END, f"Signal/Noise: {stats.signal_to_noise:.2f}\n")
        
        if stats.count_rate:
            self.stats_text.insert(tk.END, f"Count Rate: {stats.count_rate:.2f} cps\n")
        
        if stats.mean_energy:
            self.stats_text.insert(tk.END, f"Mean Energy: {stats.mean_energy:.2f} keV\n")
        
        self.stats_text.insert(tk.END, f"\nPeaks Found: {len(self.current_analysis.peaks)}\n")
        
        # Display peaks
        self._display_peaks()
    
    def _display_peaks(self):
        """Display detected peaks in treeview"""
        # Clear existing items
        for item in self.peaks_tree.get_children():
            self.peaks_tree.delete(item)
        
        if not self.current_analysis or not self.current_analysis.peaks:
            return
        
        # Add peaks to treeview
        for i, peak in enumerate(self.current_analysis.peaks, 1):
            energy_str = f"{peak.energy:.2f}" if peak.energy else "N/A"
            fwhm_str = f"{peak.fwhm:.2f}" if peak.fwhm else "N/A"
            sig_str = f"{peak.significance:.2f}" if peak.significance else "N/A"
            
            self.peaks_tree.insert("", tk.END, values=(
                i, peak.channel, f"{peak.counts:,}", energy_str, fwhm_str, sig_str
            ))
    
    def _export_report(self):
        """Export analysis report"""
        if not self.current_analysis:
            messagebox.showwarning("No Analysis", "Please analyze a spectrum first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Export Analysis Report",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.analyzer.export_analysis_report(self.current_analysis, file_path)
                messagebox.showinfo("Success", f"Analysis report exported to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export report:\n\n{str(e)}")
    
    def _clear_analysis(self):
        """Clear analysis display"""
        self.info_text.delete(1.0, tk.END)
        self.stats_text.delete(1.0, tk.END)
        for item in self.peaks_tree.get_children():
            self.peaks_tree.delete(item)
        
        self.info_text.insert(tk.END, "No spectrum loaded.\n\nUse 'Load Spectrum' to begin analysis.")
        self.stats_text.insert(tk.END, "Analysis results will appear here.")
    
    def load_spectrum_from_file(self, file_path: str):
        """Load spectrum from external source (e.g., conversion tab)"""
        try:
            self.current_spectrum = self.file_reader.read_file(file_path)
            self._display_file_info()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load spectrum:\n\n{str(e)}")

class PlottingTab:
    """Handles the plotting tab interface"""
    
    def __init__(self, parent_notebook, config: ConfigManager,
                 file_reader: SpectrumFileReader,
                 plotter: SpectrumPlotter):
        self.config = config
        self.file_reader = file_reader
        self.plotter = plotter
        self.current_spectrum: Optional[SpectrumData] = None
        self.current_analysis = None
        self.current_figure = None
        
        # Create tab
        self.frame = ttk.Frame(parent_notebook)
        parent_notebook.add(self.frame, text="Spectrum Plotting")
        
        # Check if matplotlib GUI is available
        self.gui_available = MATPLOTLIB_GUI_AVAILABLE and plotter.is_available()
        
        self._create_widgets()
    
    def _create_widgets(self):
        """Create plotting tab widgets"""
        if not self.gui_available:
            # Show unavailable message
            main_frame = ttk.Frame(self.frame, padding="20")
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            ttk.Label(main_frame, text="Plotting Not Available", 
                     font=("Arial", 16, "bold")).pack(pady=(50, 20))
            ttk.Label(main_frame, text="Matplotlib and/or tkinter backend not available.\n\n"
                                      "Install matplotlib with: pip install matplotlib",
                     justify=tk.CENTER).pack()
            return
        
        main_frame = ttk.Frame(self.frame, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(main_frame, text="Spectrum Plotting", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))
        
        # Control panel
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Left controls
        left_controls = ttk.Frame(control_frame)
        left_controls.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # File controls
        file_frame = ttk.Frame(left_controls)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="Load File", 
                  command=self._load_file).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(file_frame, text="Clear Plot", 
                  command=self._clear_plot).pack(side=tk.LEFT, padx=(0, 10))
        
        # Plot options
        options_frame = ttk.LabelFrame(left_controls, text="Plot Options", padding="10")
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Scale options
        scale_frame = ttk.Frame(options_frame)
        scale_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(scale_frame, text="Y-Scale:").pack(side=tk.LEFT, padx=(0, 5))
        self.y_scale_var = tk.StringVar(value="linear")
        ttk.Radiobutton(scale_frame, text="Linear", variable=self.y_scale_var, 
                       value="linear", command=self._update_plot).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(scale_frame, text="Log", variable=self.y_scale_var, 
                       value="log", command=self._update_plot).pack(side=tk.LEFT)
        
        # Axis options
        axis_frame = ttk.Frame(options_frame)
        axis_frame.pack(fill=tk.X, pady=(5, 5))
        
        ttk.Label(axis_frame, text="X-Axis:").pack(side=tk.LEFT, padx=(0, 5))
        self.x_axis_var = tk.StringVar(value="channel")
        ttk.Radiobutton(axis_frame, text="Channel", variable=self.x_axis_var, 
                       value="channel", command=self._update_plot).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(axis_frame, text="Energy", variable=self.x_axis_var, 
                       value="energy", command=self._update_plot).pack(side=tk.LEFT)
        
        # Display options
        display_frame = ttk.Frame(options_frame)
        display_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.show_peaks_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="Show Peaks", variable=self.show_peaks_var,
                       command=self._update_plot).pack(side=tk.LEFT, padx=(0, 10))
        
        self.show_grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(display_frame, text="Show Grid", variable=self.show_grid_var,
                       command=self._update_plot).pack(side=tk.LEFT, padx=(0, 10))
        
        # Right controls - action buttons
        right_controls = ttk.Frame(control_frame)
        right_controls.pack(side=tk.RIGHT)
        
        ttk.Button(right_controls, text="Auto Scale", 
                  command=self._auto_scale).pack(pady=(0, 5), fill=tk.X)
        ttk.Button(right_controls, text="Save Plot", 
                  command=self._save_plot).pack(pady=(0, 5), fill=tk.X)
        
        # Create matplotlib figure and canvas
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            
            self.fig = Figure(figsize=(10, 6), dpi=100)
            self.ax = self.fig.add_subplot(111)
            
            # Canvas frame
            canvas_frame = ttk.Frame(main_frame)
            canvas_frame.pack(fill=tk.BOTH, expand=True)
            
            self.canvas = FigureCanvasTkAgg(self.fig, canvas_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Navigation toolbar
            toolbar_frame = ttk.Frame(canvas_frame)
            toolbar_frame.pack(fill=tk.X)
            self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
            self.toolbar.update()
            
            # Initialize empty plot
            self._clear_plot()
            
        except Exception as e:
            ttk.Label(main_frame, text=f"Error initializing plot: {e}").pack()
    
    def _load_file(self):
        """Load spectrum file for plotting"""
        file_path = filedialog.askopenfilename(
            title="Select Spectrum File for Plotting",
            filetypes=[
                ("Spectrum files", "*.spe *.cnf"),
                ("SPE files", "*.spe"),
                ("CNF files", "*.cnf"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.current_spectrum = self.file_reader.read_file(file_path)
                self._plot_spectrum()
                self.config.add_recent_file(file_path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load spectrum:\n\n{str(e)}")
    
    def _plot_spectrum(self):
        """Plot the current spectrum"""
        if not self.current_spectrum or not self.gui_available:
            return
        
        # Create plot options
        options = PlotOptions(
            title=f"Spectrum: {self.current_spectrum.metadata.source_file.name if self.current_spectrum.metadata.source_file else 'Unknown'}",
            y_scale=self.y_scale_var.get(),
            use_energy_axis=(self.x_axis_var.get() == "energy"),
            show_peaks=self.show_peaks_var.get(),
            show_grid=self.show_grid_var.get(),
            show_statistics=True
        )
        
        try:
            # Use the plotting engine
            self.current_figure = self.plotter.create_custom_plot(
                self.current_spectrum, 
                self.current_analysis,
                options
            )
            
            # Copy to our canvas
            if self.current_figure:
                self.ax.clear()
                self.ax = self.current_figure.axes[0]
                self.canvas.figure = self.current_figure
                self.canvas.draw()
                
        except Exception as e:
            messagebox.showerror("Plot Error", f"Failed to create plot:\n\n{str(e)}")
    
    def _update_plot(self):
        """Update the current plot with new options"""
        if self.current_spectrum:
            self._plot_spectrum()
    
    def _auto_scale(self):
        """Auto-scale the plot"""
        if self.ax:
            self.ax.relim()
            self.ax.autoscale()
            self.canvas.draw()
    
    def _save_plot(self):
        """Save the current plot"""
        if not self.current_figure:
            messagebox.showwarning("No Plot", "No plot to save")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Plot",
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.plotter.save_plot(self.current_figure, file_path, dpi=300)
                messagebox.showinfo("Success", f"Plot saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save plot:\n\n{str(e)}")
    
    def _clear_plot(self):
        """Clear the plot"""
        if not self.gui_available:
            return
        
        self.ax.clear()
        self.ax.set_xlabel("Channel / Energy (keV)")
        self.ax.set_ylabel("Counts")
        self.ax.set_title("Spectrum Plot")
        self.ax.text(0.5, 0.5, "Load a spectrum file to begin plotting", 
                    transform=self.ax.transAxes, ha='center', va='center',
                    fontsize=12, alpha=0.6)
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()
        
        self.current_spectrum = None
        self.current_analysis = None
        self.current_figure = None
    
    def load_spectrum_from_data(self, spectrum: SpectrumData, analysis=None):
        """Load spectrum from external source with optional analysis"""
        self.current_spectrum = spectrum
        self.current_analysis = analysis
        if self.gui_available:
            self._plot_spectrum()

class SpectrumConverterApp:
    """Main application class"""
    
    def __init__(self):
        # Initialize configuration
        self.config = ConfigManager()
        
        # Initialize engines
        self.file_reader = SpectrumFileReader()
        self.conversion_engine = ConversionEngine()
        self.analyzer = SpectrumAnalyzer()
        self.plotter = SpectrumPlotter()
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Spectrum Converter Pro - Analysis & Conversion Suite")
        
        # Set window geometry from config
        geometry = self.config.get("window_geometry", "1000x800")
        self.root.geometry(geometry)
        self.root.resizable(True, True)
        
        # Configure style
        self._configure_style()
        
        # Create GUI
        self._create_widgets()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
    
    def _configure_style(self):
        """Configure application style"""
        style = ttk.Style()
        
        # Try to use a modern theme
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
        elif 'alt' in available_themes:
            style.theme_use('alt')
        
        # Configure accent button style
        style.configure("Accent.TButton", font=("Arial", 10, "bold"))
    
    def _create_widgets(self):
        """Create main application widgets"""
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.conversion_tab = ConversionTab(
            self.notebook, self.config, self.file_reader, self.conversion_engine
        )
        
        self.analysis_tab = AnalysisTab(
            self.notebook, self.config, self.file_reader, self.analyzer
        )
        
        self.plotting_tab = PlottingTab(
            self.notebook, self.config, self.file_reader, self.plotter
        )
        
        # Create status bar
        self.status_bar = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, pady=(5, 0))
        
        # Create menu bar
        self._create_menu()
    
    def _create_menu(self):
        """Create application menu"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Spectrum...", command=self._open_spectrum)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_closing)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Batch Conversion...", command=self._focus_conversion)
        tools_menu.add_command(label="Spectrum Analysis...", command=self._focus_analysis)
        tools_menu.add_command(label="Plot Spectrum...", command=self._focus_plotting)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
    
    def _open_spectrum(self):
        """Open spectrum file and load into current tab"""
        file_path = filedialog.askopenfilename(
            title="Open Spectrum File",
            filetypes=[
                ("Spectrum files", "*.spe *.cnf"),
                ("SPE files", "*.spe"),
                ("CNF files", "*.cnf"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            # Load into current tab
            current_tab = self.notebook.tab(self.notebook.select(), "text")
            
            if "Analysis" in current_tab:
                self.analysis_tab.load_spectrum_from_file(file_path)
            elif "Plotting" in current_tab:
                try:
                    spectrum = self.file_reader.read_file(file_path)
                    self.plotting_tab.load_spectrum_from_data(spectrum)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to load spectrum:\n\n{str(e)}")
    
    def _focus_conversion(self):
        """Switch to conversion tab"""
        self.notebook.select(0)
    
    def _focus_analysis(self):
        """Switch to analysis tab"""
        self.notebook.select(1)
    
    def _focus_plotting(self):
        """Switch to plotting tab"""
        self.notebook.select(2)
    
    def _show_about(self):
        """Show about dialog"""
        about_text = ("Spectrum Converter Pro v2.0\n\n"
                     "A comprehensive tool for gamma-ray spectrum analysis.\n\n"
                     "Features:\n"
                     "• Multi-format file conversion (SPE, CNF, TXT, Z1D)\n"
                     "• Advanced spectrum analysis with peak detection\n"
                     "• High-quality plotting and visualization\n"
                     "• Batch processing capabilities\n\n"
                     "Built with modern software engineering practices.")
        
        messagebox.showinfo("About Spectrum Converter Pro", about_text)
    
    def _on_closing(self):
        """Handle application shutdown"""
        # Save window geometry
        self.config.set("window_geometry", self.root.geometry())
        self.config.save_config()
        
        # Close application
        self.root.destroy()
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

def main():
    """Main entry point"""
    try:
        app = SpectrumConverterApp()
        app.run()
    except Exception as e:
        # Show error dialog if GUI fails to start
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Startup Error", 
                           f"Failed to start application:\n\n{str(e)}\n\n"
                           f"Please check that all dependencies are installed.")
        root.destroy()

if __name__ == "__main__":
    main()
