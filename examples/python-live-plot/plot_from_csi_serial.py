#!/usr/bin/env python3
# -*-coding:utf-8-*-

import sys
import csv
import json
import argparse
import numpy as np
import serial
from io import StringIO
from collections import deque
from typing import Optional, Tuple, List, Callable

from PyQt5.Qt import *
from PyQt5.QtCore import pyqtSignal, QThread, QTimer
import pyqtgraph as pg
from pyqtgraph import PlotWidget, ScatterPlotItem

# Constants
CSI_DATA_INDEX = 200  # buffer size
CSI_DATA_COLUMNS = 490

DATA_COLUMNS_NAMES_C5C6 = [
    "type", "id", "mac", "rssi", "rate", "noise_floor", "fft_gain", 
    "agc_gain", "channel", "local_timestamp", "sig_len", "rx_state", 
    "len", "first_word", "data"
]

DATA_COLUMNS_NAMES = [
    "type", "id", "mac", "rssi", "rate", "sig_mode", "mcs", "bandwidth", 
    "smoothing", "not_sounding", "aggregation", "stbc", "fec_coding",
    "sgi", "noise_floor", "ampdu_cnt", "channel", "secondary_channel", 
    "local_timestamp", "ant", "sig_len", "rx_state", "len", "first_word", "data"
]

# Global data buffers
class CSIDataBuffer:
    def __init__(self):
        self.csi_data_complex = np.zeros([CSI_DATA_INDEX, CSI_DATA_COLUMNS], dtype=np.complex64)
        self.agc_gain_data = np.zeros([CSI_DATA_INDEX], dtype=np.float64)
        self.fft_gain_data = np.zeros([CSI_DATA_INDEX], dtype=np.float64)
        self.fft_gains = deque(maxlen=1000)
        self.agc_gains = deque(maxlen=1000)
    
    def update_buffers(self, csi_complex_data: np.ndarray, agc_gain: float, fft_gain: float):
        """Efficiently update circular buffers"""
        self.csi_data_complex[:-1] = self.csi_data_complex[1:]
        self.csi_data_complex[-1] = csi_complex_data
        
        self.agc_gain_data[:-1] = self.agc_gain_data[1:]
        self.agc_gain_data[-1] = agc_gain
        
        self.fft_gain_data[:-1] = self.fft_gain_data[1:]
        self.fft_gain_data[-1] = fft_gain
        
        self.fft_gains.append(fft_gain)
        self.agc_gains.append(agc_gain)

csi_buffer = CSIDataBuffer()

class ZoomablePlotWidget(PlotWidget):
    """Custom PlotWidget với khả năng phóng to thu nhỏ tốt hơn"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_zoom_behavior()
        
    def setup_zoom_behavior(self):
        """Thiết lập hành vi phóng to thu nhỏ"""
        # Enable mouse scaling and dragging
        self.getPlotItem().getViewBox().setMouseMode(pg.ViewBox.RectMode)
        
        # Add zoom reset button
        self.reset_zoom_action = QAction("Reset Zoom", self)
        self.reset_zoom_action.triggered.connect(self.reset_zoom)
        self.addAction(self.reset_zoom_action)
        self.setContextMenuPolicy(Qt.ActionsContextMenu)
        
    def reset_zoom(self):
        """Reset zoom về mặc định"""
        self.getPlotItem().getViewBox().autoRange()

class CSIDataGraphicalWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_plots()
        self.setup_timer()
        self.visible_subcarriers = 0

    def setup_ui(self):
        """Initialize UI components"""
        self.resize(1280, 900)
        
        # Tạo layout chính với khả năng resize
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # Tạo splitter để có thể điều chỉnh kích thước các plot
        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)
        
        # Top row: Phase và IQ plots
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        self.plotWidget_phase = ZoomablePlotWidget()
        self.plotWidget_iq = ZoomablePlotWidget()
        
        top_layout.addWidget(self.plotWidget_phase)
        top_layout.addWidget(self.plotWidget_iq)
        
        # Middle: Amplitude plot
        self.plotWidget_amplitude = ZoomablePlotWidget()
        
        # Bottom: Phase time plot
        self.plotWidget_phase_time = ZoomablePlotWidget()
        
        # Thêm tất cả vào splitter
        splitter.addWidget(top_widget)
        splitter.addWidget(self.plotWidget_amplitude)
        splitter.addWidget(self.plotWidget_phase_time)
        
        # Thiết lập tỷ lệ ban đầu cho splitter
        splitter.setSizes([300, 300, 300])

    def setup_plots(self):
        """Initialize all plot widgets với cài đặt zoom tốt hơn"""
        self.setup_phase_plot()
        self.setup_amplitude_plot()
        self.setup_iq_plot()
        self.setup_phase_time_plot()

    def setup_phase_plot(self):
        """Setup phase data plot với zoom tốt"""
        self.plotWidget_phase.setYRange(-2*np.pi, 2*np.pi)
        self.plotWidget_phase.addLegend()
        self.plotWidget_phase.setTitle("Phase Data - Last Frame")
        self.plotWidget_phase.setLabel('left', 'Phase (rad)')
        self.plotWidget_phase.setLabel('bottom', 'Subcarrier Index')
        
        # Enable grid for better zoom experience
        self.plotWidget_phase.showGrid(x=True, y=True, alpha=0.3)
        
        self.phase_curve = self.plotWidget_phase.plot([], name="CSI Raw Data", pen='r')

    def setup_amplitude_plot(self):
        """Setup amplitude data plot với zoom tốt"""
        self.plotWidget_amplitude.addLegend()
        self.plotWidget_amplitude.setTitle("Subcarrier Amplitude Data")
        self.plotWidget_amplitude.setLabel('left', 'Amplitude')
        self.plotWidget_amplitude.setLabel('bottom', 'Time (Cumulative Packet Count)')
        
        # Enable grid and auto range
        self.plotWidget_amplitude.showGrid(x=True, y=True, alpha=0.3)
        self.plotWidget_amplitude.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis)
        
        # Initialize curves
        self.amplitude_curves = []
        self.amplitude_curves.append(
            self.plotWidget_amplitude.plot(csi_buffer.agc_gain_data, name="AGC Gain", pen='y')
        )
        self.amplitude_curves.append(
            self.plotWidget_amplitude.plot(csi_buffer.fft_gain_data, name="FFT Gain", pen='y')
        )
        
        for i in range(CSI_DATA_COLUMNS):
            curve = self.plotWidget_amplitude.plot(
                np.abs(csi_buffer.csi_data_complex[:, i]), name=str(i), pen=(255, 255, 255)
            )
            self.amplitude_curves.append(curve)

    def setup_phase_time_plot(self):
        """Setup phase over time plot với zoom tốt"""
        self.plotWidget_phase_time.addLegend()
        self.plotWidget_phase_time.setTitle("Subcarrier Phase Over Time")
        self.plotWidget_phase_time.setLabel('left', 'Phase (rad)')
        self.plotWidget_phase_time.setLabel('bottom', 'Time (Cumulative Packet Count)')
        
        # Enable grid and auto range
        self.plotWidget_phase_time.showGrid(x=True, y=True, alpha=0.3)
        self.plotWidget_phase_time.getViewBox().enableAutoRange(axis=pg.ViewBox.YAxis)
        
        self.phase_time_curves = []
        for i in range(CSI_DATA_COLUMNS):
            phase_curve = self.plotWidget_phase_time.plot(
                np.angle(csi_buffer.csi_data_complex[:, i]), name=str(i), pen=(255, 255, 255)
            )
            self.phase_time_curves.append(phase_curve)

    def setup_iq_plot(self):
        """Setup IQ scatter plot với zoom tốt"""
        self.plotWidget_iq.setLabel('left', 'Q (Imag)')
        self.plotWidget_iq.setLabel('bottom', 'I (Real)')
        self.plotWidget_iq.setTitle("IQ Plot - Last Frame")
        
        # Set initial view range
        view_box = self.plotWidget_iq.getViewBox()
        view_box.setRange(xRange=[-30, 30], yRange=[-30, 30])
        view_box.setAspectLocked(True)
        
        # Enable grid
        self.plotWidget_iq.showGrid(x=True, y=True, alpha=0.3)
        
        self.iq_scatter = ScatterPlotItem(size=6, pen=None, brush=pg.mkBrush(255, 255, 255, 120))
        self.plotWidget_iq.addItem(self.iq_scatter)
        self.iq_colors = []

    def setup_timer(self):
        """Setup update timer"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(100)  # 10 Hz update rate

    def update_curve_colors(self, color_list: List[Tuple[int, int, int]]):
        """Update colors for subcarrier curves"""
        self.visible_subcarriers = len(color_list)
        self.iq_colors = color_list
        
        # Set X range for phase plot
        self.plotWidget_phase.setXRange(0, self.visible_subcarriers)
        
        # Update curve colors
        for i, color in enumerate(color_list):
            if i < len(self.amplitude_curves):
                self.amplitude_curves[i].setPen(color)
            if i < len(self.phase_time_curves):
                self.phase_time_curves[i].setPen(color)

    def update_display(self):
        """Update all display elements"""
        self.update_iq_plot()
        self.update_phase_plot()
        self.update_amplitude_plots()
        self.update_phase_time_plots()

    def update_iq_plot(self):
        """Update IQ scatter plot"""
        if self.visible_subcarriers == 0:
            return
            
        last_frame = csi_buffer.csi_data_complex[-1]
        i_vals = np.real(last_frame[:self.visible_subcarriers])
        q_vals = np.imag(last_frame[:self.visible_subcarriers])
        
        # Tạo brush array cho hiệu suất tốt hơn
        brushes = []
        for i in range(self.visible_subcarriers):
            if i < len(self.iq_colors):
                color = self.iq_colors[i]
                brushes.append(pg.mkBrush(color[0], color[1], color[2], 180))
            else:
                brushes.append(pg.mkBrush(200, 200, 200, 180))
        
        # Cập nhật dữ liệu scatter plot
        self.iq_scatter.setData(i_vals, q_vals, brush=brushes)

    def update_phase_plot(self):
        """Update phase data plot"""
        if self.visible_subcarriers == 0:
            return
            
        last_phase = np.angle(csi_buffer.csi_data_complex[-1, :self.visible_subcarriers])
        x_data = np.arange(self.visible_subcarriers)
        self.phase_curve.setData(x_data, last_phase)

    def update_amplitude_plots(self):
        """Update amplitude over time plots"""
        amplitude_data = np.abs(csi_buffer.csi_data_complex)
        
        # Update gain curves
        self.amplitude_curves[0].setData(csi_buffer.agc_gain_data)
        self.amplitude_curves[1].setData(csi_buffer.fft_gain_data)
        
        # Update subcarrier amplitude curves
        for i in range(min(CSI_DATA_COLUMNS, len(self.amplitude_curves) - 2)):
            self.amplitude_curves[i + 2].setData(amplitude_data[:, i])

    def update_phase_time_plots(self):
        """Update phase over time plots"""
        phase_data = np.angle(csi_buffer.csi_data_complex)
        for i in range(min(CSI_DATA_COLUMNS, len(self.phase_time_curves))):
            self.phase_time_curves[i].setData(phase_data[:, i])

class ColorSchemeGenerator:
    """Generate color schemes for different CSI data lengths"""
    
    COLOR_SCHEMES = {
        52:   {"red": (0, 12), "green": (13, 26), "yellow": None},
        106:  {"red": (0, 25), "green": (27, 53), "yellow": None},
        114:  {"red": (0, 27), "green": (29, 56), "yellow": None},
        128:  {"red": (0, 31), "green": (32, 63), "yellow": None},
        234:  {"red": (0, 28), "green": (29, 56), "yellow": (60, 116)},
        256:  {"red": (0, 32), "green": (32, 63), "yellow": (64, 128)},
        384:  {"red": (0, 63), "green": (64, 127), "yellow": (128, 192)},
        490:  {"red": (0, 61), "green": (62, 122), "yellow": (123, 245)},
        512:  {"red": (0, 63), "green": (64, 127), "yellow": (128, 256)}
    }
    
    @classmethod
    def generate_colors(cls, data_length: int) -> List[Tuple[int, int, int]]:
        """Generate color scheme for given data length"""
        scheme = cls.COLOR_SCHEMES.get(data_length)
        if not scheme:
            print(f"Warning: No color scheme for data length {data_length}")
            return [(200, 200, 200)] * data_length
        
        colors = []
        for i in range(data_length):
            colors.append(cls._get_color_for_index(i, scheme))
        return colors
    
    @staticmethod
    def _get_color_for_index(index: int, scheme: dict) -> Tuple[int, int, int]:
        """Get color for specific subcarrier index"""
        red_range = scheme["red"]
        green_range = scheme["green"]
        yellow_range = scheme["yellow"]
        
        if red_range and red_range[0] <= index <= red_range[1]:
            intensity = int(255 * (index - red_range[0]) / max(1, (red_range[1] - red_range[0])))
            return (intensity, 0, 0)
        elif green_range and green_range[0] <= index <= green_range[1]:
            intensity = int(255 * (index - green_range[0]) / max(1, (green_range[1] - green_range[0])))
            return (0, intensity, 0)
        elif yellow_range and yellow_range[0] <= index <= yellow_range[1]:
            intensity = int(255 * (index - yellow_range[0]) / max(1, (yellow_range[1] - yellow_range[0])))
            return (intensity, intensity, 0)
        else:
            return (200, 200, 200)

class CSIReader:
    """Handle CSI data reading and parsing"""
    
    def __init__(self, port: str, csv_writer, log_file_fd):
        self.port = port
        self.csv_writer = csv_writer
        self.log_file_fd = log_file_fd
        self.serial_conn = None
        self.color_scheme_initialized = False

    def open_serial(self) -> bool:
        """Open serial connection"""
        try:
            self.serial_conn = serial.Serial(
                port=self.port, 
                baudrate=921600,
                bytesize=8, 
                parity='N', 
                stopbits=1,
                timeout=1.0
            )
            print("Serial port opened successfully")
            return True
        except serial.SerialException as e:
            print(f"Failed to open serial port: {e}")
            return False

    def read_and_parse_data(self, callback: Optional[Callable] = None):
        """Main data reading and parsing loop"""
        if not self.serial_conn or not self.serial_conn.is_open:
            print("Serial port not open")
            return

        while True:
            line = self.read_serial_line()
            if not line:
                continue
                
            if self.process_csi_data(line, callback):
                continue
                
            # Log non-CSI data
            self.log_file_fd.write(line + '\n')
            self.log_file_fd.flush()

    def read_serial_line(self) -> Optional[str]:
        """Read and decode a line from serial port"""
        try:
            raw_line = self.serial_conn.readline()
            return raw_line.decode('utf-8', errors='ignore').strip()
        except (serial.SerialException, UnicodeDecodeError) as e:
            print(f"Serial read error: {e}")
            return None

    def process_csi_data(self, line: str, callback: Optional[Callable]) -> bool:
        """Process CSI data line, return True if CSI data was processed"""
        if "CSI_DATA" not in line:
            return False

        try:
            csv_reader = csv.reader(StringIO(line))
            csi_data = next(csv_reader)
            
            if not self.validate_data_format(csi_data):
                return False
                
            csi_raw_data = self.parse_json_data(csi_data[-1])
            if csi_raw_data is None:
                return False
                
            if not self.validate_data_length(csi_data, csi_raw_data):
                return False
                
            self.process_valid_csi_data(csi_data, csi_raw_data, callback)
            return True
            
        except Exception as e:
            print(f"Error processing CSI data: {e}")
            self.log_file_fd.write(f"Processing error: {e}\n")
            return False

    def validate_data_format(self, csi_data: List[str]) -> bool:
        """Validate CSI data format"""
        valid_length = len(csi_data) in (len(DATA_COLUMNS_NAMES), len(DATA_COLUMNS_NAMES_C5C6))
        if not valid_length:
            print(f"Element number mismatch: {len(csi_data)}")
            self.log_file_fd.write("Element number is not equal\n")
        return valid_length

    def parse_json_data(self, json_str: str) -> Optional[List[float]]:
        """Parse JSON CSI data"""
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("Incomplete or invalid JSON data")
            self.log_file_fd.write("Data is incomplete\n")
            return None

    def validate_data_length(self, csi_data: List[str], csi_raw_data: List[float]) -> bool:
        """Validate data length consistency"""
        csi_data_len = int(csi_data[-3])
        if csi_data_len != len(csi_raw_data):
            print(f"Data length mismatch: {csi_data_len} vs {len(csi_raw_data)}")
            self.log_file_fd.write("CSI data length is not equal\n")
            return False
        return True

    def process_valid_csi_data(self, csi_data: List[str], csi_raw_data: List[float], 
                             callback: Optional[Callable]):
        """Process valid CSI data"""
        # Extract gains
        fft_gain = float(csi_data[6])
        agc_gain = float(csi_data[7])
        
        # Convert to complex data
        csi_complex = self.raw_to_complex(csi_raw_data)
        
        # Initialize color scheme on first valid data
        if not self.color_scheme_initialized:
            self.initialize_color_scheme(len(csi_raw_data), callback)
            self.color_scheme_initialized = True
        
        # Update buffers
        csi_buffer.update_buffers(csi_complex, agc_gain, fft_gain)
        
        # Write to CSV
        self.csv_writer.writerow(csi_data)

    @staticmethod
    def raw_to_complex(raw_data: List[float]) -> np.ndarray:
        """Convert raw data to complex array"""
        data_len = len(raw_data) // 2
        complex_data = np.zeros(CSI_DATA_COLUMNS, dtype=np.complex64)
        
        for i in range(data_len):
            complex_data[i] = complex(raw_data[i * 2 + 1], raw_data[i * 2])
            
        return complex_data

    def initialize_color_scheme(self, data_length: int, callback: Optional[Callable]):
        """Initialize color scheme for subcarriers"""
        colors = ColorSchemeGenerator.generate_colors(data_length)
        if callback:
            callback(colors)

    def close(self):
        """Close serial connection"""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()

class CSIThread(QThread):
    """QThread for CSI data reading"""
    data_ready = pyqtSignal(object)
    
    def __init__(self, serial_port: str, save_file_name: str, log_file_name: str):
        super().__init__()
        self.serial_port = serial_port
        self.save_file_name = save_file_name
        self.log_file_name = log_file_name
        self.save_file_fd = None
        self.log_file_fd = None
        self.csv_writer = None

    def __enter__(self):
        """Context manager entry"""
        self.setup_files()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()

    def setup_files(self):
        """Setup file handles and CSV writer"""
        self.save_file_fd = open(self.save_file_name, 'w')
        self.log_file_fd = open(self.log_file_name, 'w')
        self.csv_writer = csv.writer(self.save_file_fd)
        self.csv_writer.writerow(DATA_COLUMNS_NAMES)

    def run(self):
        """Main thread execution"""
        reader = CSIReader(self.serial_port, self.csv_writer, self.log_file_fd)
        
        if reader.open_serial():
            reader.read_and_parse_data(callback=self.data_ready.emit)
        
        reader.close()

    def cleanup(self):
        """Clean up resources"""
        for fd in [self.save_file_fd, self.log_file_fd]:
            if fd:
                fd.close()

def main():
    """Main application entry point"""
    if sys.version_info < (3, 6):
        print("Python version should be >= 3.6")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="Read CSI data from serial port and display it graphically"
    )
    parser.add_argument('-p', '--port', required=True,
                        help="Serial port number of csv_recv device")
    parser.add_argument('-s', '--store', default='./csi_data.csv',
                        help="Save the data printed by the serial port to a file")
    parser.add_argument('-l', '--log', default="./csi_data_log.txt",
                        help="Save other serial data and bad CSI data to a log file")

    args = parser.parse_args()

    # Create and run application
    app = QApplication(sys.argv)
    
    with CSIThread(args.port, args.store, args.log) as csi_thread:
        window = CSIDataGraphicalWindow()
        csi_thread.data_ready.connect(window.update_curve_colors)
        csi_thread.start()
        window.show()
        
        sys.exit(app.exec())

if __name__ == '__main__':
    main()