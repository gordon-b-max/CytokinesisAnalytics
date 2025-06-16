from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import eig, inv
from scipy.signal import savgol_filter

@dataclass
class EllipseFit:
    """Container for ellipse fitting results"""
    center: np.ndarray
    angle: float
    axes: np.ndarray
    perimeter: float
    radius: float

class CytokinesisAnalyzer:
    """Class for analyzing cytokinesis data"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.results: Dict[str, Dict] = {}
        
    def fit_ellipse(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit an ellipse to x,y coordinates using least squares method"""
        x = x[:, np.newaxis]
        y = y[:, np.newaxis]
        D = np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
        S = np.dot(D.T, D)
        C = np.zeros([6, 6])
        C[0, 2] = C[2, 0] = 2
        C[1, 1] = -1
        E, V = eig(np.dot(inv(S), C))
        n = np.argmax(np.abs(E))
        return V[:, n]

    def calculate_ellipse_parameters(self, a: np.ndarray) -> EllipseFit:
        """Calculate ellipse parameters from fitted coefficients"""
        b, c, d, f, g, a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
        
        # Calculate center
        num = b*b - a*c
        x0 = (c*d - b*f) / num
        y0 = (a*f - b*d) / num
        center = np.array([ x0, y0 ])
        
        # Calculate angle
        if b == 0:
            angle = 0 if a > c else np.pi/2
        else:
            angle = np.arctan(2*b/(a-c))/2
            if a <= c:
                angle += np.pi/2
                
        # Calculate axes
        up = 2*(a*f*f + c*d*d + g*b*b - 2*b*d*f - a*c*g)
        down1 = (b*b - a*c)*((c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        down2 = (b*b - a*c)*((a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
        axes = np.array([np.sqrt(up/down1), np.sqrt(up/down2)])
        
        # Calculate perimeter
        h = (axes[0] - axes[1])**2 / (axes[0] + axes[1])**2
        perimeter = np.pi * (axes[0] + axes[1]) * (1 + (3*h)/(10 + np.sqrt(4-3*h)))
        radius = np.mean(axes)
        
        return EllipseFit(center, angle, axes, perimeter, radius)

    def process_file(self, file_path: Path) -> Dict:
        """Process a single CSV file of cell boundary coordinates"""
        df = pd.read_csv(file_path, header=None)
        x_coords = df[0].values
        y_coords = df[1].values
        
        # Split data into frames where NaN values occur
        frames = self._split_into_frames(x_coords, y_coords)
        
        results = {
            'perimeters': [],
            'closures': [],
            'speeds': [],
            'times': []
        }
        
        prev_radius = None  # Initialize prev_radius outside the loop
        
        for i, (x, y) in enumerate(frames):
            if len(x) < 5:  # Skip frames with too few points
                continue
                
            # Fit ellipse and calculate parameters
            a = self.fit_ellipse(x, y)
            ellipse = self.calculate_ellipse_parameters(a)
            
            results['perimeters'].append(ellipse.perimeter)
            results['times'].append(i)
            
            # Calculate closure percentage
            if i == 0:
                initial_perimeter = ellipse.perimeter
            closure = 100 * (1 - ellipse.perimeter/initial_perimeter)
            results['closures'].append(closure)
            
            # Calculate speed
            if prev_radius is not None:  # Only calculate speed if we have a previous radius
                speed = abs(ellipse.radius - prev_radius) * 1000  # Convert to nm/s
                results['speeds'].append(speed)
            else:
                results['speeds'].append(np.nan)  # Add NaN for the first frame
            
            prev_radius = ellipse.radius
        
        return results

    def _split_into_frames(self, x: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split coordinate data into frames based on NaN values"""
        frames = []
        start_idx = 0
        
        for i in range(len(x)):
            if np.isnan(x[i]):
                if i > start_idx:
                    frames.append((x[start_idx:i], y[start_idx:i]))
                start_idx = i + 1
                
        if start_idx < len(x):
            frames.append((x[start_idx:], y[start_idx:]))
            
        return frames

    def analyze_directory(self) -> Dict:
        """Analyze all CSV files in the directory"""
        for file_path in self.data_dir.glob('*.csv'):
            self.results[file_path.name] = self.process_file(file_path)
        return self.results

    def generate_plots(self, output_dir: Path):
        """Generate analysis plots"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Create a subfolder for speed vs closure plots
        speed_closure_dir = output_dir / 'speed_vs_closure'
        speed_closure_dir.mkdir(exist_ok=True)
        
        # Plot perimeters
        plt.figure(figsize=(10, 6))
        for file_name, data in self.results.items():
            plt.plot(data['times'], data['perimeters'], label=file_name)
        plt.xlabel('Time')
        plt.ylabel('Perimeter')
        plt.title('Cell Perimeter Over Time')
        plt.legend()
        plt.savefig(output_dir / 'perimeters.png')
        plt.close()
        
        # Plot closure percentages
        plt.figure(figsize=(10, 6))
        for file_name, data in self.results.items():
            plt.plot(data['times'], data['closures'], label=file_name)
        plt.xlabel('Time')
        plt.ylabel('Closure Percentage')
        plt.title('Cell Closure Over Time')
        plt.legend()
        plt.savefig(output_dir / 'closures.png')
        plt.close()

        # Create individual speed vs closure plots for each file
        for file_name, data in self.results.items():
            plt.figure(figsize=(10, 6))
            
            # Convert to numpy arrays and remove NaN values
            closures = np.array(data['closures'])
            speeds = np.array(data['speeds'])
            
            # Create mask for valid (non-NaN) values
            valid_mask = ~np.isnan(closures) & ~np.isnan(speeds)
            
            if np.any(valid_mask):  # Only plot if we have valid data
                # Get valid data
                valid_closures = closures[valid_mask]
                valid_speeds = speeds[valid_mask]
                
                # Sort the data for proper fitting
                sort_idx = np.argsort(valid_closures)
                sorted_closures = valid_closures[sort_idx]
                sorted_speeds = valid_speeds[sort_idx]
                
                # Plot the actual data in blue
                plt.plot(sorted_closures, sorted_speeds, 'b-', linewidth=1, alpha=0.7, label='Raw Data')
                
                # Create the fitted line using Savitzky-Golay filter
                try:
                    # Use window length of 7 and polynomial order of 4
                    fitted_speeds = savgol_filter(sorted_speeds, window_length=21, polyorder=4)
                    plt.plot(sorted_closures, fitted_speeds, 'r-', linewidth=2, label='Fitted Curve')
                except:
                    # If Savitzky-Golay fails, try simple moving average
                    window_size = 5
                    fitted_speeds = np.convolve(sorted_speeds, np.ones(window_size)/window_size, mode='valid')
                    plt.plot(sorted_closures[window_size-1:], fitted_speeds, 'r-', linewidth=2, label='Fitted Curve')
                
                plt.xlabel('Closure Percentage')
                plt.ylabel('Speed (nm/s)')
                plt.title(f'Speed vs Closure Percentage - {file_name}')
                
                # Add grid and legend
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend()
                
                # Save the plot
                plt.savefig(speed_closure_dir / f'speed_vs_closure_{file_name}.png')
                plt.close()

def main():
    """Main Class Execution"""
    data_dir = Path('data')
    output_dir = Path('results')
    
    analyzer = CytokinesisAnalyzer(data_dir)
    results = analyzer.analyze_directory()
    analyzer.generate_plots(output_dir)
    
    # Save results to CSV
    summary_df = pd.DataFrame({
        'file': [],
        'max_speed': [],
        'max_closure': [],
        'time_to_closure': []
    })
    
    for file_name, data in results.items():
        summary_df = summary_df.append({
            'file': file_name,
            'max_speed': max(data['speeds']),
            'max_closure': max(data['closures']),
            'time_to_closure': data['times'][np.argmax(data['closures'])]
        }, ignore_index=True)
    
    summary_df.to_csv(output_dir / 'summary.csv', index=False)

if __name__ == '__main__':
    main()
    
    
    
    