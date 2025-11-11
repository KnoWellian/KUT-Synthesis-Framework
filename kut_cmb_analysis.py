"""
KnoWellian Universe Theory (KUT) Synthesis Framework
CMB Cairo Q-Lattice Signature Detection and Analysis

This module provides tools for detecting the predicted Cairo Q-Lattice
geometric signature in Cosmic Microwave Background (CMB) data as described
in "The KnoWellian Schizophrenia" paper.

Author: David Noel Lynch (with AI collaboration)
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, ndimage
from scipy.spatial import Delaunay, distance
from scipy.stats import chi2
import healpy as hp
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class CairoQLatticeDetector:
    """
    Detects Cairo Q-Lattice (pentagonal tiling) signatures in CMB data.
    
    The Cairo Q-Lattice is predicted by KUT to be the fundamental geometric
    structure of the KRAM (KnoWellian Resonant Attractor Manifold).
    """
    
    def __init__(self, nside: int = 512):
        """
        Initialize the detector.
        
        Parameters:
        -----------
        nside : int
            HEALPix resolution parameter (512 recommended for Planck data)
        """
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.golden_ratio = (1 + np.sqrt(5)) / 2  # φ ≈ 1.618
        
    def load_cmb_data(self, filename: Optional[str] = None) -> np.ndarray:
        """
        Load CMB temperature map data.
        
        Parameters:
        -----------
        filename : str, optional
            Path to HEALPix FITS file. If None, generates synthetic data.
            
        Returns:
        --------
        cmb_map : np.ndarray
            CMB temperature fluctuation map in HEALPix format
        """
        if filename is None:
            print("Generating synthetic CMB data with Cairo Q-Lattice signature...")
            return self._generate_synthetic_cmb()
        else:
            print(f"Loading CMB data from {filename}...")
            return hp.read_map(filename, verbose=False)
    
    def _generate_synthetic_cmb(self) -> np.ndarray:
        """
        Generate synthetic CMB with embedded Cairo Q-Lattice signature.
        This serves as a test case for the detection algorithm.
        """
        # Generate standard CMB power spectrum
        ell = np.arange(3 * self.nside)
        cl = self._theoretical_cl(ell)
        
        # Create base map from power spectrum
        np.random.seed(42)
        cmb_map = hp.synfast(cl, self.nside, verbose=False)
        
        # Add subtle Cairo Q-Lattice geometric signature
        cmb_map += self._inject_cairo_signature(amplitude=0.05)
        
        return cmb_map
    
    def _theoretical_cl(self, ell: np.ndarray) -> np.ndarray:
        """
        Simplified theoretical CMB power spectrum.
        """
        # Simplified ΛCDM-like spectrum
        ell_peak = 220
        amplitude = 5000
        cl = amplitude * np.exp(-(ell - ell_peak)**2 / (2 * 100**2))
        cl[ell < 2] = 0
        return cl
    
    def _inject_cairo_signature(self, amplitude: float = 0.1) -> np.ndarray:
        """
        Inject a Cairo Q-Lattice geometric signature into the map.
        
        This creates subtle pentagonal clustering patterns that would arise
        from the KRAM's geometric structure.
        """
        signature = np.zeros(self.npix)
        
        # Create pentagonal vertices on the sphere
        n_pentagons = 144  # Fibonacci number
        theta = np.random.uniform(0, np.pi, n_pentagons)
        phi = np.random.uniform(0, 2*np.pi, n_pentagons)
        
        # Convert to pixel indices
        pix_centers = hp.ang2pix(self.nside, theta, phi)
        
        # Create pentagonal influence regions
        for center_pix in pix_centers:
            # Get neighboring pixels
            neighbors = hp.get_all_neighbours(self.nside, center_pix)
            neighbors = neighbors[neighbors >= 0]  # Remove invalid pixels
            
            # Apply pentagonal modulation
            scale = self.golden_ratio
            signature[center_pix] += amplitude * scale
            signature[neighbors] += amplitude * scale / 2
            
        return signature
    
    def extract_hotspots(self, cmb_map: np.ndarray, 
                         threshold_sigma: float = 2.5) -> np.ndarray:
        """
        Extract significant temperature fluctuation hotspots.
        
        Parameters:
        -----------
        cmb_map : np.ndarray
            CMB temperature map
        threshold_sigma : float
            Significance threshold in standard deviations
            
        Returns:
        --------
        hotspot_pixels : np.ndarray
            Indices of hotspot pixels
        """
        # Remove mean and normalize
        normalized = (cmb_map - np.mean(cmb_map)) / np.std(cmb_map)
        
        # Find peaks above threshold
        hotspots = np.where(np.abs(normalized) > threshold_sigma)[0]
        
        print(f"Found {len(hotspots)} hotspots above {threshold_sigma}σ")
        return hotspots
    
    def compute_topological_features(self, hotspot_pixels: np.ndarray) -> Dict:
        """
        Compute topological features of hotspot distribution.
        
        This uses Topological Data Analysis (TDA) to identify geometric
        structures in the hotspot pattern.
        """
        # Convert pixel indices to 3D Cartesian coordinates
        theta, phi = hp.pix2ang(self.nside, hotspot_pixels)
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        coords = np.column_stack([x, y, z])
        
        # Compute Delaunay triangulation on sphere
        # (approximate for large number of points)
        if len(coords) > 1000:
            # Sample for computational efficiency
            indices = np.random.choice(len(coords), 1000, replace=False)
            coords_sample = coords[indices]
        else:
            coords_sample = coords
        
        # Project to 2D for triangulation (stereographic projection)
        coords_2d = self._stereographic_projection(coords_sample)
        tri = Delaunay(coords_2d)
        
        # Analyze vertex valencies (number of edges per vertex)
        valencies = self._compute_vertex_valencies(tri)
        
        # Count geometric features
        features = {
            'total_vertices': len(coords),
            'sampled_vertices': len(coords_sample),
            'triangles': len(tri.simplices),
            'valency_distribution': np.bincount(valencies),
            'mean_valency': np.mean(valencies),
            'pentagonal_excess': self._compute_pentagonal_excess(valencies)
        }
        
        return features
    
    def _stereographic_projection(self, coords_3d: np.ndarray) -> np.ndarray:
        """
        Stereographic projection from sphere to plane.
        """
        x, y, z = coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2]
        u = x / (1 - z + 1e-10)
        v = y / (1 - z + 1e-10)
        return np.column_stack([u, v])
    
    def _compute_vertex_valencies(self, tri: Delaunay) -> np.ndarray:
        """
        Compute vertex valencies (degree) from triangulation.
        """
        n_vertices = tri.points.shape[0]
        valencies = np.zeros(n_vertices, dtype=int)
        
        for simplex in tri.simplices:
            for vertex in simplex:
                valencies[vertex] += 1
        
        return valencies
    
    def _compute_pentagonal_excess(self, valencies: np.ndarray) -> float:
        """
        Compute the pentagonal excess ratio P_excess.
        
        This measures the over-representation of 5-valent vertices
        compared to the expected distribution for a random network.
        
        Returns:
        --------
        P_excess : float
            Ratio of observed 5-valent vertices to expected (null model)
        """
        if len(valencies) == 0:
            return 0.0
        
        # Count 5-valent vertices
        n_pentagonal = np.sum(valencies == 5)
        
        # Null model: random planar graph has mean valency ≈ 6
        # Expected distribution is roughly Poisson
        mean_valency = np.mean(valencies)
        expected_pentagonal = len(valencies) * np.exp(-mean_valency) * \
                              (mean_valency**5) / np.math.factorial(5)
        
        if expected_pentagonal > 0:
            P_excess = n_pentagonal / expected_pentagonal
        else:
            P_excess = 0.0
        
        return P_excess
    
    def compute_bispectrum_signature(self, cmb_map: np.ndarray, 
                                     ell_max: int = 500) -> Tuple[np.ndarray, float]:
        """
        Compute bispectrum to detect non-Gaussian features.
        
        The Cairo Q-Lattice should produce a characteristic bispectrum
        with pentagonal symmetry.
        
        Parameters:
        -----------
        cmb_map : np.ndarray
            CMB temperature map
        ell_max : int
            Maximum multipole for analysis
            
        Returns:
        --------
        bispectrum : np.ndarray
            Estimated bispectrum values
        f_NL : float
            Non-Gaussianity parameter
        """
        print("Computing bispectrum (simplified estimator)...")
        
        # Convert to alm coefficients
        alm = hp.map2alm(cmb_map, lmax=ell_max)
        
        # Simplified bispectrum estimator
        # Full computation requires triplet summations - this is a proxy
        alm_abs = np.abs(alm)
        
        # Estimate non-Gaussianity through higher-order moments
        fourth_moment = np.mean(alm_abs**4)
        second_moment = np.mean(alm_abs**2)
        
        # f_NL proxy (not rigorous, but indicates non-Gaussianity)
        f_NL = (fourth_moment / second_moment**2 - 3) * 100
        
        return alm_abs, f_NL
    
    def statistical_significance(self, P_excess: float, 
                                  n_samples: int = 1000) -> float:
        """
        Compute statistical significance of pentagonal excess.
        
        Parameters:
        -----------
        P_excess : float
            Observed pentagonal excess ratio
        n_samples : int
            Number of bootstrap samples for null distribution
            
        Returns:
        --------
        sigma : float
            Significance in standard deviations
        """
        print(f"Computing statistical significance (P_excess = {P_excess:.3f})...")
        
        # Generate null distribution via bootstrap
        null_distribution = []
        for _ in range(n_samples):
            # Random planar graph valency distribution
            random_valencies = np.random.poisson(6, size=500)
            null_P = self._compute_pentagonal_excess(random_valencies)
            null_distribution.append(null_P)
        
        null_distribution = np.array(null_distribution)
        
        # Compute z-score
        mean_null = np.mean(null_distribution)
        std_null = np.std(null_distribution)
        
        if std_null > 0:
            sigma = (P_excess - mean_null) / std_null
        else:
            sigma = 0.0
        
        return sigma
    
    def visualize_results(self, cmb_map: np.ndarray, 
                          hotspot_pixels: np.ndarray,
                          features: Dict):
        """
        Create visualization of analysis results.
        """
        fig = plt.figure(figsize=(16, 12))
        
        # 1. CMB temperature map
        ax1 = fig.add_subplot(2, 3, 1)
        hp.mollview(cmb_map, title='CMB Temperature Map', 
                    hold=True, cbar=True, sub=(2, 3, 1))
        plt.text(0.05, 0.95, f"NSIDE={self.nside}", 
                transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
        
        # 2. Hotspot distribution
        hotspot_map = np.zeros(self.npix)
        hotspot_map[hotspot_pixels] = 1
        hp.mollview(hotspot_map, title='Detected Hotspots', 
                    hold=True, cbar=False, sub=(2, 3, 2))
        
        # 3. Valency distribution
        ax3 = fig.add_subplot(2, 3, 3)
        if len(features['valency_distribution']) > 0:
            valencies = np.arange(len(features['valency_distribution']))
            ax3.bar(valencies, features['valency_distribution'], 
                   color='steelblue', alpha=0.7)
            ax3.axvline(5, color='red', linestyle='--', linewidth=2, 
                       label='Pentagonal (5-valent)')
            ax3.set_xlabel('Vertex Valency')
            ax3.set_ylabel('Count')
            ax3.set_title('Vertex Valency Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Power spectrum
        ax4 = fig.add_subplot(2, 3, 4)
        cl = hp.anafast(cmb_map, lmax=1000)
        ell = np.arange(len(cl))
        ax4.plot(ell[2:], ell[2:] * (ell[2:] + 1) * cl[2:] / (2 * np.pi), 
                color='darkblue', linewidth=1.5)
        ax4.set_xlabel('Multipole ℓ')
        ax4.set_ylabel('ℓ(ℓ+1)Cℓ/2π [μK²]')
        ax4.set_title('Angular Power Spectrum')
        ax4.set_xscale('log')
        ax4.grid(True, alpha=0.3)
        
        # 5. Statistical summary
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.axis('off')
        
        P_excess = features['pentagonal_excess']
        sigma = self.statistical_significance(P_excess)
        
        summary_text = f"""
        CAIRO Q-LATTICE ANALYSIS RESULTS
        ═══════════════════════════════════
        
        Total Hotspots: {features['total_vertices']}
        Triangles Formed: {features['triangles']}
        Mean Valency: {features['mean_valency']:.2f}
        
        PENTAGONAL EXCESS RATIO:
        P_excess = {P_excess:.3f}
        
        Statistical Significance:
        σ = {sigma:.2f}
        
        FALSIFICATION THRESHOLD:
        P_excess < 0.1 (KUT predicts > 1.2)
        
        Current Status: {'SIGNATURE DETECTED' if P_excess > 1.0 else 'NO SIGNATURE'}
        Confidence: {abs(sigma):.1f}σ
        """
        
        ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # 6. Golden ratio connections
        ax6 = fig.add_subplot(2, 3, 6)
        golden_ratios = []
        for i in range(2, min(20, len(features['valency_distribution']))):
            if features['valency_distribution'][i] > 0:
                ratio = features['valency_distribution'][i] / \
                        features['valency_distribution'][max(2, i-1)]
                golden_ratios.append(ratio)
        
        if len(golden_ratios) > 0:
            ax6.plot(range(len(golden_ratios)), golden_ratios, 
                    'o-', color='gold', linewidth=2, markersize=8)
            ax6.axhline(self.golden_ratio, color='red', linestyle='--', 
                       linewidth=2, label=f'φ = {self.golden_ratio:.3f}')
            ax6.set_xlabel('Valency Index')
            ax6.set_ylabel('Ratio to Previous')
            ax6.set_title('Golden Ratio Test')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def run_full_analysis(cmb_filename: Optional[str] = None):
    """
    Run complete Cairo Q-Lattice detection analysis.
    
    Parameters:
    -----------
    cmb_filename : str, optional
        Path to CMB data file. If None, uses synthetic data.
    """
    print("="*70)
    print("KnoWellian Universe Theory - CMB Cairo Q-Lattice Detection")
    print("="*70)
    print()
    
    # Initialize detector
    detector = CairoQLatticeDetector(nside=512)
    
    # Load or generate data
    cmb_map = detector.load_cmb_data(cmb_filename)
    print(f"CMB map shape: {cmb_map.shape}")
    print(f"Temperature range: [{np.min(cmb_map):.3e}, {np.max(cmb_map):.3e}]")
    print()
    
    # Extract hotspots
    hotspot_pixels = detector.extract_hotspots(cmb_map, threshold_sigma=2.5)
    print()
    
    # Compute topological features
    print("Computing topological features...")
    features = detector.compute_topological_features(hotspot_pixels)
    print(f"Mean vertex valency: {features['mean_valency']:.2f}")
    print(f"Pentagonal excess P_excess: {features['pentagonal_excess']:.3f}")
    print()
    
    # Compute bispectrum
    alm, f_NL = detector.compute_bispectrum_signature(cmb_map, ell_max=500)
    print(f"Non-Gaussianity parameter f_NL ≈ {f_NL:.2f}")
    print()
    
    # Statistical significance
    sigma = detector.statistical_significance(features['pentagonal_excess'])
    print(f"Statistical significance: {sigma:.2f}σ")
    print()
    
    # Determine result
    print("="*70)
    if features['pentagonal_excess'] > 1.0 and sigma > 3.0:
        print("RESULT: Cairo Q-Lattice signature DETECTED")
        print(f"        P_excess = {features['pentagonal_excess']:.3f} > 1.0")
        print(f"        Significance = {sigma:.1f}σ > 3σ")
        print("        ✓ KUT prediction CONFIRMED")
    elif features['pentagonal_excess'] < 0.1:
        print("RESULT: Cairo Q-Lattice signature NOT DETECTED")
        print(f"        P_excess = {features['pentagonal_excess']:.3f} < 0.1")
        print("        ✗ KUT prediction FALSIFIED")
    else:
        print("RESULT: INCONCLUSIVE")
        print(f"        P_excess = {features['pentagonal_excess']:.3f}")
        print(f"        Significance = {sigma:.1f}σ")
        print("        Further analysis required")
    print("="*70)
    print()
    
    # Visualize
    print("Generating visualization...")
    fig = detector.visualize_results(cmb_map, hotspot_pixels, features)
    plt.savefig('kut_cmb_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved: kut_cmb_analysis.png")
    plt.show()
    
    return detector, cmb_map, features


if __name__ == "__main__":
    # Run analysis with synthetic data
    detector, cmb_map, features = run_full_analysis()
    
    print("\nTo analyze real Planck data:")
    print("1. Download Planck 2018 Legacy CMB maps from:")
    print("   https://pla.esac.esa.int/pla/")
    print("2. Run: run_full_analysis('path/to/planck_map.fits')")
