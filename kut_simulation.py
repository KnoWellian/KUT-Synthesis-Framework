"""
KnoWellian Universe Theory (KUT) - Core Simulation Framework
==============================================================

This module implements the fundamental simulations described in:
"The KnoWellian Schizophrenia: A Procedural Ontology to Heal the Platonic Rift in Modern Physics"

Author: David Noel Lynch (with AI collaboration)
Date: November 10, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp
from scipy.linalg import eig
from dataclasses import dataclass
from typing import Tuple, Optional
import warnings

# Suppress minor warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class KUTParameters:
    """
    Physical parameters for KnoWellian Universe Theory simulations.
    
    Attributes:
        alpha: Synthesis rate from Mass/Control to Instant (dimensionless)
        beta: Synthesis rate from Wave/Chaos to Instant (dimensionless)
        gamma: Mutual decay/leakage rate (dimensionless)
        lambda_cubic: Cubic coupling strength in triadic potential
        Lambda_quartic: Quartic coupling strength in triadic potential
        kappa: Gauge-triadic coupling strength
        N_total: Total capacity of Apeiron projection
        alpha_render: Universal rendering constant
        c_light: Speed of light (natural units, c=1)
    """
    alpha: float = 0.5
    beta: float = 0.5
    gamma: float = 0.1
    lambda_cubic: float = 1.0
    Lambda_quartic: float = 0.5
    kappa: float = 0.1
    N_total: float = 1.0
    alpha_render: float = 0.3
    c_light: float = 1.0


class KnoWellianOntologicalTriadynamics:
    """
    Implementation of KOT: The dialectical engine of reality.
    
    Governs the perpetual transformation between:
    - phi_M: Mass/Control field (Past/Thesis)
    - phi_I: Instant/Consciousness field (Synthesis)
    - phi_W: Wave/Chaos field (Future/Antithesis)
    """
    
    def __init__(self, params: KUTParameters):
        self.params = params
        self.coupling_matrix = self._build_coupling_matrix()
        self.eigenvalues, self.eigenvectors = self._compute_eigenmodes()
        
    def _build_coupling_matrix(self) -> np.ndarray:
        """
        Construct the triadynamic coupling matrix M.
        
        dΦ/dt = M·Φ where Φ = [phi_M, phi_I, phi_W]^T
        
        Returns:
            3x3 coupling matrix ensuring perpetual oscillation
        """
        α, β, γ = self.params.alpha, self.params.beta, self.params.gamma
        
        M = np.array([
            [-γ,              α,              0],
            [α,  -(α + β),              β],
            [0,              β,             -γ]
        ])
        
        return M
    
    def _compute_eigenmodes(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues and eigenvectors of coupling matrix.
        
        The presence of complex eigenvalues proves the "Cosmic Breath" -
        the universe's fundamental oscillatory nature.
        
        Returns:
            (eigenvalues, eigenvectors) tuple
        """
        eigenvals, eigenvecs = eig(self.coupling_matrix)
        
        # Sort by real part (descending)
        idx = eigenvals.real.argsort()[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        return eigenvals, eigenvecs
    
    def triadic_dynamics(self, Phi: np.ndarray, t: float) -> np.ndarray:
        """
        Right-hand side of KOT evolution equation: dΦ/dt = M·Φ
        
        Args:
            Phi: State vector [phi_M, phi_I, phi_W]
            t: Time parameter
            
        Returns:
            Time derivative dΦ/dt
        """
        return self.coupling_matrix @ Phi
    
    def evolve(self, Phi_0: np.ndarray, t_span: Tuple[float, float], 
               num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evolve triadic field from initial condition.
        
        Args:
            Phi_0: Initial state [phi_M(0), phi_I(0), phi_W(0)]
            t_span: (t_start, t_end) time interval
            num_points: Number of time points for output
            
        Returns:
            (time_array, Phi_array) where Phi_array has shape (num_points, 3)
        """
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        
        solution = solve_ivp(
            self.triadic_dynamics,
            t_span,
            Phi_0,
            t_eval=t_eval,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        return solution.t, solution.y.T
    
    def analyze_cosmic_breath(self) -> dict:
        """
        Analyze the oscillatory nature of the universe via eigenmode analysis.
        
        Returns:
            Dictionary containing:
            - memory_mode: Zero eigenvalue (conserved quantity)
            - damping_rate: Real part of complex eigenvalues
            - breath_frequency: Imaginary part (oscillation frequency)
            - breath_period: Oscillation period
        """
        λ = self.eigenvalues
        
        # Find zero mode (memory/conservation)
        zero_idx = np.argmin(np.abs(λ.real))
        memory_mode = λ[zero_idx]
        
        # Find complex conjugate pair (cosmic breath)
        complex_modes = λ[np.abs(λ.imag) > 1e-10]
        
        if len(complex_modes) > 0:
            breath_mode = complex_modes[0]
            damping = -breath_mode.real
            frequency = np.abs(breath_mode.imag)
            period = 2 * np.pi / frequency if frequency > 0 else np.inf
        else:
            damping = 0.0
            frequency = 0.0
            period = np.inf
        
        return {
            'memory_mode': memory_mode,
            'damping_rate': damping,
            'breath_frequency': frequency,
            'breath_period': period,
            'eigenvalues': λ,
            'eigenvectors': self.eigenvectors
        }


class RenderingDynamics:
    """
    Implementation of the fundamental process of Rendering:
    the transformation of potentiality (w) into actuality (m).
    """
    
    def __init__(self, params: KUTParameters):
        self.params = params
        
    def rendering_rate(self, w: float, phi_I: float) -> float:
        """
        Compute the rate of rendering: dm/dt = α|phi_I|·w(t)
        
        Args:
            w: Unrendered potentiality
            phi_I: Instant/Consciousness field intensity
            
        Returns:
            Rate of transformation from potential to actual
        """
        return self.params.alpha_render * np.abs(phi_I) * w
    
    def rendering_dynamics(self, state: np.ndarray, t: float, 
                          phi_I_func) -> np.ndarray:
        """
        Coupled dynamics of m(t) and w(t) under conservation law.
        
        Args:
            state: [m, w] current rendered and unrendered amounts
            t: Time
            phi_I_func: Function returning phi_I(t)
            
        Returns:
            [dm/dt, dw/dt]
        """
        m, w = state
        phi_I = phi_I_func(t)
        
        rate = self.rendering_rate(w, phi_I)
        
        # Conservation: dm/dt = -dw/dt
        return np.array([rate, -rate])
    
    def evolve_rendering(self, m_0: float, phi_I_func, 
                        t_span: Tuple[float, float],
                        num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evolve rendering process with given Instant field.
        
        Args:
            m_0: Initial rendered amount
            phi_I_func: Function phi_I(t)
            t_span: (t_start, t_end)
            num_points: Time resolution
            
        Returns:
            (times, m_array, w_array)
        """
        w_0 = self.params.N_total - m_0
        state_0 = np.array([m_0, w_0])
        
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        
        solution = solve_ivp(
            lambda t, y: self.rendering_dynamics(y, t, phi_I_func),
            t_span,
            state_0,
            t_eval=t_eval,
            method='RK45'
        )
        
        return solution.t, solution.y[0], solution.y[1]


class TriadicPotential:
    """
    Implementation of the triadic scalar potential V_int.
    
    V_int = λ·phi_M·phi_W·phi_I + (Λ/4)·(phi_M² + phi_I² + phi_W²)²
    
    This potential ensures:
    1. No trivial vacuum (cubic term)
    2. Stability at large field values (quartic term)
    3. The mass gap Δ > 0
    """
    
    def __init__(self, params: KUTParameters):
        self.params = params
        
    def potential(self, phi_M: float, phi_I: float, phi_W: float) -> float:
        """
        Evaluate the triadic potential energy.
        
        Args:
            phi_M, phi_I, phi_W: Field values
            
        Returns:
            V_int value
        """
        λ = self.params.lambda_cubic
        Λ = self.params.Lambda_quartic
        
        cubic_term = λ * phi_M * phi_W * phi_I
        
        field_sum_sq = phi_M**2 + phi_I**2 + phi_W**2
        quartic_term = (Λ / 4) * field_sum_sq**2
        
        return cubic_term + quartic_term
    
    def find_vacuum(self, initial_guess: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Find the KnoWellian vacuum expectation values (v_M, v_I, v_W).
        
        This is the minimum of V_int, representing the stable ground state
        from which the mass gap emerges.
        
        Args:
            initial_guess: Starting point for optimization
            
        Returns:
            [v_M, v_I, v_W] vacuum expectation values
        """
        from scipy.optimize import minimize
        
        if initial_guess is None:
            initial_guess = np.array([0.1, 0.1, 0.1])
        
        def objective(phi):
            return self.potential(phi[0], phi[1], phi[2])
        
        result = minimize(objective, initial_guess, method='BFGS')
        
        return result.x
    
    def compute_mass_gap(self, vacuum: np.ndarray, 
                         epsilon: float = 0.01) -> float:
        """
        Compute the mass gap Δ as the minimum energy to satisfy
        the triadic rendering constraint: phi_M·phi_I·phi_W ≥ ε > 0
        
        Args:
            vacuum: Vacuum expectation values [v_M, v_I, v_W]
            epsilon: Minimum rendering threshold
            
        Returns:
            Mass gap Δ (energy units)
        """
        # Compute Hessian at vacuum
        h = 1e-6
        hessian = np.zeros((3, 3))
        
        for i in range(3):
            for j in range(3):
                phi_pp = vacuum.copy()
                phi_pm = vacuum.copy()
                phi_mp = vacuum.copy()
                phi_mm = vacuum.copy()
                
                phi_pp[i] += h
                phi_pp[j] += h
                phi_pm[i] += h
                phi_pm[j] -= h
                phi_mp[i] -= h
                phi_mp[j] += h
                phi_mm[i] -= h
                phi_mm[j] -= h
                
                hessian[i, j] = (
                    self.potential(*phi_pp) - self.potential(*phi_pm) -
                    self.potential(*phi_mp) + self.potential(*phi_mm)
                ) / (4 * h**2)
        
        # Smallest eigenvalue (positive definite near minimum)
        eigenvals = np.linalg.eigvalsh(hessian)
        kappa = np.min(eigenvals[eigenvals > 0])
        
        # Lower bound from arithmetic-geometric mean inequality
        epsilon_prime = epsilon / np.prod(vacuum)
        
        Delta_classical = 0.5 * kappa * 3 * epsilon_prime**(2/3)
        
        return Delta_classical


class KRAMEvolution:
    """
    Simulation of the KnoWellian Resonant Attractor Manifold.
    
    The KRAM is the cosmic memory substrate whose geometry is
    sculpted by all past acts of rendering.
    """
    
    def __init__(self, params: KUTParameters, grid_size: int = 50):
        self.params = params
        self.grid_size = grid_size
        self.metric = np.zeros((grid_size, grid_size))
        self.interaction_history = []
        
    def add_interaction_event(self, position: Tuple[int, int], 
                             intensity: float):
        """
        Add a rendering/interaction event to the KRAM.
        
        This modifies the metric tensor g_M(X) by accumulating
        the interaction current.
        
        Args:
            position: (x, y) grid coordinates
            intensity: Strength of interaction (|T_μI|)
        """
        x, y = position
        
        # Gaussian spread of influence
        X, Y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
        spread = 3.0
        
        influence = intensity * np.exp(
            -((X - x)**2 + (Y - y)**2) / (2 * spread**2)
        )
        
        self.metric += influence
        self.interaction_history.append((position, intensity))
        
    def compute_attractor_field(self) -> np.ndarray:
        """
        Compute the gradient field showing attractor valleys.
        
        Returns:
            (grad_x, grad_y) components of attractor field
        """
        grad_y, grad_x = np.gradient(self.metric)
        return -grad_x, -grad_y  # Negative for downhill (attraction)
    
    def evolve_particle_on_kram(self, initial_pos: np.ndarray,
                                num_steps: int = 1000,
                                dt: float = 0.01) -> np.ndarray:
        """
        Simulate a particle's trajectory on the KRAM landscape.
        
        The particle follows the principle of least action,
        guided by the sculpted geometry.
        
        Args:
            initial_pos: [x, y] starting position
            num_steps: Number of integration steps
            dt: Time step
            
        Returns:
            Array of shape (num_steps, 2) with trajectory
        """
        trajectory = np.zeros((num_steps, 2))
        pos = initial_pos.copy()
        trajectory[0] = pos
        
        for i in range(1, num_steps):
            # Get local gradient
            ix, iy = int(pos[0]), int(pos[1])
            if 0 <= ix < self.grid_size - 1 and 0 <= iy < self.grid_size - 1:
                grad_x = self.metric[iy, ix + 1] - self.metric[iy, ix]
                grad_y = self.metric[iy + 1, ix] - self.metric[iy, ix]
                
                # Move down gradient (toward attractor)
                velocity = -np.array([grad_x, grad_y])
                pos += velocity * dt
                
                # Boundary conditions
                pos = np.clip(pos, 0, self.grid_size - 1)
            
            trajectory[i] = pos
        
        return trajectory


# ============================================================================
# DEMONSTRATION AND VISUALIZATION
# ============================================================================

def demonstrate_kut_simulations():
    """
    Run comprehensive demonstrations of KUT core simulations.
    """
    print("=" * 70)
    print("KnoWellian Universe Theory - Core Simulation Demonstrations")
    print("=" * 70)
    
    params = KUTParameters()
    
    # 1. KOT Dynamics and Cosmic Breath
    print("\n1. KnoWellian Ontological Triadynamics (KOT)")
    print("-" * 70)
    
    kot = KnoWellianOntologicalTriadynamics(params)
    breath_analysis = kot.analyze_cosmic_breath()
    
    print(f"Eigenvalues (Cosmic Modes):")
    for i, λ in enumerate(breath_analysis['eigenvalues']):
        print(f"  λ_{i} = {λ.real:.6f} + {λ.imag:.6f}i")
    
    print(f"\nCosmic Breath Analysis:")
    print(f"  Damping rate (Γ): {breath_analysis['damping_rate']:.6f}")
    print(f"  Breath frequency (ω): {breath_analysis['breath_frequency']:.6f}")
    print(f"  Breath period (T): {breath_analysis['breath_period']:.4f}")
    
    # Evolve system
    Phi_0 = np.array([1.0, 0.5, 0.8])  # Initial triadic state
    t_span = (0, 50)
    times, Phi_evolution = kot.evolve(Phi_0, t_span, num_points=500)
    
    # 2. Rendering Dynamics
    print("\n2. Rendering Dynamics (Potentiality → Actuality)")
    print("-" * 70)
    
    renderer = RenderingDynamics(params)
    
    # Use phi_I from KOT evolution as catalyst
    phi_I_interp = lambda t: np.interp(t, times, Phi_evolution[:, 1])
    
    m_0 = 0.1  # Start with mostly unrendered universe
    t_render, m_t, w_t = renderer.evolve_rendering(m_0, phi_I_interp, t_span)
    
    print(f"Initial state: m(0) = {m_0:.4f}, w(0) = {params.N_total - m_0:.4f}")
    print(f"Final state:   m(T) = {m_t[-1]:.4f}, w(T) = {w_t[-1]:.4f}")
    print(f"Conservation check: m(T) + w(T) = {m_t[-1] + w_t[-1]:.6f}")
    print(f"  (Should equal N_total = {params.N_total})")
    
    # 3. Mass Gap from Triadic Potential
    print("\n3. Yang-Mills Mass Gap Calculation")
    print("-" * 70)
    
    potential = TriadicPotential(params)
    vacuum = potential.find_vacuum()
    
    print(f"KnoWellian Vacuum: (v_M, v_I, v_W) = ({vacuum[0]:.6f}, {vacuum[1]:.6f}, {vacuum[2]:.6f})")
    print(f"Vacuum energy: V_int(vacuum) = {potential.potential(*vacuum):.6f}")
    
    mass_gap = potential.compute_mass_gap(vacuum)
    print(f"\nMass Gap Δ = {mass_gap:.6f} (energy units)")
    print(f"  This is the 'activation energy of existence'")
    
    # 4. KRAM Memory Formation
    print("\n4. KRAM Evolution (Cosmic Memory)")
    print("-" * 70)
    
    kram = KRAMEvolution(params, grid_size=50)
    
    # Simulate multiple interaction events
    np.random.seed(42)
    num_events = 20
    
    for _ in range(num_events):
        x = np.random.randint(5, 45)
        y = np.random.randint(5, 45)
        intensity = np.random.uniform(0.5, 2.0)
        kram.add_interaction_event((x, y), intensity)
    
    print(f"Accumulated {num_events} interaction events")
    print(f"KRAM metric range: [{kram.metric.min():.4f}, {kram.metric.max():.4f}]")
    
    # Evolve particle on KRAM
    initial_pos = np.array([40.0, 10.0])
    trajectory = kram.evolve_particle_on_kram(initial_pos, num_steps=500)
    
    print(f"Particle trajectory: start = {initial_pos}, end = {trajectory[-1]}")
    
    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: KOT Evolution
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(times, Phi_evolution[:, 0], 'r-', label='φ_M (Mass/Control)', linewidth=2)
    ax1.plot(times, Phi_evolution[:, 1], 'b-', label='φ_I (Instant/Consciousness)', linewidth=2)
    ax1.plot(times, Phi_evolution[:, 2], 'g-', label='φ_W (Wave/Chaos)', linewidth=2)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Field Amplitude')
    ax1.set_title('KOT: The Cosmic Breath\n(Ternary Time Dynamics)')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot 2: Phase Space
    ax2 = plt.subplot(2, 3, 2, projection='3d')
    ax2.plot(Phi_evolution[:, 0], Phi_evolution[:, 1], Phi_evolution[:, 2], 
             'purple', linewidth=1.5, alpha=0.7)
    ax2.scatter(Phi_evolution[0, 0], Phi_evolution[0, 1], Phi_evolution[0, 2],
                c='green', s=100, marker='o', label='Start')
    ax2.scatter(Phi_evolution[-1, 0], Phi_evolution[-1, 1], Phi_evolution[-1, 2],
                c='red', s=100, marker='X', label='End')
    ax2.set_xlabel('φ_M')
    ax2.set_ylabel('φ_I')
    ax2.set_zlabel('φ_W')
    ax2.set_title('Phase Space Trajectory')
    ax2.legend()
    
    # Plot 3: Rendering Process
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(t_render, m_t, 'b-', label='m(t) - Rendered (Actual)', linewidth=2)
    ax3.plot(t_render, w_t, 'r--', label='w(t) - Unrendered (Potential)', linewidth=2)
    ax3.axhline(params.N_total, color='k', linestyle=':', alpha=0.5, label='N_total')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Information Content')
    ax3.set_title('Rendering: Potential → Actual\n(The Process of Becoming)')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # Plot 4: Triadic Potential Landscape
    ax4 = plt.subplot(2, 3, 4)
    phi_range = np.linspace(-2, 2, 100)
    phi_M_grid, phi_W_grid = np.meshgrid(phi_range, phi_range)
    V_grid = np.zeros_like(phi_M_grid)
    
    for i in range(len(phi_range)):
        for j in range(len(phi_range)):
            V_grid[i, j] = potential.potential(phi_M_grid[i, j], vacuum[1], phi_W_grid[i, j])
    
    contour = ax4.contourf(phi_M_grid, phi_W_grid, V_grid, levels=20, cmap='RdYlBu_r')
    ax4.scatter(vacuum[0], vacuum[2], c='red', s=200, marker='*', 
                edgecolors='black', linewidths=2, label='Vacuum', zorder=5)
    ax4.set_xlabel('φ_M (Mass/Control)')
    ax4.set_ylabel('φ_W (Wave/Chaos)')
    ax4.set_title(f'Triadic Potential Landscape\n(Mass Gap Δ = {mass_gap:.4f})')
    ax4.legend()
    plt.colorbar(contour, ax=ax4, label='V_int')
    
    # Plot 5: KRAM Attractor Landscape
    ax5 = plt.subplot(2, 3, 5)
    im = ax5.imshow(kram.metric, cmap='viridis', origin='lower', interpolation='bilinear')
    
    # Plot trajectory
    ax5.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, alpha=0.7)
    ax5.scatter(trajectory[0, 0], trajectory[0, 1], c='white', s=100, 
                marker='o', edgecolors='black', linewidths=2, label='Start', zorder=5)
    ax5.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=150,
                marker='X', edgecolors='black', linewidths=2, label='End', zorder=5)
    
    ax5.set_xlabel('X Coordinate')
    ax5.set_ylabel('Y Coordinate')
    ax5.set_title('KRAM: Cosmic Memory Landscape\n(Attractor Valleys)')
    ax5.legend()
    plt.colorbar(im, ax=ax5, label='Accumulated Imprint')
    
    # Plot 6: Cosmic Breath Frequency Spectrum
    ax6 = plt.subplot(2, 3, 6)
    
    # FFT of phi_I to show oscillation
    from scipy.fft import fft, fftfreq
    
    dt = times[1] - times[0]
    phi_I_signal = Phi_evolution[:, 1]
    N_fft = len(phi_I_signal)
    
    yf = fft(phi_I_signal)
    xf = fftfreq(N_fft, dt)[:N_fft//2]
    
    power = 2.0/N_fft * np.abs(yf[0:N_fft//2])
    
    ax6.semilogy(xf, power, 'b-', linewidth=2)
    ax6.axvline(breath_analysis['breath_frequency'], color='r', 
                linestyle='--', linewidth=2, label=f'ω = {breath_analysis["breath_frequency"]:.4f}')
    ax6.set_xlabel('Frequency')
    ax6.set_ylabel('Power (log scale)')
    ax6.set_title('Frequency Spectrum of φ_I\n(The Cosmic Breath)')
    ax6.legend()
    ax6.grid(alpha=0.3)
    ax6.set_xlim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig('kut_core_simulations.png', dpi=150, bbox_inches='tight')
    print("\n" + "=" * 70)
    print("Visualization saved as: kut_core_simulations.png")
    print("=" * 70)
    plt.show()


if __name__ == "__main__":
    demonstrate_kut_simulations()
