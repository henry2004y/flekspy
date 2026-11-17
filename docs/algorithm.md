# Algorithm

## Key Features of FLEKS

- Multi-Scale Modeling: FLEKS is often used as part of the MHD-AEPIC (Magnetohydrodynamics with Adaptively Embedded Particle-in-Cell) framework. This hybrid method bridges the gap between fluid-based plasma models (MHD) and detailed kinetic plasma models (PIC), allowing for the best of both worlds: efficient simulations with localized kinetic accuracy.

- Adaptive Kinetic Regions: FLEKS features dynamically adaptive PIC regions. This means that the computationally demanding PIC simulations can be focused specifically on areas where kinetic effects are crucial, enabling efficient resource allocation. We now have Adaptive Mesh Refinement (AMR) support for FLEKS as well!

- Exascale-Ready: FLEKS is designed to take advantage of powerful exascale computing systems, allowing for large-scale, highly detailed plasma simulations.

- Accurate and Efficient Time-Stepping: FLEKS includes adaptive time-stepping, which adjusts the simulation time step based on the requirements for accuracy within different regions, enhancing simulation efficiency.

- Load Balancing and Noise Reduction: Particle splitting and merging algorithms maintain optimal computational load balancing across computing cores and help suppress statistical noise in simulations.

- Particle Trajectory Tracking: FLEKS includes a test-particle module, used for tracking individual particle trajectories in the dynamic electromagnetic fields.

The semi-implicit solver as in FLEKS is useful for problems that cross both the ion and electron scales where a large domain is needed for studying the systematic impact. These features of the current energy-conserving semi-implicit method are particularly well-suited for multi-scale problems where resolving electron scales is computationally infeasible, but retaining some kinetic electron effects (like Landau damping, whistler waves) is physically necessary.

## Limitations for Classical PIC Models

Particle-In-Cell (PIC) models are incredibly useful for plasma simulation, but they do come with limitations imposed by numerical stability requirements. Let's break down those requirements and their impact:

- **Time Step**: The Courant-Friedrichs-Lewy (CFL) condition in numerical simulations dictates that a particle (or information) shouldn't travel more than one grid cell within a single time step.  Mathematically, for a 1D simulation, it looks like: $\Delta t \le \Delta x / v_\mathrm{max}$ where $\Delta t$ is the time step, $\Delta x$ is the grid cell size, and $v_\mathrm{max}$ is the maximum velocity of particles in the system. If the CFL condition is violated, particles can artificially "jump" over cells, leading to inaccurate results and potential instabilities.

- **Resolving the Debye Length**: The Debye length ($\lambda_D$) is a fundamental length scale in plasmas, representing the distance over which charged particles shield each other's electric fields. To accurately model plasma behavior, your grid resolution needs to be smaller than the Debye length: $\Delta x < \lambda_D$. Failure in resolving $\lambda_D$ results in the finite-grid instability, as demonstrated in the famous PIC simulation book. This under-resolution introduces errors in the calculation of the electric field and charge density, leading to unphysical heating of the plasma over time. The instability manifests as an artificial growth rate in the plasma wave dispersion relation, causing the plasma to become hotter than it should be. This can significantly affect the accuracy and reliability of the simulation results. The energy-conserving PIC algorithms can help to suppress the instability.

- **Resolving the Plasma Frequency**: The plasma frequency ($\omega_p$) represents the natural oscillation frequency of electrons in a plasma. Your simulation time step should be small enough to resolve this: $\Delta t < 1 / \omega_p$. Failing to meet this leads to inaccurate representation of plasma oscillations and potential numerical heating.

- **Satisfying the Gauss' Law**: Since in classical PIC algorithms we only solve for Faraday's law and Ampere's law, the charge conservation is not guaranteed. Gauss's law is critical in the conservation of both momentum and energy.

Finer grid resolutions and smaller time steps directly increase the number of calculations needed in a simulation. This translates to significantly higher computational demands. For plasmas containing extremely fast particles, the CFL condition can force you to use impractically small time steps to maintain stability.

Many modern PIC codes use relatively high order shape functions, and as a result, the worst-case numerical growth rates are undetectably small; in addition, some codes use energy-conserving methods which usually prevent this numerical instability from arising.

### How to Loose The Numerical Constraints?

How to handle the limitations of classical PIC algorithms? Why is FLEKS called a semi-implicit PIC model? To answer that, let us first review the differences of explicit and implicit schemes.

PIC methods can be explicit, semi-implicit or fully implicit. Plasmas are governed by two sets of equations: the equations for the motion of the particles and the equations for the evolution of the fields. The two sets of equations are coupled because the field equations need the sources (current and charge) from the particles, and the particles need the fields to compute the force. There is no specific ordering of the equations, which is also consistent with the essence of plasma physics: collective interaction of particles with the fields. Fully implicit methods are required to solve the discretized equations to solve the nonlinear field-particle interactions. In fully implicit methods, the particle equations of motion and the field equations are solved together within a nonlinear solver. In explicit methods, conversely, the coupling between particles and fields is suspended for a small time step. In that small time interval, one assumes that the known fields can be used unchanged for moving the particles, and the particle information can be used unchanged to evolve the fields. This assumption leads to the convenience of using explicit timestepping algorithms which has been proved to be the most efficient method in solving the equations.

However, in explicit PIC methods, timesteps are severely limited by numerical stability considerations. Specifically, not only the time resolution needs to be high enough to resolve the electron plasma frequency $\omega_{pe}$, $\Delta t < 2\omega_{pe}^{-1}$, but also the space resolution needs to cover the Debye length $\lambda_D$ to avoid the finite-grid instability ([@birdsall2018plasma]). The constraints are greatly loosened by the implicit approach.

Another consequence in following the explicit PIC methods is that energy is not conserved. Using a good resolution, energy is acceptably maintained. There is always a secular trend of energy increase, but as the resolution is relaxed closer to the stability limit of the finite grid instability, the energy increase becomes more severe, until at the instability limit, it starts to grow exponentially. [@birdsall2018plasma] This effect cannot be avoided, but it can be improved by using smoothing and higher-order interpolation techniques. Recent structure-preserving geometric particle-in-cell methods use symplectic integrators to ensure local energy conservation at small time steps. The implicit PIC method, instead, conserves energy exactly, whatever resolution is used. This feature is both physically important and practically useful, which serves as a requirement for running PIC models for sufficiently a long time. @barnes2021finite demonstrates that while EC-PIC algorithms are not free from the finite-grid instability for drifting plasmas in principle, they feature a benign stability threshold for finite-temperature plasmas that make them usable in practice for a large class of problems without the need to consider the size of the Debye length.

The semi-implicit PIC method tries to make a compromise and retain some of the advantages of both approaches. In semi-implicit methods, the particles and the fields are still advanced together, and an iteration is needed, but the coupling is linearized, and the iteration uses linear solvers. [@lapenta2023advances] With the addition of Gauss' law conservation correction ([@chen2019gauss]), the semi-implicit energy conserving PIC method has shown improved numerical stability performance.

- **Explicit PIC**: In traditional explicit PIC, field values at the _current_ time step are used to calculate forces on particles. These forces then determine how particles move to the next time step. The fields are then updated based on this new particle distribution. This explicit approach leads to the strict stability constraints (CFL, resolving Debye length, etc.) we discussed earlier.

- **Fully Implicit PIC**: To get around stability limitations, fully implicit PIC models solve a system of equations where fields at the _future_ time step are considered unknowns along with _future_ particle positions.  This means the movement of particles and the update of fields are implicitly linked. Solution involves large matrix inversions at each time step.

- **Semi-Implicit PIC**[^common]: Semi-implicit methods aim for a middle ground that retains much of the stability of fully implicit PIC while keeping each step linear. Earlier 1980s formulations linearize the particle--field coupling via fluid closures or Taylor expansions of the particle push, whereas FLEKS uses the energy-conserving semi-implicit method (ECSIM / GL-ECSIM):
  - **Classical linearized semi-implicit methods (1980s)**:
    1. *Implicit Moment Method*: Approximates the future current using a set of fluid moment equations derived from the particle distribution function.
    2. *Direct Implicit Method*: Performs a formal linear Taylor expansion of the particle orbit and interpolation operator with respect to the future electric field.

    Because both approximations break exact work symmetry between particles and fields, they lack exact discrete energy conservation and can still suffer numerical heating when $\Delta x \gg \lambda_D$.
  - **Energy-Conserving Semi-Implicit PIC (ECSIM / GL-ECSIM, used in FLEKS)**: Keeps Maxwell's equations and the particle velocity response to $\mathbf{E}^{n+\theta}$ **implicit**, while keeping particle positions frozen at the staggered half-step $\mathbf{x}_p^{n+1/2}$ (and the magnetic field in the Lorentz rotation lagged at $\mathbf{B}^n$) during the field solve. Because $\mathbf{x}_p^{n+1/2}$ is fixed during the solve, the average particle velocity $\bar{\mathbf{v}}_p$ is an **exact linear** function of $\mathbf{E}^{n+\theta}$ with no Taylor expansion, eliminating all particle velocities analytically into a nodal mass matrix for a linear GMRES grid solve, followed by a single back-substitution particle push. At $\theta = 0.5$ (with no artificial longitudinal damping), total energy is conserved to solver tolerance regardless of $\Delta x / \lambda_D$.

| Feature | Fully Implicit PIC | Semi-Implicit PIC (GL-ECSIM) |
|---|---|---|
| Stability | Unconditionally stable on fields and orbits, but requires nonlinear particle--field iteration | Removes light-wave CFL ($c'\Delta t/\Delta x$), $\omega_{pe}\Delta t$, and $\Delta x/\lambda_D$ limits for $\theta \ge 0.5$; still limited by particle thermal CFL ($u_{\max}\Delta t/\Delta x < 1$) |
| Accuracy | Fully time-centered in both $\mathbf{x}_p$ and $\mathbf{v}_p$; exact energy/charge conservation possible | Exact discrete energy conservation at $\theta = 0.5$; $\mathcal{O}(\Delta t\,\partial_t\mathbf{B})$ error from lagged $\mathbf{B}^n$ in rotation; `#DIVE` maintains Gauss's law |
| Computational Cost | Highest, due to nonlinear JFNK/Picard iterations re-pushing particles every solve | One mass-matrix deposit, one linear GMRES grid solve, and one back-substitution particle push per step |

[^common]: One common misconception is that in FLEKS's semi-implicit method, the field solver is implicit while the particle pusher is explicit. Treating fields implicitly with an explicit current $\mathbf{J}^n$ would leave plasma oscillations explicit ($\omega_{pe}\Delta t < 2$, $\Delta x \lesssim \lambda_D$). Instead, both fields and particle velocities are coupled implicitly during the field solve, while particle positions $\mathbf{x}_p^{n+1/2}$ and the rotation field $\mathbf{B}^n$ remain explicit; see [What "Semi-Implicit" Means in FLEKS](#what-semi-implicit-means-in-fleks).

## Unit Conversion

A physical quantity consists of two parts: a value and a unit. For example, the weight of a table can be expressed as $w = 50\,\text{kg} = 50,000\,\text{g}$, where 50 and 50,000 are the values, and kg and g are the units. If we define a new unit, wg, such that $1\,\text{wg} = 2.5\,\text{kg}$, the table's weight becomes $w = 20\,\text{wg}$. In general, a physical quantity $u$ can be expressed as $u = \bar{u} u^*$, where $\bar{u}$ is its numerical value and $u^*$ is its unit.

### Lorentz Equation

**Base Normalization Scales**

- Mass: $m_0 = m_p$ (the proton mass)
- Charge: $q_0 = e$ (the elementary charge)
- Time: $t_0 = 1 / \omega_{pp}$ (the inverse of the proton plasma frequency, $\omega_{pp} = \sqrt{n_0 e^2 / (m_p \epsilon_0)}$ in SI)
- Length: $x_0 = c / \omega_{pp}$ (the proton inertial length, $d_p$)

**Normalized (Code) Variables**

This new base changes our normalized variables.

- Time: $\tilde{t} = t / t_0$
- Position: $\tilde{\vec{x}} = \vec{x} / x_0$
- Mass: $\tilde{m} = m / m_0 = m / m_p$
- Charge: $\tilde{q} = q / q_0 = q / e$

In FLEKS, we normalize to the ion scale.

- For a proton: $\tilde{m} = m_p / m_p = 1$ and $\tilde{q} = e / e = 1$.
- For an electron: $\tilde{m} = m_e / m_p \approx 1/1836$ (physical) and $\tilde{q} = -e / e = -1$.

The derived normalization constants are:

- Velocity: $v_0 = x_0 / t_0 = (c / \omega_{pp}) / (1 / \omega_{pp}) = c$.
    - $\tilde{\vec{v}} = \vec{v} / c$
- Momentum: $p_0 = m_0 v_0 = m_p c$.
    - $\tilde{\vec{p}} = \vec{p} / (m_p c)$
    
The relativistic relationship $\vec{p} = \gamma m \vec{v}$ becomes $\tilde{\vec{p}} = \tilde{\gamma} \tilde{m} \tilde{\vec{v}}$, where $\tilde{\gamma} = (1 - \tilde{v}^2)^{-1/2}$.

**Derivation from SI Units**

1. Physical Equation (SI):
$\frac{d\vec{p}}{dt} = q(\vec{E} + \vec{v} \times \vec{B})$

2. Substitute Normalized Variables:
$\frac{d(\tilde{\vec{p}} p_0)}{d(\tilde{t} t_0)} = (\tilde{q} q_0) \left( (\tilde{\vec{E}} E_0) + (\tilde{\vec{v}} v_0) \times (\tilde{\vec{B}} B_0) \right)$

3. Isolate Normalized Equation:
$\left[ \frac{p_0}{t_0} \right] \frac{d\tilde{\vec{p}}}{d\tilde{t}} = (\tilde{q} q_0 E_0) \tilde{\vec{E}} + (\tilde{q} q_0 v_0 B_0) (\tilde{\vec{v}} \times \tilde{\vec{B}})$

$\frac{d\tilde{\vec{p}}}{d\tilde{t}} = \tilde{q} \left[ \frac{q_0 E_0 t_0}{p_0} \right] \tilde{\vec{E}} + \tilde{q} \left[ \frac{q_0 v_0 B_0 t_0}{p_0} \right] (\tilde{\vec{v}} \times \tilde{\vec{B}})$

4. Define $E_0$ and $B_0$ (Set brackets to 1):
    - $E_0 = \frac{p_0}{q_0 t_0} = \frac{m_p c}{e (1 / \omega_{pp})} = \frac{m_p c \omega_{pp}}{e}$
    - $B_0 = \frac{p_0}{q_0 v_0 t_0} = \frac{m_p c}{e c (1 / \omega_{pp})} = \frac{m_p \omega_{pp}}{e}$

**Derivation from CGS Units**

1. Physical Equation (CGS):
$\frac{d\vec{p}}{dt} = q \left( \vec{E} + \frac{\vec{v}}{c} \times \vec{B} \right)$

2. Substitute Normalized Variables:
$\frac{d(\tilde{\vec{p}} p_0)}{d(\tilde{t} t_0)} = (\tilde{q} q_0) \left( (\tilde{\vec{E}} E_0) + \frac{(\tilde{\vec{v}} v_0)}{c} \times (\tilde{\vec{B}} B_0) \right)$

3. Substitute $v_0 = c$:
$\left[ \frac{p_0}{t_0} \right] \frac{d\tilde{\vec{p}}}{d\tilde{t}} = (\tilde{q} q_0 E_0) \tilde{\vec{E}} + \tilde{q} \left[ \frac{q_0 c B_0}{c} \right] (\tilde{\vec{v}} \times \tilde{\vec{B}})$

The $c$ in the magnetic term cancels out perfectly.

4. Isolate Normalized Equation:
$\left[ \frac{p_0}{t_0} \right] \frac{d\tilde{\vec{p}}}{d\tilde{t}} = (\tilde{q} q_0 E_0) \tilde{\vec{E}} + (\tilde{q} q_0 B_0) (\tilde{\vec{v}} \times \tilde{\vec{B}})$

$\frac{d\tilde{\vec{p}}}{d\tilde{t}} = \tilde{q} \left[ \frac{q_0 E_0 t_0}{p_0} \right] \tilde{\vec{E}} + \tilde{q} \left[ \frac{q_0 B_0 t_0}{p_0} \right] (\tilde{\vec{v}} \times \tilde{\vec{B}})$

5. Define $E_0$ and $B_0$ (Set brackets to 1):
    - $E_0 = \frac{p_0}{q_0 t_0} = \frac{m_p c}{e (1 / \omega_{pp})} = \frac{m_p c \omega_{pp}}{e}$
    - $B_0 = \frac{p_0}{q_0 t_0} = \frac{m_p c}{e (1 / \omega_{pp})} = \frac{m_p c \omega_{pp}}{e}$ (Note: In CGS, $\omega_{pp} = \sqrt{4\pi n_0 e^2 / m_p}$, and $E_0=B_0$).

**Summary: The Final Normalized Equations**

In both SI and CGS, the final, unitless equations are identical:
\begin{gather}
\frac{d\tilde{\vec{x}}}{d\tilde{t}} &= \tilde{\vec{v}} \\
\frac{d\tilde{\vec{p}}}{d\tilde{t}} &= \tilde{q} (\tilde{\vec{E}} + \tilde{\vec{v}} \times \tilde{\vec{B}})
\end{gather}

The physics is captured by the normalized mass and charge parameters fed to the pusher:
    - Protons: $\tilde{m} = 1$, $\tilde{q} = 1$
    - Electrons: $\tilde{m} = m_e/m_p$, $\tilde{q} = -1$

In the non-relativistic limit, the speed of light does not appear in these equations, so the velocity unit $v_0$ can be chosen freely. FLEKS does not fix $x_0 = d_p$ and $v_0 = c$: the length and velocity units (`lNorm` and `uNorm`) are free parameters, and Maxwell's equations are solved with $c = 1$ in code units. The velocity unit is therefore also the simulated speed of light $c'$. See Consequences of Changing the Speed of Light below.

### Mass Normalization

In the CGS system, the equation of motion for a particle is:
\begin{equation}
\frac{d\mathbf{v}}{dt} = \frac{q}{m}\left(\mathbf{E} + \frac{\mathbf{v}}{c} \times \mathbf{B}\right)
\end{equation}
In FLEKS, all quantities are normalized in a way that avoids introducing new constants. From the equation of motion, we derive the following relation:
\begin{equation}
\frac{q_{cgs}^* E_{cgs}^* t_{cgs}^*}{m_{cgs}^* v_{cgs}^*} = 1
\end{equation}
From Gauss's law, $\nabla \cdot \mathbf{E} = 4\pi\rho$, we get:
\begin{equation}
E_{cgs}^* = \rho_{cgs}^* x_{cgs}^*
\end{equation}
where $\rho_{cgs}^* = q_{cgs}^*/(x_{cgs}^*)^3$ is the charge density, not the mass density. Combining these two equations, we obtain:
\begin{equation}
q_{cgs}^* = v_{cgs}^* \sqrt{m_{cgs}^* x_{cgs}^*}
\end{equation}

In the code, a proton has $\bar{q} = \bar{m} = 1$, and the code speed of light is 1. To keep the physical gyrofrequency $qB/(mc)$ with the physical $c$, the condition $\bar{q}/(\bar{m}\bar{c}) = 1$ is required for a proton, where $\bar{c} = c/v_{cgs}^*$ is the true speed of light in code units. This means the normalization units $q_{cgs}^*$ and $m_{cgs}^*$ must satisfy:
\begin{equation}
\frac{q_{cgs}^*}{m_{cgs}^* v_{cgs}^*} = \frac{q_{cgs,p}}{m_{cgs,p}c_{cgs}}
\end{equation}
where $q_{cgs,p}$ and $m_{cgs,p}$ are the charge and mass of a proton in CGS units, respectively. Using this, we find the normalization mass in CGS units:
\begin{equation}
m_{cgs}^* = x_{cgs}^* \left(\frac{c_{cgs}m_{cgs,p}}{q_{cgs,p}}\right)^2
\end{equation}
Note that $m_{cgs}^*$ includes both a numerical value and the unit "g". The expression above calculates this value. In the SI system, this value is calculated in `BATSRUS::ModPIC.f90`, where the proton's charge and mass are known. To use this, we must convert the expression to SI units. Let $m_{cgs}^*$ and $m_{SI}^*$ be the numerical values of the normalization mass in CGS and SI units, respectively. For a mass of 1.5 kg, $m_{SI}^* = 1.5$ and $m_{cgs}^* = 1500$, so the conversion is $m_{cgs}^* = 1000 m_{SI}^*$. For charge, assuming the speed of light is $3 \times 10^8 \text{ m/s}$, we have $1 \text{ C} = 3 \times 10^9 \text{ esu}$. This gives us:
\begin{equation}
1000 m_{SI}^* = 100 x_{SI}^* \left(\frac{100 c_{SI} \cdot 1000 m_{SI,p}}{3 \times 10^9 q_{SI,p}}\right)^2
\end{equation}
\begin{equation}
m_{SI}^* = 10^7 x_{SI}^* \left(\frac{m_{SI,p}}{q_{SI,p}}\right)^2
\end{equation}

Once $m_{SI}^*$ is obtained, it is passed to FLEKS and converted to $m_{cgs}^*$.

### Length and Velocity Normalization

These two are free parameters.

### Charge and Current Normalization

The charge normalization is given by $q_{cgs}^* = v_{cgs}^* \sqrt{m_{cgs}^* x_{cgs}^*}$, as shown previously. The current density normalization is $j_{cgs}^* = q_{cgs}^* v_{cgs}^* / (x_{cgs}^*)^3$.

### B and E Normalization

In CGS units, B and E have the same dimension. Substituting the expression for $q_{cgs}^*$ into the first equation, we get:
\begin{equation}
E_{cgs}^* = B_{cgs}^* = v_{cgs}^* \sqrt{\frac{m_{cgs}^*}{(x_{cgs}^*)^3}}
\end{equation}

### Consequences of Changing the Speed of Light

**Summary.** In FLEKS, the speed of light of the simulation is the velocity unit, $c' = v_{cgs}^*$ (`uNorm`). Choosing $c' < c$ is equivalent to keeping all charges, masses and the vacuum permeability $\mu_0$ physical, and increasing the vacuum permittivity:
\begin{equation}
\epsilon_0' = \frac{1}{\mu_0 c'^2} = \epsilon_0 \left(\frac{c}{c'}\right)^2 .
\end{equation}
This is the only change. A quantity is modified if and only if it depends on $\epsilon_0$. The Lorentz force, gyromotion, inertial lengths, Alfvén speed and the MHD electric fields are kept. The plasma frequency, Debye length, charge separation, electric field energy and light waves are modified.

#### How $c'$ enters the code

- Maxwell's equations and the equation of motion are solved with $c = 1$ in code units (see the Maxwell equations in the Semi-Implicit Methodology section below). The velocity unit is therefore the simulated speed of light. A reduced speed of light is not a separate parameter: choosing a small velocity unit *is* choosing a small $c'$. In a stand-alone run, $c'$ is `uNorm` of `#NORMALIZATION`. In a coupled run, it is the velocity unit passed from the MHD code.
- The mass unit is computed with the **physical** $c$ (see Mass Normalization above). A proton with $\bar{q} = \bar{m} = 1$ in code units therefore has the physical $q/(mc)$, i.e. the physical gyrofrequency. With $\bar{c} \equiv c/c'$, the true speed of light in code units, this is the condition $\bar{q}/(\bar{m}\bar{c}) = 1$.
- As a consequence, the code charge of a particle is $c'/c$ times its true charge, while its code mass is the true mass. For the magnetic force this is exactly compensated by $1/c'$ in place of $1/c$. For Gauss's law it is not: the same number of particles carries $c'/c$ times less charge in code units.
- The MHD quantities $\rho$, $\mathbf{u}$, $p$ and $\mathbf{B}$ are converted to code units without any reference to $c$. The conversions of $\mathbf{J}$ and $\mathbf{E}$ use $c'$ (`Si2NoJ` and `Si2NoE` in `FluidInterface.cpp`). In SI language this is the $\mu_0$-fixed, $\epsilon_0'$-increased system above.

#### What is changed

| Quantity | Change for $c \to c'$ | Reason |
|---|---|---|
| Electric field produced by a given charge separation | $\times\,(c'/c)^2$ | Gauss's law, $\nabla\cdot\mathbf{E} = \rho_q/\epsilon_0'$ |
| $\delta n/n$ needed to sustain a given $\mathbf{E}$ | $\times\,(c/c')^2$ | same |
| Plasma frequency $\omega_{ps}$ | $\times\, c'/c$ | $\omega_{ps}^2 = n_s q_s^2/(\epsilon_0' m_s)$, or $\omega_{ps} = c'/d_s$ |
| $\omega_{pe}/\Omega_{ce}$ | $= c'/v_{Ae}$ | $v_{Ae} = B/\sqrt{\mu_0 n m_e}$ is the electron Alfvén speed |
| Debye length $\lambda_D$ | $\times\, c/c'$ | $\lambda_D = v_{th,e}/\omega_{pe}$; $\lambda_D/d_e = v_{th,e}/c'$ |
| Displacement current $\epsilon_0'\,\partial\mathbf{E}/\partial t$ | $\times\,(c/c')^2$ | Ampère's law |
| Vacuum EM waves | propagate at $c'$ | |
| Electric field energy density $\epsilon_0' E^2/2$ | $\times\,(c/c')^2$ for the same $\mathbf{E}$ | |
| Relativistic effects | $\gamma$ computed with $c'$ | only for relativistic test particles; see below |

Langmuir waves, Debye shielding, electron holes, double layers and other electrostatic or charge-separation phenomena therefore have modified frequencies, scales or amplitudes.

#### What is kept

| Quantity | Reason |
|---|---|
| Lorentz force $q(\mathbf{E} + \mathbf{v}\times\mathbf{B})$ for given SI fields; $\mathbf{E}\times\mathbf{B}$ drift | no $\epsilon_0$; the $\mathbf{E}$ conversion and the force law are both exact |
| Gyrofrequency $\Omega_{cs}$, gyroradius $r_{Ls}$ | $qB/m$ is physical |
| Alfvén, sound and thermal speeds | depend only on $B$, $\rho$, $p$ and $\mu_0$ |
| Inertial lengths $d_s = c/\omega_{ps} = \sqrt{m_s/(\mu_0 n_s q_s^2)}$ | $\epsilon_0$ cancels; equivalently $d_i = v_A/\Omega_{ci}$ |
| Faraday's law, $\mu_0\mathbf{J} = \nabla\times\mathbf{B}$ when the displacement current is small | no $\epsilon_0$ |
| Motional, Hall and $\nabla\cdot\mathbf{P}_e$ electric fields | set by the electron force balance (generalized Ohm's law); requires quasi-neutrality |
| Poynting flux $\mathbf{E}\times\mathbf{B}/\mu_0$ | no $\epsilon_0$ |

A common misreading is that the electric force is $\bar{c}$ times weaker because the pusher uses $d\bar{\mathbf{v}}/d\bar{t} = \bar{\mathbf{E}}$ instead of $\bar{c}\,\bar{\mathbf{E}}$. That compares Gaussian-unit fields defined with two different speeds of light. In Gaussian units $E$ itself scales as $1/c$, and this compensates the smaller code charge exactly. For a given electric field in SI units, the particle acceleration is physical. The ratio between the electric force and the $\mathbf{v}\times\mathbf{B}$ force is not altered. What is weakened is the electric field that a given charge separation produces, which is why $\omega_{pe}$ is lowered.

#### Why quasi-neutral physics survives

In a quasi-neutral plasma the electric field is whatever the electrons need for force balance. The plasma supplies it through a tiny charge separation, not through the vacuum response. From Gauss's law, a field $E \sim uB$ that varies on a scale $L$ needs
\begin{equation}
\frac{\delta n}{n} \sim \frac{u}{v_A}\,\frac{d_i}{L}\left(\frac{v_A}{c'}\right)^2 .
\end{equation}
Reducing $c'$ raises $\delta n/n$, but the electric field is unchanged as long as $\delta n/n \ll 1$. The constraint is tightest at electron scales ($L \sim d_e$, electron flows $\sim v_{Ae}$), where $\delta n/n \sim (v_{Ae}/c')^2$.

#### Why reduce $c$ in FLEKS

The reason is *not* the light-wave CFL or the $\omega_{pe}\Delta t$ limit of explicit PIC:

- The semi-implicit ($\theta \ge 0.5$) energy-conserving field solve is stable for $c'\Delta t/\Delta x > 1$ and $\omega_{pe}\Delta t \gg 1$. The particle push is not separable from the field solve (see the footnote on semi-implicit methods above).
- The full-PIC time step is set only by the particle CFL, $\Delta t = \mathrm{cfl}\cdot\Delta x/u_{max}$, where $u_{max}$ is the maximum thermal speed (`Pic.cpp`). The speed of light does not enter.
- The light CFL $c'\Delta t_{sub}\sqrt{\sum_d \Delta x_d^{-2}} < 1$ applies only to the explicit field solver option and its field subcycling.

The cost of a large $c'$ shows up in the implicit linear system. Both the $\delta^2\nabla\times\nabla\times$ term ($\delta = c'\theta\Delta t$) and the mass-matrix term (of order $(\omega_{pe}\theta\Delta t)^2$) grow as $c'^2$. The system becomes ill-conditioned, GMRES needs many more iterations, and the converged field is more sensitive to particle noise. A smaller $c'$ keeps this system well conditioned.

#### Choosing $c'$

$c'$ should exceed every physical speed by a comfortable margin, typically a factor of 10 or more:

1. $c' \gg u$, $v_A$, $v_{th,i}$.
2. $c' \gg v_{Ae}$ (equivalently $\omega_{pe} \gg \Omega_{ce}$). This keeps quasi-neutrality at electron scales, which matters for Hall fields and the electron diffusion region.
3. $c' \gg v_{th,e}$. This keeps $\lambda_D \ll d_e$.
4. $c'$ well above the fastest whistler phase speed you want to model, $\approx v_{Ae}/2$ at $k d_e \approx 1$. Otherwise the full-PIC R-mode deviates from the Hall-MHD whistler. In `tests/whistler` of FLEKS, $c'/v_A \approx 1$, so the full-PIC run measures the full cold-plasma R-mode rather than the Hall whistler.

$v_{Ae} = v_A\sqrt{m_i/m_e}$ uses the *simulated* mass ratio. A reduced mass ratio and a reduced $c'$ therefore have to be chosen together. For example, with $v_A = 100$ km/s and $m_i/m_e = 100$, $v_{Ae} = 1000$ km/s, and $c' = 3000$ km/s gives only $\omega_{pe}/\Omega_{ce} = 3$.

#### Interpreting the output

- $\mathbf{E}$, $\mathbf{B}$, $\mathbf{J}$, the moments and the Poynting flux converted to SI are physical, provided the plasma stays quasi-neutral.
- The net charge density (or $\nabla\cdot\mathbf{E}$), the electric field energy, and electrostatic oscillations and structures follow the "changed" table above. They should not be compared with observations without rescaling.
- The full-PIC mover is non-relativistic, so $c'$ is not a speed limit for the plasma particles. Particles faster than $c'$ indicate that $c'$ is too small.
- Relativistic test particles (`#TPRELATIVISTIC`) compute $\gamma$ with the code speed of light, i.e. $c'$, and stop with an error when a particle exceeds $c'$. Their relativistic corrections are physical only if $c'$ is the true speed of light.
- The model is appropriate for non-relativistic, low-frequency ($\omega \ll \omega_{pe}'$), quasi-neutral physics dominated by magnetic and inductive effects. It shares this regime with Darwin and magneto-inductive models, but unlike them it retains the (slowed) displacement current and light waves.

## Semi-Implicit Methodology

FLEKS uses the Gauss-Law satisfying energy conserving semi-implicit method (GL-ECSIM). Its semi-implicit nature allows the PIC model to run on a coarser grid with a larger time step than explicit PIC methods, which are limited by the stability conditions to $\Delta t < \Delta x/c$ and the grid resolution $\Delta x < \zeta \lambda_D$, where c is the speed of light, $\lambda_D$ is the Debye length, and $\zeta$ is of order one and depends on the numerical details of the method. GL-ECSIM facilitates the coupling between the MHD and the PIC models, and its energy conserving property helps eliminating numerical instabilities and spurious waves that would violate the energy conservation.

ECSIM uses a staggered grid, where the electric field is defined at cell nodes, and the magnetic field is defined at cell centers. The Maxwell's equations are solved implicitly:
\begin{align}
\frac{\mathbf{B}^{n+1} - \mathbf{B}^n}{\Delta t} &= - c\nabla\times\mathbf{E}^{n+\theta} \\
\frac{\mathbf{E}^{n+1} - \mathbf{E}^n}{\Delta t} &= + c\nabla\times\mathbf{B}^{n+\theta} - 4\pi\bar{\mathbf{J}}
\end{align}
where $\theta \in [0.5, 1]$ is the time centering parameter. $\bar{\mathbf{J}}$ is the predicted current density at $n + \frac{1}{2}$ time stage, and it can be expressed as a linear function of the unknown electric field $\mathbf{E}^{n+\theta}$.
The variables at time stage $n+\theta$ can be written as linear combinations of values at the time steps n and n + 1:
\begin{align}
\mathbf{E}^{n+\theta} &= (1-\theta)\mathbf{E}^n + \theta \mathbf{E}^{n+1} \\
\mathbf{B}^{n+\theta} &= (1-\theta)\mathbf{B}^n + \theta \mathbf{B}^{n+1}
\end{align}

After rearranging the equations above and using the identity $\nabla\times\nabla\times\mathbf{E}= \nabla(\nabla\cdot\mathbf{E}) - \nabla^2\mathbf{E}$, we come up with an equation of $\mathbf{E}^{n+\theta}$
\begin{equation}
\mathbf{E}^{n+\theta} + \delta^2\left[ \nabla(\nabla\cdot\mathbf{E}^{n+\theta}) - \nabla^2\mathbf{E}^{n+\theta} \right] = \mathbf{E}^n + \delta\left( \nabla\times\mathbf{B}^n - \frac{4\pi}{c}\bar{\mathbf{J}} \right)
\end{equation}
where $\delta = c\theta \Delta t$. After applying finite difference discretizations to the gradient and divergence operators, we obtain a linear system of equations for the discrete values of $\mathbf{E}^{n+\theta}$ at the cell nodes. The iterative generalized minimal residual method (GMRES) is used to solve the equations to obtain $\mathbf{E}^{n+\theta}$. Once $\mathbf{E}^{n+\theta}$ is obtained, the electric and magnetic fields are advanced to $t^{n+1}$ via $\mathbf{E}^{n+1} = \frac{1}{\theta}\mathbf{E}^{n+\theta} - \frac{1-\theta}{\theta}\mathbf{E}^n$ and $\mathbf{B}^{n+1} = \mathbf{B}^n - c\Delta t\,\nabla\times\mathbf{E}^{n+\theta}$.

The position and velocity of the macro-particle are staggered in time, i.e., the particle velocity $\mathbf{v}_p$ is at the integer time stage and the location $\mathbf{x}_p$ is at the half time stage. First the velocity is pushed to time level n + 1 by
\begin{equation}
\mathbf{v}_p^{n+1} = \mathbf{v}_p^n + \frac{q_p \Delta t}{m_p}\left( \mathbf{E}^{n+\theta}(\mathbf{x}_p^{n+1/2}) + \frac{\mathbf{v}_p^n + \mathbf{v}_p^{n+1}}{2}\times\mathbf{B}^n(\mathbf{x}_p^{n+1/2}) \right)
\end{equation}

The fields $\mathbf{E}^{n+\theta}(\mathbf{x}_p^{n+1/2})$ and $\mathbf{B}^{n}(\mathbf{x}_p^{n+1/2})$ are interpolated to the particle locations $\mathbf{x}_p^{n+1/2}$. $q_p$ and $m_p$ are the charge and mass of particle species p. Finally, the particle position is updated to a preliminary new position
\begin{equation}
\tilde{\mathbf{x}}_p^{n+3/2} = \mathbf{x}_p^{n+1/2} + \Delta t \mathbf{v}_p^{n+1}
\end{equation}

Because Gauss's law is not automatically satisfied or controlled in the original ECSIM algorithm, artificial effects can develop in long simulations. @chen2019gauss proposed several methods to reduce the error and eliminate the artifacts.

The above equations require the new particle velocities to compute the current in the Maxwell's equations and the particle pusher requires the new advanced electric field to move the particles. In the spirit of the semi-implicit method, we do not want to solve two coupled sets with a single nonlinear iteration and find instead a way to extract analytically from the equations of motion the information needed for computing the current without first moving the particles. In previous semi-implicit methods this is done via a linearization procedure. The new mover used here allows us to derive the current rigorously without any approximation.

The current for each species s in each grid location g is
\begin{equation}
\bar{\mathbf{J}}_{sg} = \frac{1}{V_g} \sum_{p\in s}q_p \frac{\mathbf{v}_p^n + \mathbf{v}_p^{n+1}}{2} W(\mathbf{x}_p^{n+1/2} - \mathbf{x}_g) = \frac{1}{V_g} \sum_{p\in s}q_p \frac{\mathbf{v}_p^n + \mathbf{v}_p^{n+1}}{2} W_{pg}
\end{equation}
where the summation is over particles of the same species s.

Manipulating the vectors[^v-avg], the average velocity between time steps n and n+1 can be rewritten as
\begin{equation}
\frac{\mathbf{v}_p^n + \mathbf{v}_p^{n+1}}{2} = \hat{\mathbf{v}}_p + \beta_s\hat{\mathbf{E}}_p
\end{equation}
where the hatted quantities have been rotated by the magnetic field
\begin{align}
\hat{\mathbf{v}}_p &= \pmb{\alpha}_p^n \mathbf{v}_p^n \\
\hat{\mathbf{E}}_p &= \pmb{\alpha}_p^n \mathbf{E}_p^{n+\theta}
\end{align}
via a rotation matrix defined as
\begin{equation}
\pmb{\alpha}_p^n = \frac{1}{1 + (\beta_s B_p^n)^2} \left( \overleftrightarrow{I} - \beta_s \overleftrightarrow{I}\times\mathbf{B}_p^n +\beta_s^2\mathbf{B}_p^n \mathbf{B}_p^n \right)
\end{equation}
where $\overleftrightarrow{I}$ is the dyadic tensor and $\beta_s = q_p \Delta t / 2m_p$. The elements of the rotation matrix are indicated as $\alpha_p^{ij,n}$ with label i and j referring to the 3 components of the vector space (x, y, z).

[^v-avg]: Because $\bar{\mathbf{v}}_p$ is strictly linear in $\mathbf{E}_p^{n+\theta}$ with no nonlinear trajectory dependence, all particle velocities are eliminated analytically into the mass matrix before GMRES runs, and $\mathbf{v}_p^{n+1}$ is evaluated afterward in a single non-iterative back-substitution pass.

Substituting this linear expression for the average velocity into the species current definition gives, without any approximation or linearization:
\begin{equation}
\bar{\mathbf{J}}_{sg} = \frac{1}{V_g}\sum_p q_p \hat{\mathbf{v}}_p W_{pg} + \frac{\beta_s}{V_g}\sum_p q_p \hat{\mathbf{E}}_p^{n+\theta} W_{pg}
\end{equation}
where the summation is intended over all particles of species s.

Let
\begin{equation}
\hat{\mathbf{J}}_{sg} = \sum_p q_p \hat{\mathbf{v}}_p W_{pg}
\end{equation}
be the tentative current based on the hatted velocity; together with $\hat{\mathbf{E}}_p = \pmb{\alpha}_p^n \mathbf{E}_p^{n+\theta}$, we have
\begin{equation}
\bar{\mathbf{J}}_{sg} = \hat{\mathbf{J}}_{sg} + \frac{\beta_s}{V_g}\sum_p q_p \pmb{\alpha}_p^n \mathbf{E}_p^{n+\theta} W_{pg}
\end{equation}

The fields at the particle positions are computed by interpolation:
\begin{equation}
\mathbf{E}_p^{n+\theta} = \mathbf{E}^{n+\theta}(\mathbf{x}_p^{n+1/2}) = \sum_g \mathbf{E}_g^{n+\theta} W_{pg}
\end{equation}

It follows that
\begin{equation}
\bar{\mathbf{J}}_{sg} = \hat{\mathbf{J}}_{sg} + \frac{\beta_s}{V_g}\sum_p\sum_{g^\prime} q_p \pmb{\alpha}_p^n \mathbf{E}_{g^\prime}^{n+\theta}W_{pg^\prime} W_{pg}
\end{equation}

Exchanging the order of summation and introducing the elements of the mass matrices as
\begin{equation}
M_{s,gg^\prime}^{ij} \equiv \sum_p q_p \alpha_p^{ij,n}W_{pg^\prime} W_{pg}
\end{equation}
we obtain, in matrix form
\begin{equation}
\bar{\mathbf{J}}_{sg} = \hat{\mathbf{J}}_{sg} + \frac{\beta_s}{V_g}\sum_{g^\prime} M_{s,gg^\prime}\mathbf{E}_{g^\prime}^{n+\theta}
\end{equation}

The mass-matrix definition above gives the elements of the coupling matrices that are the most distinctive feature of the method. There are $3v$ such matrices, where $v$ is the dimensionality of the magnetic field and velocity vector, not to be confused with the spatial dimensionality $d$. The indices $i$ and $j$ vary in the $3v$-space—for example, for full 3-component vectors there are 9 component matrices—and each matrix is symmetric and sparse with only $2d$ diagonals (compact nearest-neighbor stencil support).

(what-semi-implicit-means-in-fleks)=
### What "Semi-Implicit" Means in FLEKS

A natural misreading of the step order is that the field solver is implicit while the particle pusher is explicit. If Maxwell's equations were solved implicitly with an explicit current $\mathbf{J}^n$, the plasma oscillation loop $\partial_t\mathbf{E} = -4\pi\mathbf{J}$ and $\partial_t\mathbf{v} = (q/m)\mathbf{E}$ would remain explicit, so the scheme would still blow up whenever $\omega_{pe}\Delta t > 2$ or $\Delta x \gg \lambda_D$ (the finite-grid instability).

Instead, in FLEKS's ECSIM formulation, **semi-implicit** refers to how the particle phase space $(\mathbf{x}_p, \mathbf{v}_p)$ and the Lorentz force are split between implicit and explicit treatment within each time step:

| Part of the Vlasov--Maxwell step | Treatment in FLEKS | Consequence |
|---|---|---|
| Maxwell curl equations ($\mathbf{E}^{n+\theta}, \mathbf{B}^{n+\theta}$) | **Implicit** | Eliminates the light-wave CFL ($c'\Delta t/\Delta x$) for $\theta \ge 0.5$ |
| Particle velocity response to $\mathbf{E}$ ($\bar{\mathbf{v}}_p \leftrightarrow \mathbf{E}^{n+\theta}$) | **Implicit** | Eliminates the $\omega_{pe}\Delta t$ and $\Delta x/\lambda_D$ plasma constraints |
| Particle position $\mathbf{x}_p$ in shape weights $W(\mathbf{x}_p - \mathbf{x}_g)$ and drift | **Explicit** (frozen at staggered $\mathbf{x}_p^{n+1/2}$) | Keeps the grid system linear; retains the particle thermal CFL ($u_{\max}\Delta t/\Delta x < 1$) |
| Magnetic field in Lorentz rotation ($\bar{\mathbf{v}}_p \times \mathbf{B}$) | **Explicit / lagged** at $\mathbf{B}^n(\mathbf{x}_p^{n+1/2})$ | Avoids quadratic nonlinearity in $\mathbf{E}^{n+\theta}$ while doing zero spurious work |

#### Why the Boris push is single-pass: exact Schur-complement back-substitution

Freezing particle positions at $\mathbf{x}_p^{n+1/2}$ and lagging the rotation field at $\mathbf{B}^n$ makes every particle's $3\times 3$ Boris velocity equation strictly linear in $\mathbf{E}^{n+\theta}$. Consequently, the coupled system over all particle velocities $\{\bar{\mathbf{v}}_p\}$, $\mathbf{B}^{n+1}$, and $\mathbf{E}^{n+\theta}$ is solved in three exact algebraic stages:

1. **Analytical elimination (`calc_mass_matrix`)**: Accumulating $\hat{\mathbf{J}}$ and the 27-point (9-point in 2D) nodal mass matrix $\mathbf{M}$ eliminates all $N_p$ particle velocities (and $\mathbf{B}^{n+1}$) analytically *before* GMRES runs, forming the Schur complement on the $3N_{\text{nodes}}$ grid unknowns.
2. **Grid solve (`update_E`)**: GMRES solves the linear system for $\mathbf{E}^{n+\theta}$ on the grid, with the linearized plasma response already embedded in $\mathbf{M}$.
3. **Back-substitution (`particle_mover`)**: Plugging the converged $\mathbf{E}^{n+\theta}$ into the Boris mover yields $\mathbf{v}_p^{n+1}$ in a single pass, followed by the explicit drift $\mathbf{x}_p^{n+3/2} = \mathbf{x}_p^{n+1/2} + \Delta t\,\mathbf{v}_p^{n+1}$. This single-pass velocity push is exact algebraic back-substitution of the implicit velocity solve, not an explicit velocity step.

#### Solver properties that follow from this split

1. **Eliminated electromagnetic and plasma-scale stability constraints**: Because both Maxwell wave propagation and the $\mathbf{v}_p \leftrightarrow \mathbf{E}^{n+\theta}$ response are implicit, the scheme is stable for $c'\Delta t/\Delta x > 1$, $\omega_{pe}\Delta t \gg 1$, and $\Delta x \gg \lambda_D$ whenever $\theta \ge 0.5$.
2. **Remaining particle thermal CFL**: Because $\mathbf{x}_p$ is frozen during the solve and drifted explicitly afterward, $\Delta t$ remains bounded by particle streaming across a cell ($u_{\max}\Delta t/\Delta x < 1$, typically $\mathrm{cfl} \approx 0.1\text{--}0.4$).
3. **Exact work cancellation and freedom from finite-grid heating**: Evaluating both the mass matrix and the Boris mover at the same frozen $\mathbf{x}_p^{n+1/2}$ and $\mathbf{B}^n$ makes $\bar{\mathbf{J}}$ in Ampère's law match the mover's current to solver tolerance, while $\bar{\mathbf{v}}_p \cdot (\bar{\mathbf{v}}_p \times \mathbf{B}^n) \equiv 0$ ensures the magnetic rotation does zero spurious work.
4. **Why Gauss's-law cleaning (`#DIVE`) is needed**: Depositing $\bar{\mathbf{J}}$ at the frozen position $\mathbf{x}_p^{n+1/2}$ before drifting to $\mathbf{x}_p^{n+3/2}$ does not satisfy discrete charge continuity $\nabla\cdot\bar{\mathbf{J}} + (\rho^{n+1} - \rho^n)/\Delta t = 0$ identically (unlike trajectory-integrated Esirkepov currents in explicit PIC). `#DIVE` therefore corrects particle positions to enforce $\nabla\cdot\mathbf{E} = 4\pi\rho$.
5. **Lagged-$\mathbf{B}^n$ trade-off**: Keeping $\mathbf{B}^n$ explicit inside $\pmb{\alpha}_p^n$ avoids nonlinear coupling between $\bar{\mathbf{v}}_p$ and $\mathbf{B}^{n+\theta}$ and preserves exact energy conservation, at the cost of an $\mathcal{O}(\Delta t\,\partial_t\mathbf{B})$ truncation error in the magnetic rotation when $\mathbf{B}$ varies in time.

### Role of the Time-Centering Parameter $\theta$

In `#DISCRETIZATION`, the `theta` parameter ($\theta \in [0.5, 1.0]$, default `0.51`) directly sets the time-level weight $t^{n+\theta}$ in $\mathbf{E}^{n+\theta} = \theta\mathbf{E}^{n+1} + (1-\theta)\mathbf{E}^n$ and $\mathbf{B}^{n+\theta} = \theta\mathbf{B}^{n+1} + (1-\theta)\mathbf{B}^n$. Two common points of confusion arise around this parameter:

1. **$\theta = 0.5$ (not $\theta = 0$) is time-centered Crank--Nicolson**: Some implicit PIC literature parameterizes decentering relative to the half-step as $t^{n+1/2+\delta}$ with $\delta = \theta - 0.5$, where $\delta = 0$ is centered. In FLEKS, $\theta$ is the direct weight in $[0.5, 1.0]$: values $\theta < 0.5$ are unstable for oscillatory modes, and `theta = 0` divides by zero when reconstructing $\mathbf{E}^{n+1} = \frac{1}{\theta}\mathbf{E}^{n+\theta} - \frac{1-\theta}{\theta}\mathbf{E}^n$.
2. **Unrelated upwind-limiter parameters**: `#UPWINDE` (`limiterThetaE`) and `#UPWINDB` (`theta`) are separate Sweby flux-limiter parameters in $[0, 2]$, where `0` gives first-order Lax--Friedrichs upwinding and `1`--`2` give second-order TVD limiters.

#### Discrete energy identity

Regardless of $\theta$, FLEKS's Boris push and mass matrix always use the half-step factor $\beta_s = q_p\Delta t/(2m_p)$ and midpoint velocity $\bar{\mathbf{v}}_p = \frac{1}{2}(\mathbf{v}_p^n + \mathbf{v}_p^{n+1})$. Consequently, `theta` shifts the time centering of the **electromagnetic fields**, not the midpoint centering of the particle velocity update, so the discrete kinetic energy change is always $\Delta K = \Delta t\sum_{s,g} V_g \bar{\mathbf{J}}_{sg}\cdot\mathbf{E}_g^{n+\theta}$. Combining this with the $\theta$-weighted Maxwell equations (when `ratioDivC2C = 0`, removing the optional extended-stencil longitudinal operator in `#DISCRETIZATION`) yields the exact discrete energy identity for all $\theta$:
\begin{equation}
\Delta (K + W_E + W_B) = -\left(\theta - \frac{1}{2}\right)\frac{1}{4\pi}\int \left(|\mathbf{E}^{n+1} - \mathbf{E}^n|^2 + |\mathbf{B}^{n+1} - \mathbf{B}^n|^2\right) dV .
\end{equation}

#### Temporal order of accuracy: dispersion vs. damping

Expanding $\mathbf{E}^{n+\theta}$ around the midpoint $t^{n+1/2}$,
\begin{equation}
\mathbf{E}^{n+\theta} = \frac{\mathbf{E}^{n+1} + \mathbf{E}^n}{2} + \left(\theta - \frac{1}{2}\right)\Delta t\,\partial_t\mathbf{E}(t^{n+1/2}) + \mathcal{O}(\Delta t^2) ,
\end{equation}
shows how $\theta$ controls formal convergence:

- **At $\theta = 0.5$** (with `ratioDivC2C = 0` and `#DIVE` off), the $(\theta - 0.5)\Delta t$ term vanishes, giving **second-order temporal accuracy** ($\mathcal{O}(\Delta t^2)$) for vacuum electromagnetic waves, electrostatic modes, or static $\mathbf{B}$ (whereas time-varying $\mathbf{B}$ still carries the $\mathcal{O}(\Delta t\,\partial_t\mathbf{B})$ lagged-rotation error, and default `#DIVE` uses a fixed `0.51` charge-interpolation weight).
- **For any fixed $\theta > 0.5$**, the $(\theta - 0.5)\Delta t$ term makes the formal asymptotic global error **first-order**, $\mathcal{O}((\theta - 0.5)\Delta t)$.
- **Wave phase vs. amplitude**: For a linear oscillatory mode $e^{-i\omega t}$, the $\theta$-scheme amplification factor $g = \frac{1 - i(1-\theta)\omega\Delta t}{1 + i\theta\omega\Delta t}$ yields the numerical frequency
  \begin{equation}
  \omega_{\mathrm{num}} = \omega\left[1 - \left(\frac{1}{12} + \left(\theta - \frac{1}{2}\right)^2\right)(\omega\Delta t)^2 + \mathcal{O}(\Delta t^4)\right] - i\left(\theta - \frac{1}{2}\right)\omega^2\Delta t .
  \end{equation}
  Thus **wave phase and dispersion remain second-order accurate ($\mathcal{O}(\Delta t^2)$) for all $\theta$**; the first-order truncation term enters purely as amplitude damping with prefactor $\theta - 0.5$.

#### Meaning of $\theta = 0.5$, $\theta = 0.51$, and $\theta = 1.0$

- **$\theta = 0.5$ (Time-symmetric Crank--Nicolson)**: Fields, frozen particle positions $\mathbf{x}_p^{n+1/2}$, and midpoint current $\bar{\mathbf{J}}$ all align at $t^{n+1/2}$, giving exact energy conservation ($\Delta E_{\mathrm{tot}} = 0$). However, the scheme is A-stable rather than L-stable ($|g(\omega\Delta t)| \equiv 1$ at all frequencies). Because $\mathbf{E}^{n+1} = 2\mathbf{E}^{n+1/2} - \mathbf{E}^n$, unresolved stiff modes ($\omega_{pe}\Delta t \gg 1$ or $c'k\Delta t \gg 1$) pinned near equilibrium at the half-step satisfy $\delta\mathbf{E}^{n+1} \approx -\delta\mathbf{E}^n$ ($g \to -1$), so PIC shot-noise, boundary, and AMR transients ring indefinitely as $(-1)^n$ oscillations.
- **$\theta = 1.0$ (Backward-Euler fields + midpoint Boris push)**: Fields are evaluated directly at the end of the step ($\mathbf{E}^{n+\theta} = \mathbf{E}^{n+1}$, $\mathbf{B}^{n+\theta} = \mathbf{B}^{n+1}$) with no extrapolation. Unresolved electromagnetic modes ($|g| = (1 + c'^2k^2\Delta t^2)^{-1/2} \to 0$ as $c'k\Delta t \to \infty$) and the stiff plasma electric-field root ($g_1 \approx -2/(\omega_{pe}\Delta t)^2 \to 0$ as $\omega_{pe}\Delta t \to \infty$) are damped in a single step, while the midpoint Boris velocity update retains a zero-current particle-velocity root $g_2 \to -1$. The drawback is heavy dissipation: every step removes $\frac{1}{8\pi}\int(|\Delta\mathbf{E}|^2 + |\Delta\mathbf{B}|^2)\,dV$ of total energy, draining electron thermal energy through PIC shot-noise field fluctuations, damping resolved waves at rate $\gamma = \frac{1}{2}\omega^2\Delta t$ ($50\times$ stronger than at $\theta = 0.51$), and stiffening the GMRES operator (a $4\times$ larger curl--curl term and $2\times$ larger mass-matrix term than at $\theta = 0.5$).
- **$\theta \approx 0.51\text{--}0.6$ (Practical default `0.51`)**: Shifts the stiff amplification factor to $|g(\infty)| = (1 - \theta)/\theta < 1$ ($\approx 0.96$ at $\theta = 0.51$), damping unresolved $(-1)^n$ oscillations over a few dozen steps while reducing the per-step energy sink by $50\times$ relative to $\theta = 1.0$.

| Property | $\theta = 0.5$ | $\theta = 0.51$ (default) | $\theta = 1.0$ |
|---|---|---|---|
| Field evaluation | Midpoint $\frac{1}{2}(\mathbf{E}^n + \mathbf{E}^{n+1})$ | Slightly future-biased $t^{n+0.51}$ | End-of-step $\mathbf{E}^{n+1}$ (no extrapolation) |
| Energy change per step | $0$ (exact conservation) | $-0.01\times\frac{1}{4\pi}(\|\Delta\mathbf{E}\|^2 + \|\Delta\mathbf{B}\|^2)$ | $-0.50\times\frac{1}{4\pi}(\|\Delta\mathbf{E}\|^2 + \|\Delta\mathbf{B}\|^2)$ |
| Stiff limit ($\omega\Delta t \to \infty$) | $g \to -1$ (undamped $(-1)^n$ ringing) | $g \to -0.49/0.51 \approx -0.96$ (decays in ~25 steps) | Field root $g \to 0$ (damped in 1 step) |
| Resolved waves ($\omega\Delta t \ll 1$) | 2nd-order phase & amplitude; zero damping | 2nd-order phase; weak damping ($\gamma = 0.01\,\omega^2\Delta t$) | 2nd-order phase; strong 1st-order damping ($\gamma = 0.5\,\omega^2\Delta t$) and electron cooling |

## Particle Mover

### Boris Algorithm

FLEKS uses the Boris particle mover algorithm:
\begin{equation}
\frac{\mathbf{v}^{n+1} - \mathbf{v}^{n}}{\Delta t} = \frac{q}{m}\left(\mathbf{E}^{n+\theta} + \bar{\mathbf{v}} \times \mathbf{B}\right)
\end{equation}
\begin{equation}
\frac{\mathbf{x}^{n+1/2} - \mathbf{x}^{n-1/2}}{\Delta t} = \mathbf{v}^n
\end{equation}
where $\bar{\mathbf{v}} = (\mathbf{v}^{n+1} + \mathbf{v}^{n}) / 2$. The traditional method for updating the velocity involves the following steps:

1.  **Acceleration:**
    \begin{equation}
    \mathbf{v}^- = \mathbf{v}^{n} + \frac{q\Delta t}{2m}\mathbf{E}^{n+\theta}
    \end{equation}

2.  **Rotation:**
    \begin{equation}
    \mathbf{a} = \mathbf{v}^- + \mathbf{v}^- \times \mathbf{t}
    \end{equation}
    \begin{equation}
    \mathbf{v}^+ = \mathbf{v}^- + \mathbf{a} \times \mathbf{s}
    \end{equation}
    where $\mathbf{t} = \frac{q\Delta t}{2m} \mathbf{B}$ and $\mathbf{s} = \frac{2\mathbf{t}}{1+t^2}$.

3.  **Acceleration:**
    \begin{equation}
    \mathbf{v}^{n+1} = \mathbf{v}^+ + \frac{q\Delta t}{2m}\mathbf{E}^{n+\theta}
    \end{equation}

FLEKS solves the same velocity equation, but with a slightly different implementation:

1.  **Acceleration:**
    \begin{equation}
    \mathbf{v}^- = \mathbf{v}^{n} + \frac{q\Delta t}{2m}\mathbf{E}^{n+\theta}
    \end{equation}

2.  **Calculate $\bar{\mathbf{v}}$:**

    From the expressions for $\mathbf{v}^+$ and $\mathbf{v}^-$, it is straightforward to show that $\mathbf{v}^{n+1} - \mathbf{v}^{n} = \mathbf{v}^+ - \mathbf{v}^- + \frac{q\Delta t}{m} \mathbf{E}^{n+\theta}$. Substituting this into the velocity update equation allows us to eliminate the electric field from the expression $\frac{\mathbf{v}^{+} - \mathbf{v}^{-}}{\Delta t} = \frac{q}{m}\bar{\mathbf{v}}\times\mathbf{B}$. Since $\mathbf{v}^+ = 2\bar{\mathbf{v}} - \mathbf{v}^-$, we get an equation for $\bar{\mathbf{v}}$:
    \begin{equation}
    \bar{\mathbf{v}} = \mathbf{v}^- + \frac{q\Delta t}{2m} \bar{\mathbf{v}} \times \mathbf{B} = \mathbf{v}^- + \bar{\mathbf{v}} \times \mathbf{t}
    \end{equation}

    To solve for $\bar{\mathbf{v}}$, we first take the dot product with $\mathbf{t}$ to get $\bar{\mathbf{v}} \cdot \mathbf{t} = \mathbf{v}^- \cdot \mathbf{t}$. Then, we take the cross product with $\mathbf{t}$ to get:
    \begin{equation}
    \bar{\mathbf{v}} \times \mathbf{t} = \mathbf{v}^- \times \mathbf{t} + (\bar{\mathbf{v}} \times \mathbf{t}) \times \mathbf{t} = \mathbf{v}^- \times \mathbf{t} + (\bar{\mathbf{v}} \cdot \mathbf{t})\mathbf{t} - t^2\bar{\mathbf{v}} = \mathbf{v}^- \times \mathbf{t} + (\mathbf{v}^- \cdot \mathbf{t})\mathbf{t} - t^2\bar{\mathbf{v}}
    \end{equation}
    Substituting the expression for $\bar{\mathbf{v}} \times \mathbf{t}$ back into the equation for $\bar{\mathbf{v}}$, we find:
    \begin{equation}
    \bar{\mathbf{v}} = \frac{\mathbf{v}^- + \mathbf{v}^- \times \mathbf{t} + (\mathbf{v}^- \cdot \mathbf{t})\mathbf{t}}{1+t^2}
    \end{equation}

3.  **Final Velocity:**
    It is then easy to obtain $\mathbf{v}^{n+1} = 2\bar{\mathbf{v}} - \mathbf{v}^{n}$.

### Relativistic Boris Algorithm

Using $\mathbf{u} = \gamma \mathbf{v}$, the relativistic velocity update equation is:
\begin{equation}
\frac{\mathbf{u}^{n+1} - \mathbf{u}^{n}}{\Delta t} = \frac{q}{m}\left(\mathbf{E}^{n+\theta} + \bar{\mathbf{u}} \times \frac{\mathbf{B}}{\gamma^{n+1/2}}\right)
\end{equation}

1.  Convert $\mathbf{v}$ to $\mathbf{u}$.
2.  **Acceleration:**
    \begin{equation}
    \mathbf{u}^- = \mathbf{u}^{n} + \frac{q\Delta t}{2m}\mathbf{E}^{n+\theta}
    \end{equation}

3.  Calculate $\bar{\mathbf{u}}$ using:
    \begin{equation}
    \bar{\mathbf{u}} = \frac{\mathbf{u}^- + \mathbf{u}^- \times \mathbf{t} + (\mathbf{u}^- \cdot \mathbf{t})\mathbf{t}}{1+t^2}
    \end{equation}
    where $\mathbf{t} = \frac{q\Delta t}{2m} \frac{\mathbf{B}}{\gamma^{n+1/2}}$. From the definitions of $\gamma$ and $\mathbf{u} = \gamma\mathbf{v}$, we can obtain $\gamma^{n+1/2} = \sqrt{1 + (u^-/c)^2}$.

4.  The final velocity is then $\mathbf{u}^{n+1} = 2\bar{\mathbf{u}} - \mathbf{u}^{n}$.
5.  Convert $\mathbf{u}$ back to $\mathbf{v}$.

## Miscellaneous

### Calculating the Total Pressure Tensor from Subgroups

The pressure tensor for each subgroup is:
\begin{equation}
p^s_{m,n} = \frac{1}{V} \sum_{i=1} w_i^s (v_{i,m}^s - \bar{v}_m^s) (v_{i,n}^s - \bar{v}_n^s)
\end{equation}
where $V$ is the volume, and $\bar{v}_m^s$ and $\bar{v}_n^s$ are the average velocities of subgroup $s$.

The total pressure tensor is:
\begin{equation}
p_{m,n} = \frac{1}{V} \sum_{s}\sum_{i=1} w_i^s (v_{i,m}^s - \bar{v}_m) (v_{i,n}^s - \bar{v}_n)
\end{equation}
where $\bar{v}_m$ and $\bar{v}_n$ are the average velocities of all particles, and $\rho_s$ is the density of subgroup $s$:
\begin{equation}
\bar{v}_m = \frac{\sum_{s}\sum_{i=1} w_i^s v_{i,m}^s}{\sum_{s}\sum_{i=1} w_i^s} = \frac{\sum_s \rho_s \bar{v}_m^s}{\sum_{s}\rho_s}
\end{equation}
\begin{equation}
\bar{v}_n = \frac{\sum_{s}\sum_{i=1} w_i^s v_{i,n}^s}{\sum_{s}\sum_{i=1} w_i^s} = \frac{\sum_s \rho_s \bar{v}_n^s}{\sum_{s}\rho_s}
\end{equation}

The total pressure tensor can be expanded as:
\begin{align*}
p_{m,n} &= \frac{1}{V} \sum_{s}\sum_{i=1} w_i^s \left[(v_{i,m}^s - \bar{v}_m^s) + (\bar{v}_m^s - \bar{v}_m)\right] \left[(v_{i,n}^s - \bar{v}_n^s) + (\bar{v}_n^s - \bar{v}_n)\right] \\
&= \frac{1}{V} \sum_{s}\sum_{i=1} w_i^s (v_{i,m}^s - \bar{v}_m^s)(v_{i,n}^s - \bar{v}_n^s) \\
&\quad + \frac{1}{V} \sum_{s}\sum_{i=1} w_i^s (v_{i,m}^s - \bar{v}_m^s)(\bar{v}_n^s - \bar{v}_n) \\
&\quad + \frac{1}{V} \sum_{s}\sum_{i=1} w_i^s (\bar{v}_m^s - \bar{v}_m)(v_{i,n}^s - \bar{v}_n^s) \\
&\quad + \frac{1}{V} \sum_{s}\sum_{i=1} w_i^s (\bar{v}_m^s - \bar{v}_m)(\bar{v}_n^s - \bar{v}_n)
\end{align*}
The second and third terms are zero. The first term is the sum of the pressure tensors of the subgroups. The fourth term is $\sum_s \rho_s (\bar{v}_m^s - \bar{v}_m)(\bar{v}_n^s - \bar{v}_n)$. Thus, the final expression is:
\begin{equation}
p_{m,n} = \sum_s p^s_{m,n} + \sum_s \rho_s (\bar{v}_m^s - \bar{v}_m)(\bar{v}_n^s - \bar{v}_n)
\end{equation}

### Noise Reduction

> One man's noise is another man's signal [@birdsall2018plasma]. Without noise, there would be no particle simulation; without proper treatment of noise, there would still be no particle simulation.

In Particle-in-Cell simulations, a key challenge is managing statistical noise. Compared to deterministic continuum kinetic schemes, PIC intrinsically possesses numerical errors associated with particle noise, which decreases slowly as one increases the number of particles. Specifically, the noise in PIC schemes decreases as $1/\sqrt{P_c}$ where $P_c = N_p/N_c$ is the number of particles per cell [@birdsall2018plasma].

- Representing the Plasma: A PIC simulation doesn't track every single particle in the real plasma. Instead, it uses "super-particles" to represent a large group of real particles. This introduces statistical fluctuations since a finite number of super-particles can't perfectly reflect the smooth, continuous behavior of a real plasma.

- Field Calculations: Particle positions and velocities are used to calculate the electric and magnetic fields in the simulation. Statistical fluctuations in the particle distribution can lead to noisy fields, which in turn, feed back into the particle motion, potentially amplifying the noise further.

FLEKS employs several techniques to mitigate noise:

- Higher-Order Particle Weighting: Using smoother mathematical functions (splines) to distribute the charge and current of super-particles onto the simulation grid. This reduces the abrupt changes that lead to noise.

- Particle Splitting and Merging: This technique helps control the number of super-particles in different simulation regions, leading to a more consistent and accurate noise level. Let's break down how this works:
  - Particle Splitting
    - When it happens: In simulation regions with high particle density, super-particles become very "heavy" (representing many real particles). This can make them computationally unwieldy and introduce local noise.
    - How it works: FLEKS splits heavy super-particles into several smaller super-particles, each representing a smaller number of real particles. This reduces local noise and improves computational load balancing.
  - Particle Merging
    - When it happens: In regions of low particle density, too few super-particles can also be a source of noise. This makes it hard to accurately calculate local fields.
    - How it works: FLEKS merges multiple low-weight super-particles into a single heavier one. This smooths out the particle distribution and promotes more accurate field calculations.

Key Point: Particle splitting and merging in FLEKS are not just about noise reduction. They also play a crucial role in achieving proper load balancing across many computational cores, essential for simulations to run efficiently.