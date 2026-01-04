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

- **Semi-Implicit PIC**[^common]: Semi-implicit methods aim for a hybrid approach. They partially _linearize_ the coupling between particles and fields to improve stability somewhat. This involves using field values from the current and potentially from previous time steps, to calculate particle motion and field updates. Several techniques exist for this linearization (implicit moment, direct implicit etc.).
  - Particle Push: Similar to explicit PIC, the method first uses the electric fields from the current time step (or potentially from a combination of current and previous time steps) to calculate the force acting on each particle.
  - Linearized Update:  Here's where the 'implicitness' comes in.  The semi-implicit method employs a specific technique to linearize the relationship between how particles move and how the fields update.  Some common linearization techniques include:
    1. Implicit Moment Method: This method solves a set of moment equations for the particles' distribution function, which is then used to update the fields in a linearized manner.
    2. Direct Implicit Method: This method performs a formal linear expansion of the coupling operator between particles and fields, allowing for a linearized update of the fields.
  - Field Update: Once the coupling between particles and fields is linearized, the method updates the electric fields on the grid using the information from the particle push step implicitly.
However, the semi-implicit methods prosposed in the 1980s (Implicit Moment Method, Direct Implicit Method) lack energy conservation, which causes instability issues with grid resolution much larger than the Debye length.

| Feature | Fully Implicit PIC | Semi-Implicit PIC |
|---|---|---|
| Stability | Highly stable, larger time steps allowed | More stable than explicit, but not as unconditionally stable as fully implicit |
| Accuracy | Can be very accurate | Accuracy can depend on the specific linearization technique, some energy conservation issues may exist |
| Computational Cost | Highest, due to matrix inversions at each step | More expensive than explicit, but often less so than fully implicit |

[^common]: One common misconception is that in a semi-implicit method, the particle pusher is explicit while the field solver is implicit. This is not exactly true, because we cannot simply separate the two components.

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

In the non-relativistic limit, there is no speed of light. In the pusher, we use c as the velocity normalization, but we can choose it freely.

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

In the code, the condition $\bar{q}/(\bar{m}\bar{c}) = 1$ is required for a proton. This means the normalization units $q_{cgs}^*$ and $m_{cgs}^*$ must satisfy:
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

Reducing the speed of light changes the propagation speed of electromagnetic waves in the simulation, but it does not change the interaction between particles and the magnetic field. In PIC codes, this is often called the Darwin, magneto-inductive, or "electrostatic-less" approximation, depending on the exact formulation. In the mass normalization, we require $\bar{q}/(\bar{m}\bar{c}) = 1$, not $\bar{q}/\bar{m} = 1$, because we do not assume $\bar{c}=1$. The speed of light in the first equation is always the physical speed of light, regardless of the choice of $v_{cgs}^*$. For example, if we set $v_{cgs}^* = 0.1c$, then $\bar{c}$ must be 10 to satisfy $\bar{q}/(\bar{m}\bar{c}) = 1$.

The primary goal of a reduced-$c$ model is to overcome the two most restrictive time-step constraints in an explicit EM-PIC code (although FLEKS is semi-implicit, the particle pusher is still explicit):

1. The Courant-Friedrichs-Lewy (CFL) Condition: $\Delta t < \Delta x / c$. This is required to resolve light waves.
2. The Plasma Frequency Constraint: $\Delta t \lesssim 0.1 / \omega_{pe}$. This is required to resolve electron plasma oscillations (Langmuir waves).

By reducing $c$ to $v_{cgs}^*$, the code's field solver relaxes the CFL condition to $\Delta t < \Delta x / v_{cgs}^*$. This is a huge win. However, this doesn't solve the $\omega_{pe}$ constraint. The electric force $q\mathbf{E}$ (specifically the electrostatic part, $\mathbf{E}_{es}$) is the restoring force for Langmuir waves. By implementing $\frac{d\bar{v}}{d\bar{t}} = \bar{\mathbf{E}}$ instead of $\frac{d\bar{v}}{d\bar{t}} = \bar{c}\bar{\mathbf{E}}$, we are artificially weakening the particle's response to the electric field.
This modification effectively disables high-frequency electrostatic physics. The restoring force for Langmuir waves is made $\bar{c}$ times weaker, which dramatically lowers the plasma frequency $\omega_{pe}$ (or, more accurately, damps the waves entirely).

Most PIC simulations use a reduced speed of light. While their results may be interpreted differently than in FLEKS, they also alter the ratio between the electric force and the $\mathbf{v} \times \mathbf{B}$ force. This reduced-c model is therefore designed to simulate phenomena where:

- $v \ll c$ (non-relativistic).
- Frequencies are low: $\omega \ll \omega_{pe}$.
- Physics is dominated by magnetic and inductive effects (gyromotion, $\mathbf{v} \times \mathbf{B}$ forces, $\nabla \times \mathbf{E} = -\partial \mathbf{B}/\partial t$).
- Physics is not dominated by electrostatic effects (Langmuir waves, Debye shielding, double layers).

## Semi-Implicit Methodology

FLEKS uses the Gauss-Law satisfying energy conserving semi-implicit method (GL-ECSIM). Its semi-implicit nature allows the PIC model to run on a coarser grid with a larger time step than explicit PIC methods, which are limited by the stability conditions to $\Delta t < \Delta x/c$ and the grid resolution $\Delta x < \zeta \lambda_D$, where c is the speed of light, $\lambda_D$ is the Debye length, and $\zeta$ is of order one and depends on the numerical details of the method. GL-ECSIM facilitates the coupling between the MHD and the PIC models, and its energy conserving property helps eliminating numerical instabilities and spurious waves that would violate the energy conservation.

ECSIM uses a staggered grid, where the electric field is defined at cell nodes, and the magnetic field is defined at cell centers. The Maxwell's equations are solved implicitly:
\begin{align}
\frac{\mathbf{B}^{n+1} - \mathbf{B}^n}{\Delta t} &= - c\nabla\times\mathbf{E}^{n+\theta} \\
\frac{\mathbf{E}^{n+1} - \mathbf{E}^n}{\Delta t} &= - c\nabla\times\mathbf{B}^{n+\theta} - 4\pi\bar{\mathbf{J}}
\end{align}
where $\theta \in [0.5, 1]$ is the time centering parameter. $\bar{\mathbf{J}}$ is the predicted current density at $n + \frac{1}{2}$ time stage, and it can be expressed as a linear function of the unknown electric field $\mathbf{E}^{n+\theta}$.
The variables at time stage $n+\theta$ can be written as linear combinations of values at the time steps n and n + 1:
\begin{align}
\mathbf{E}^{n+\theta} &= (1-\theta)\mathbf{E}^n + \theta \mathbf{E}^{n+1} \\
\mathbf{B}^{n+\theta} &= (1-\theta)\mathbf{B}^n + \theta \mathbf{B}^{n+1}
\end{align}

After rearranging the equations above and using the identity $\nabla\times\nabla\times\mathbf{E}= \nabla(\nabla\cdot\mathbf{E}) − \nabla^2\mathbf{E}$, we come up with an equation of $\mathbf{E}^{n+\theta}$
\begin{equation}
\mathbf{E}^{n+\theta} + \delta^2\left[ \nabla(\nabla\cdot\mathbf{E}^{n+\theta}) − \nabla^2\mathbf{E}^{n+\theta} \right] = \mathbf{E}^n + \delta\left( \nabla\times\mathbf{B}^n - \frac{4\pi}{c}\bar{\mathbf{J}} \right)
\end{equation}
where $\delta = c\theta \Delta t$. After applying finite difference discretizations to the gradient and divergence operators, we obtain a linear system of equations for the discrete values of $\mathbf{E}^{n+\theta}$ at the cell nodes. The iterative generalized minimal residual method (GMRES) is used to solve the equations to obtain $\mathbf{E}^{n+\theta}$. Using equations (1) and (3), the magnetic field $\mathbf{B}^{n+1}$ and electric field $\mathbf{E}^{n+1}$ at the next time step can be obtained, respectively.

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

[^v-avg]: This is important for having an explicit particle pusher.

Substituting then @eq-vavg-form2 into @eq-Jbar, we obtain without any approximation or linearization:
\begin{equation}
\bar{\mathbf{J}}_{sg} = \frac{1}{V_g}\sum_p q_p \hat{\mathbf{v}}_p W_{pg} + \frac{\beta_s}{V_g}\sum_p q_p \hat{\mathbf{E}}_p^{n+\theta} W_{pg}
\end{equation}
where the summation is intended over all particles of species s.

Let
\begin{equation}
\hat{\mathbf{J}}_{sg} = \sum_p q_p \hat{\mathbf{v}}_p W_{pg}
\end{equation}
be the current based on hatted velocity, we have, together with @eq-hat-v-E,
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

@eq-mass-matrix defines the elements of the mass matrices that are the most peculiar characteristic of the method proposed here. There are 3v such matrices, where v is the dimensionality of the magnetic field and velocity vector, not to be confused with the dimensionality of the geometry used for space d. The indices i and j in @eq-mass-matrix vary in the 3v-space. For example for full 3-components vectors, and there are 9 mass matrices. Each matrix is symmetric and very sparse with just 2d diagonals.

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