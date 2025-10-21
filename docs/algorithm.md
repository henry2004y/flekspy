# Algorithm

## Unit Conversion

A physical quantity consists of two parts: a value and a unit. For example, the weight of a table can be expressed as $w = 50 \text{ kg} = 50,000 \text{ g}$, where 50 and 50,000 are the values, and kg and g are the units. If we define a new unit, wg, such that $1 \text{ wg} = 2.5 \text{ kg}$, the table's weight becomes $w = 20 \text{ wg}$. In general, a physical quantity $u$ can be expressed as $u = \bar{u} u^*$, where $\bar{u}$ is its numerical value and $u^*$ is its unit.

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

Reducing the speed of light changes the propagation speed of electromagnetic waves in the simulation, but it does not change the interaction between particles and the magnetic field. In the mass normalization, we require $\bar{q}/(\bar{m}\bar{c}) = 1$, not $\bar{q}/\bar{m} = 1$, because we do not assume $\bar{c}=1$. The speed of light in the first equation is always the physical speed of light, regardless of the choice of $v_{cgs}^*$. For example, if we set $v_{cgs}^* = 0.1c$, then $\bar{c}$ must be 10 to satisfy $\bar{q}/(\bar{m}\bar{c}) = 1$.

Since reducing $v_{cgs}^*$ does not affect the particle-magnetic field interaction, the particle's gyromotion (both gyro-radius and gyro-frequency) remains unchanged. The inertial length, which is the gyro-radius of a particle moving at the Alfven velocity, is also unaffected because neither the gyromotion nor the Alfven velocity changes with $v_{cgs}^*$.

However, what about the interaction between particles and the electric field? Since $\bar{c} = \bar{q}/\bar{m}$ is not necessarily 1, the Coulomb force on a proton should be $(\bar{q}/\bar{m})E = \bar{c}E$. In the code, $\bar{c}$ is ignored, suggesting that the simulated Coulomb force is $\bar{c}$ times weaker than in reality. Is this reasonable? What are its consequences? This needs to be clarified.

Most PIC simulations use a reduced speed of light. While their results may be interpreted differently than in FLEKS/MHD-EPIC, I believe they also alter the ratio between the Coulomb force and the $\mathbf{v} \times \mathbf{B}$ force. Therefore, our approach of reducing the speed of light is likely reasonable.

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
