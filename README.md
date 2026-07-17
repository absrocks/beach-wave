# Beach Wave - Turbulent Statistics Data

Turbulent statistics extracted from OpenFOAM simulations of wave-driven hydrodynamics over different beach profile configurations via ParaView. The data enables comparison of turbulence characteristics across beach nourishment strategies and erosion scenarios.

## Repository Structure

```
full_scale/turbulent_staticstics_data/
├── eroded_profile/                   # Beach profile after erosion
├── bar_nourishment_full_initial/     # Bar-type beach nourishment intital Profile
├── bar_nourishment_full_final/       # Bar-type beach nourishment final Profile
├── berm_nourishment_full_initial/    # Berm-type beach nourishment intital Profile
├── berm_nourishment_full_final/      # Berm-type beach nourishment final Profile
└── profile_nourishment_initial/      # Profile nourishment intital Profile
└── profile_nourishment_final/        # Profile nourishment final Profile
```

Each directory contains **96 data files** covering cross-shore positions from **x = 33 m** to **x = 48.0 m** at 0.03 m intervals.

## Data Format

Files are named `TKE_epsilon_turb_pw_avg_Y_epsilon_pw_avg_Y<X>.dat` where `<X>` is the cross-shore coordinate of the vertical profile.

Each file is tab-separated with a header line:

```
# X (m) TKE_avg epsilon_turb_pw_avg_Y_avg epsilon_pw_avg_Y_avg
 33.0303  8.58764246e-05  0.547719438  0.0725584536
 ...
```

| Column             | Description                                    | Unit    |
|--------------------|------------------------------------------------|---------|
| `X (m)`            | Cross-shore position                           | m       |
| `TKE_avg`          | Span-averaged and depth-averaged turbulent Kinetic Energy         | m²/s²   |
| `epsilon_turb_pw_avg_Y_avg` | Span-averaged and depth-averaged turbulent dissipation rate        | m²/s³   |
| `epsilon_pw_avg_Y_avg`      | Span-averaged and depth-averaged total dissipation rate            | m²/s³   |

Each file contains ~113 data points spanning a cross-shore range of approximately 33 m to 48 m at 0.03 m resolution.

### Turbulent Kinetic Energy
$$\mathrm{TKE} = k_{\mathrm{resolved}} + k_{\mathrm{sgs}}$$

$$k_{\mathrm{resolved}} = \tfrac{1}{2}\left(\langle u'^2 \rangle + \langle v'^2 \rangle + \langle w'^2 \rangle\right)$$

where

$$u' = u - \langle u \rangle$$

and $\langle u \rangle$ is the velocity averaged along the spanwise direction.

### Turbulent Dissipation Rate

$$\varepsilon = \varepsilon_{\mathrm{resolved}} + \varepsilon_{\mathrm{sgs}} = 2\nu\ \langle S'_{ij} S'_{ij} \rangle + 2\langle \nu_{\mathrm{sgs}} S'_{ij} S'_{ij} \rangle$$

where the fluctuating strain-rate tensor is

$$S'_{ij} = \tfrac{1}{2}\left(\frac{\partial u'_i}{\partial x_j} + \frac{\partial u'_j}{\partial x_i}\right)$$

with $\nu$ the kinematic viscosity and $\nu_{\mathrm{sgs}}$ the subgrid-scale eddy viscosity.


## Scenarios

| Scenario                | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **Eroded profile**      | Beach profile degraded by wave-driven erosion processes                     |
| **Bar nourishment**     | Sediment placed as a submerged bar offshore to dissipate wave energy        |
| **Berm nourishment**    | Sediment placed on the upper beach face (berm) for direct shore protection  |
| **Profile nourishment** | Sediment distributed across the full beach profile                          |

## Usage

The data can be loaded with any tool that reads whitespace-delimited text. For example, in Python:

```python
import numpy as np

data = np.loadtxt(
    "turbulent_staticstics_data/initial_profile/TKE_epsilon_turb_epsilon_20.0.dat",
    skiprows=1,
)
x = data[:, 0]
tke = data[:, 1]
eps_turb = data[:, 2]
eps_total = data[:, 3]
```

Alternatively, use `python plot_data.py`.

## License

Please contact the repository owner for licensing and citation information.
