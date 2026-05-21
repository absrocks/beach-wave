# Beach Wave - Turbulent Statistics Data

Turbulent statistics extracted from OpenFOAM simulations of wave-driven hydrodynamics over different beach profile configurations via ParaView. The data enables comparison of turbulence characteristics across beach nourishment strategies and erosion scenarios.

## Repository Structure

```
turbulent_staticstics_data/
├── initial_profile/        # Reference (equilibrium) beach profile
├── eroded_profile/         # Beach profile after erosion
├── bar_nourishment/        # Bar-type beach nourishment
├── berm_nourishment/       # Berm-type beach nourishment
└── profile_nourishment/    # Full profile nourishment
```

Each directory contains **96 data files** covering cross-shore positions from **x = 18.5 m** to **x = 28.0 m** at 0.1 m intervals.

## Data Format

Files are named `TKE_epsilon_turb_epsilon_<X>.dat` where `<X>` is the cross-shore coordinate of the vertical profile.

Each file is tab-separated with a header line:

```
# X (m)  TKE_avg  epsilon_turb_avg  epsilon_avg
 18.05   9.42910932e-05  8.08574217e-06  1.72762275e-06
 18.10   3.51141883e-05  1.17610031e-06  2.51906888e-06
 ...
```

| Column             | Description                                    | Unit    |
|--------------------|------------------------------------------------|---------|
| `X (m)`            | Cross-shore position                           | m       |
| `TKE_avg`          | Span-averaged and depth-averaged turbulent Kinetic Energy         | m²/s²   |
| `epsilon_turb_avg` | Span-averaged and depth-averaged turbulent dissipation rate        | m²/s³   |
| `epsilon_avg`      | Span-averaged and depth-averaged total dissipation rate            | m²/s³   |

Each file contains ~113 data points spanning a cross-shore range of approximately 18.05 m to 28.0 m at 0.05 m resolution.

$$\mathrm{TKE} = k_{\mathrm{resolved}} + k_{\mathrm{sgs}}$$

$$k_{\mathrm{resolved}} = \tfrac{1}{2}\left(\langle u'^2 \rangle + \langle v'^2 \rangle + \langle w'^2 \rangle\right)$$

where

$$u' = u - \langle u \rangle$$

and $\langle u \rangle$ is the velocity averaged along the spanwise direction.




## Scenarios

| Scenario                | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **Initial profile**     | Baseline equilibrium beach profile before any intervention or erosion       |
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
