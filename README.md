# ICCP – Intensive Care Cerebral Perfusion Model

## Overview

The ICCP is a lumped resistance–compliance (RC) model developed to simulate global cerebral blood flow (CBF) and middle cerebral artery (MCA) flow velocity from continuous bedside available ICU signals such as arterial blood pressure (ABP) and arterial carbon dioxide tension (PaCO₂) signals.

The model represents cerebral circulation through parallel arterial branches, a microvascular compartment, and a venous compartment. Cerebral autoregulation is implemented through four regulatory mechanisms:

- Myogenic mechanism
- CO₂ reactivity
- Metabolic mechanism
- Endothelial (shear-stress-mediated) mechanism

The model was developed as part of a master's thesis investigating cerebral hemodynamics and autoregulation in patients receiving venoarterial extracorporeal membrane oxygenation (VA ECMO).

## Requirements

The model was developed in Python 3.14 and requires the following packages:

- NumPy
- SciPy
- Matplotlib
- Pandas

Install the dependencies using:

`pip install -r requirements.txt`

The package versions used during development are specified in `requirements.txt`.

## Model Inputs

The model accepts the following time-dependent inputs:

| Input | Description | Unit |
|---|---|---|
| ABP | Arterial blood pressure | mmHg |
| PaCO₂ | Arterial carbon dioxide tension | mmHg |

Additional physiological parameters, including intracranial pressure (ICP), central venous pressure (CVP), and arterial oxygen tension (PaO₂), are defined in the model and can be modified.

## Model Outputs

The model calculates several hemodynamic variables, including:

- Global cerebral blood flow (CBF)
- MCA flow and estimated MCA flow velocity
- Compartmental pressures and volumes
- Microvascular resistance
- Myogenic, CO₂, metabolic, and endothelial activation signals

## Model Implementation

The cerebral circulation is modeled using a system of ordinary differential equations (ODEs), numerically integrated using the LSODA solver through SciPy's `solve_ivp` function.

The default model parameters represent a healthy reference subject. Patient-specific ABP and PaCO₂ signals can be supplied as time-dependent inputs.

## Example Usage 

from ICCP_model import HealthySubject, ParallelSplitCerebralRC

# Initialize reference subject and model
subject = HealthySubject()
model = ParallelSplitCerebralRC(subject)

# Simulate 900 seconds
sol = model.simulate(t_span=(0, 900), dt=0.2)

# Extract simulated global cerebral blood flow
CBF = [
    model.compute_algebraic(t, sol.y[:, i])["Qmicro_out"]
    for i, t in enumerate(sol.t)
]

print(f"Mean CBF: {sum(CBF)/len(CBF):.2f} mL/s")

## Limitations

This model is intended for academic research and physiological simulation. It is a simplified lumped representation of cerebral circulation and does not resolve patient-specific cerebrovascular anatomy or spatial blood-flow distributions.

It is not intended for clinical decision-making.

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.

