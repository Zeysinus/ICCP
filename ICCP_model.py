import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from copy import deepcopy
from scipy.signal import find_peaks


# ============================================================
# HEALTHY SUBJECT PARAMETERS
# ============================================================

class HealthySubject:
    def __init__(self):

        # ============================================================s
        # BASIC PHYSIOLOGY
        # ============================================================
        self.brain_mass = 1400.0          # g
        self.CMRO2 = 3.5                  # mL O2 / 100 g / min

        # ============================================================
        # PRESSURES [mmHg]
        # ============================================================
        self.P_ICP = 10
        self.P_CVP = 6 
        self.P_star = 0.0

        self.Pin0_target = 80.0           # systemic arterial input pressure
        self.Part0_target = 62.00        # common downstream arterial pressure (intravascular pressure)
        self.Pmicro0_target = 45.0
        self.Pv0_target = 14.0

        self.Pout0 = max(self.P_CVP, self.P_ICP + self.P_star)

        # ============================================================
        # BASELINE CBF
        # ============================================================
        self.cbf_base = 52.5              # mL/min/100g

        self.Q0 = (self.cbf_base / 60.0) * (self.brain_mass / 100.0)
        # Q0 ≈ 12.25 mL/s

        # ============================================================
        # PARALLEL SPLIT FRACTIONS
        # ============================================================
        # These describe how the inflow is divided at baseline.
        self.frac_MCA = 1.0 / 3.0
        self.frac_W = 2.0 / 3.0

        # ============================================================
        # ARTERIAL VOLUMES [mL]
        # ============================================================
        # Total arterial volume is split into MCA and Willis/other branch.
        self.Vart0_total = 10.0

        self.VMCA0_target = self.frac_MCA * self.Vart0_total
        self.VW0_target = self.frac_W * self.Vart0_total

        self.Vmicro0_target = 20.0
        self.Vv0_target = 70.0

        # ============================================================
        # COMPLIANCES [mL/mmHg]
        # ============================================================
        # Branch compliances are separate.
        self.C_MCA = (self.VMCA0_target / (self.Part0_target - self.P_ICP))
        self.C_W = (self.VW0_target / (self.Part0_target - self.P_ICP))

        self.C_art_total = self.C_MCA + self.C_W

        self.C_micro = self.Vmicro0_target / (self.Pmicro0_target - self.P_ICP)
        self.C_v = self.Vv0_target / (self.Pv0_target - self.P_ICP)

        # ============================================================
        # PARALLEL BRANCH RESISTANCES [mmHg*s/mL]
        # ============================================================
        # Both branches run from Pin to their own compliant branch.
        dP_art = self.Pin0_target - self.Part0_target

        self.R_MCA = (dP_art / (self.frac_MCA * self.Q0))*1.6
        self.R_W = (dP_art / (self.frac_W * self.Q0))*1.6
        self.R_art_total = 1.0 / (1.0 / self.R_MCA + 1.0 / self.R_W)

        # ============================================================
        # MICROVASCULAR RESISTANCE
        # ============================================================
        self.R_up0 = (self.Part0_target - self.Pmicro0_target) / self.Q0
        self.R_down0 = (self.Pmicro0_target - self.Pv0_target) / self.Q0

        self.R_micro0 = self.R_up0 + self.R_down0

        self.R_up_frac = self.R_up0 / self.R_micro0
        self.R_down_frac = self.R_down0 / self.R_micro0

        # ============================================================
        # VENOUS OUTFLOW RESISTANCE
        # ============================================================
        self.R_out = (self.Pv0_target - self.Pout0) / self.Q0*1.5

        # ============================================================
        # AUTOREGULATION PARAMETERS
        # ============================================================
        self.tau_myogenic = 2.0
        self.tau_CO2 = 20.0
        self.tau_metabolic = 10.0
        self.tau_shear_endo = 60.0

        # Strengths of mechanisms
        self.R_span_myo = 1.0
        self.R_span_CO2 = 0.9
        self.R_span_met = 0.45
        self.R_span_endo = 0.3

        # Myogenic setpoint based on transmural arterial pressure
        self.P_trans_set = (self.Part0_target) - self.P_ICP

        # Resistance reserve
        self.k_dilate_min = 0.25
        self.k_constrict_max = 4.0

        self.R_micro_min = self.R_micro0 * self.k_dilate_min
        self.R_micro_max = self.R_micro0 * self.k_constrict_max

        # ============================================================
        # BLOOD GAS PARAMETERS
        # ============================================================
        self.PaCO2_ref = 40.0
        self.PaO2_ref = 100.0

        self.PaCO2_func = lambda t: self.PaCO2_ref
        self.PaO2_func = lambda t: self.PaO2_ref

        self.Hb_conc = 15.0
        self.SaO2 = 0.98
        self.HCO3_a = 24.0
        self.RQ_func = 1.0

        CaCO2_ref = (
            0.03 * self.PaCO2_ref
            + self.HCO3_a
        ) / 100.0

        VO2 = (
            self.CMRO2 / 60.0
        ) * (
            self.brain_mass / 100.0
        )

        VCO2 = self.RQ_func * VO2

        self.CvCO2_ref = (
            CaCO2_ref
            + VCO2 / self.Q0
        )
        CaO2_ref = (
            1.34 * self.Hb_conc * self.SaO2
            + 0.0031 * self.PaO2_ref
        ) / 100.0

        VO2 = (
            self.CMRO2 / 60.0
        ) * (
            self.brain_mass / 100.0
        )

        self.CvO2_ref = CaO2_ref - VO2 / self.Q0

        # Endothelial / shear parameter
        self.radius_MCA = 1.5             # mm

        # External pressure functions
        self.ICP_func = lambda t: self.P_ICP
        self.CVP_func = lambda t: self.P_CVP
        r = self.radius_MCA * 1e-3
        eta = 0.0035
        Q_MCA0 = self.frac_MCA * self.Q0
        self.shear_ref0 = (4 * eta * Q_MCA0 * 1e-6) / (np.pi * r**3)


# ============================================================
# PARALLEL SPLIT CEREBRAL RC MODEL
# ============================================================

class ParallelSplitCerebralRC:
    def __init__(self, subject, pa_func=None, paco2_func=None):
        self.__dict__.update(subject.__dict__)

        self.pa_func = pa_func if pa_func is not None else (
            lambda t: self.Pin0_target
        )
        # change signature
        self.PaCO2_func = paco2_func if paco2_func is not None else getattr(subject, "PaCO2_func", lambda t: self.PaCO2_ref)
        self.ICP_func = getattr(subject, "ICP_func", lambda t: self.P_ICP)
        self.CVP_func = getattr(subject, "CVP_func", lambda t: self.P_CVP)
        #self.PaCO2_func = getattr(subject, "PaCO2_func", lambda t: self.PaCO2_ref)
        self.PaO2_func = getattr(subject, "PaO2_func", lambda t: self.PaO2_ref)
        

    def initial_state(self):

        VMCA0 = self.VMCA0_target
        VW0 = self.VW0_target
        Vmicro0 = self.Vmicro0_target
        Vv0 = self.Vv0_target

        O2v0 = self.CvO2_ref * Vmicro0
        CO2v0 = self.CvCO2_ref * Vmicro0

        A_myo0 = 0.0
        A_CO20 = 0.0
        A_met0 = 0.0
        A_endo0 = 0.0

        return [
            VMCA0,
            VW0,
            Vmicro0,
            Vv0,
            O2v0,
            CO2v0,
            A_myo0,
            A_CO20,
            A_met0,
            A_endo0
        ]


    def simulate(self, t_span=(0, 900), dt=0.2, x0=None): 

        if x0 is None:
            x0 = self.initial_state()

        t0, t1 = t_span

        # t_eval: never goes beyond t_span[1]
        t_eval = np.arange(t0, t1, dt)

        # Add exact final time if it is not already included
        if t_eval[-1] < t1:
            t_eval = np.append(t_eval, t1)

        sol = solve_ivp(
            self.rhs,
            (t0, t1),
            x0,
            t_eval=t_eval,
            method="LSODA",
            rtol=1e-6,
            atol=1e-8,
        )

        return sol

    def compute_algebraic(self, t, x):

        (
            VMCA,
            VW,
            Vmicro,
            Vv,
            O2v_content,
            CO2v_content,
            A_myo,
            A_CO2,
            A_met,
            A_endo,
        ) = x

        # ============================================================
        # EXTERNAL PRESSURES
        # ============================================================
        ICP = self.ICP_func(t)
        CVP = self.CVP_func(t)
        Pin = self.pa_func(t)

        Pout = max(CVP, ICP + self.P_star)

        # ============================================================
        # BRANCH PRESSURES
        # ============================================================
        P_MCA = ICP + VMCA / self.C_MCA
        P_W = ICP + VW / self.C_W

        # ============================================================
        # COMMON DOWNSTREAM ARTERIAL PRESSURE
        # ============================================================
        # Total arterial volume over total compliance.
        P_arterial = ICP + (VMCA + VW) / (self.C_MCA + self.C_W)

        # ============================================================
        # MICROVASCULAR AND VENOUS PRESSURES
        # ============================================================
        Pmicro = ICP + Vmicro / self.C_micro
        Pv = ICP + Vv / self.C_v

        # ============================================================
        # PARALLEL BRANCH FLOWS
        # ============================================================
        Q_MCA = (Pin - P_MCA) / self.R_MCA
        Q_W = (Pin - P_W) / self.R_W

        Qa_total = Q_MCA + Q_W

        # ============================================================
        # AUTOREGULATED MICROVASCULAR RESISTANCE
        # ============================================================
        R_CO2_base = self.R_micro0 * (1.0 - self.R_span_CO2 * A_CO2)
        R_CO2_base = np.clip(R_CO2_base, self.R_micro_min, self.R_micro_max)

        R_micro = R_CO2_base * (
            1.0
            + self.R_span_myo * A_myo
            - self.R_span_met * A_met
            - self.R_span_endo * A_endo
        )

        R_micro = np.clip(R_micro, self.R_micro_min, self.R_micro_max)

        R_up = max(self.R_up_frac * R_micro, 1e-9)
        R_down = max(self.R_down_frac * R_micro, 1e-9)

        # ============================================================
        # MICROVASCULAR AND VENOUS FLOWS
        # ============================================================
        Qmicro_in = (P_arterial - Pmicro) / R_up
        Qmicro_out = (Pmicro - Pv) / R_down
        Qout = (Pv - Pout) / self.R_out

        # ============================================================
        # DISTRIBUTION OF MICROVASCULAR INFLOW DEMAND
        # ============================================================
        # Qmicro_in leaves the combined arterial system.
        # We divide this outflow between MCA and W branches according to
        # their conductances. This keeps volume conservation stable.
        G_MCA = 1.0 / self.R_MCA
        G_W = 1.0 / self.R_W

        alpha_MCA = G_MCA / (G_MCA + G_W)
        alpha_W = G_W / (G_MCA + G_W)

        Q_to_micro_from_MCA = alpha_MCA * Qmicro_in
        Q_to_micro_from_W = alpha_W * Qmicro_in

        return {
            "ICP": ICP,
            "CVP": CVP,
            "Pin": Pin,
            "Pout": Pout,

            "P_MCA": P_MCA,
            "P_W": P_W,
            "P_arterial": P_arterial,
            "Pmicro": Pmicro,
            "Pv": Pv,

            "Q_MCA": Q_MCA,
            "Q_W": Q_W,
            "Qa_total": Qa_total,

            "Qmicro_in": Qmicro_in,
            "Qmicro_out": Qmicro_out,
            "Qout": Qout,

            "Q_to_micro_from_MCA": Q_to_micro_from_MCA,
            "Q_to_micro_from_W": Q_to_micro_from_W,

            "R_micro": R_micro,
            "R_up": R_up,
            "R_down": R_down,


            "R_art_total": self.R_art_total,

            "alpha_MCA": alpha_MCA,
            "alpha_W": alpha_W,
        }

    def rhs(self, t, x):

        (
            VMCA,
            VW,
            Vmicro,
            Vv,
            O2v_content,
            CO2v_content,
            A_myo,
            A_CO2,
            A_met,
            A_endo,
        ) = x

        alg = self.compute_algebraic(t, x)

        # ============================================================
        # VOLUME BALANCES
        # ============================================================
        dVMCA = alg["Q_MCA"] - alg["Q_to_micro_from_MCA"]
        dVW = alg["Q_W"] - alg["Q_to_micro_from_W"]

        dVmicro = alg["Qmicro_in"] - alg["Qmicro_out"]
        dVv = alg["Qmicro_out"] - alg["Qout"]
        P_trans = alg["P_arterial"] - alg["ICP"]

        # ============================================================
        # BLOOD GAS BALANCES
        # ============================================================
        PaO2 = self.PaO2_func(t)
        PaCO2 = self.PaCO2_func(t)

        VO2 = (self.CMRO2 / 60.0) * (self.brain_mass / 100.0)

        CaO2 = (1.34 * self.Hb_conc * self.SaO2 + 0.0031 * PaO2) / 100.0
        CvO2 = O2v_content / max(Vmicro, 1e-6)

        dO2v_content = (
            CaO2 * alg["Qmicro_in"]
            - CvO2 * alg["Qmicro_out"]
            - VO2
        )

        CaCO2 = (0.03 * PaCO2 + self.HCO3_a) / 100.0
        CvCO2 = CO2v_content / max(Vmicro, 1e-6)

        VCO2 = self.RQ_func * VO2

        dCO2v_content = (
            CaCO2 * alg["Qmicro_in"]
            - CvCO2 * alg["Qmicro_out"]
            + VCO2
        )

        # ============================================================
        # MYOGENIC AUTOREGULATION
        # ============================================================

        A_myo_inf = np.clip(
            ((P_trans - self.P_trans_set) / self.P_trans_set),
            -1.0,
            1.0,
        )

        dA_myo = (A_myo_inf - A_myo) / self.tau_myogenic

        # ============================================================
        # CO2 REACTIVITY
        # ============================================================
        PaCO2 = self.PaCO2_func(t)   
        k_CO2_flow = 0.35  # mL/s per mmHg
        CBF_CO2_target = self.Q0 + k_CO2_flow * (PaCO2 - self.PaCO2_ref)

        A_CO2_inf = np.clip(
            (CBF_CO2_target - self.Q0) / self.Q0,
            -1.0,
            1.0
        )
        dA_CO2 = (A_CO2_inf - A_CO2) / self.tau_CO2

        # ============================================================
        # METABOLIC MECHANISM
        # ============================================================
        o2_deficit = (self.CvO2_ref - CvO2) / max(self.CvO2_ref, 1e-9)
        A_met_inf = np.clip(o2_deficit, -1.0, 1.0)

        dA_met = (A_met_inf - A_met) / self.tau_metabolic

        # ============================================================
        # ENDOTHELIAL / SHEAR MECHANISM
        # ============================================================
        eta = 0.0035
        area = np.pi * (self.radius_MCA * 1e-3) ** 2

        velocity = abs(alg["Q_MCA"]) * 1e-6 / max(area, 1e-12)
        shear = 4.0 * eta * velocity / max(self.radius_MCA * 1e-3, 1e-12)

        A_endo_inf = np.clip(
            (shear - self.shear_ref0) / abs(self.shear_ref0),
            -1.0,
            1.0,
        )

        dA_endo =  (A_endo_inf - A_endo) / self.tau_shear_endo

        return [
            dVMCA,
            dVW,
            dVmicro,
            dVv,
            dO2v_content,
            dCO2v_content,
            dA_myo,
            dA_CO2,
            dA_met,
            dA_endo,
        ]

# ------------------------------------------------------------
# ArrayInput class
# ------------------------------------------------------------
class ArrayInput:
    def __init__(self, t_array, y_array):
        self.t_array = np.asarray(t_array)
        self.y_array = np.asarray(y_array)
    def __call__(self, t):
        return float(np.interp(t, self.t_array, self.y_array))

# ------------------------------------------------------------
# Brady-style Mx calculation for patient
# ------------------------------------------------------------
def compute_mx_brady_patient(t, pressure, flow_signal, epoch_length=300, block_length=10):
    t_start = t[0]
    t_end = t[-1]
    current_epoch_start = t_start+300 # NEW: Skip first 5 minutes to avoid initial transients
    mx_epochs = []

    while current_epoch_start + epoch_length <= t_end:
        mask = (t >= current_epoch_start) & (t < current_epoch_start + epoch_length)
        t_epoch = t[mask]
        p_epoch = pressure[mask]
        f_epoch = flow_signal[mask]

        block_edges = np.arange(t_epoch[0], t_epoch[-1], block_length)
        p_blocks, f_blocks = [], []
        for b_start in block_edges:
            b_mask = (t_epoch >= b_start) & (t_epoch < b_start + block_length)
            if np.sum(b_mask) > 1:
                p_blocks.append(np.mean(p_epoch[b_mask]))
                f_blocks.append(np.mean(f_epoch[b_mask]))

        if p_blocks:
            mx_epochs.append(np.corrcoef(p_blocks, f_blocks)[0,1])
        current_epoch_start += epoch_length

    return np.nanmean(mx_epochs)


# ============================================================
# Fast model simulation for R_span_myo calibration
# ============================================================

def run_model_calibration(
        R_span_myo,
        abp_func,
        paco2_func,
        t_data,
        dt=0.02 # can be adapted depending on performance
):

    """
    Run ICCP model simulation for myogenic gain calibration.

    Only Mxa is calculated to reduce computational cost.
    Full physiological outputs should be extracted using
    the final calibrated simulation.
    """

    subject = deepcopy(
        HealthySubject()
    )

    subject.R_span_myo = R_span_myo


    model = ParallelSplitCerebralRC(
        subject,
        pa_func=abp_func,
        paco2_func=paco2_func
    )


    sol = model.simulate(
        (t_data[0], t_data[-1]),
        dt=dt
    )


    # Extract MCA flow only
    Q_MCA = np.array(
        [
            model.compute_algebraic(
                sol.t[i],
                sol.y[:, i]
            )["Q_MCA"]

            for i in range(len(sol.t))
        ]
    )


    # Convert flow to MCA velocity proxy
    radius = subject.radius_MCA * 1e-3
    area = np.pi * radius**2

    MCAv = (
        Q_MCA * 1e-6
    ) / area * 100


    ABP = np.array(
        [
            abp_func(t)
            for t in sol.t
        ]
    )


    Mxa = compute_mx_brady_patient(
        sol.t,
        ABP,
        MCAv
    )


    return Mxa
