import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

def isentropic_ratios(M, gamma=1.4):
    factor = 1 + (gamma - 1) / 2 * M ** 2
    T0_T = factor
    p0_p = factor ** (gamma / (gamma - 1))
    rho0_rho = factor ** (1 / (gamma - 1))
    return T0_T, p0_p, rho0_rho


def area_mach_ratio(M, gamma=1.4):
    term = (2 / (gamma + 1)) * (1 + (gamma - 1) / 2 * M ** 2)
    exponent = (gamma + 1) / (2 * (gamma - 1))
    return (1 / M) * term ** exponent



def normal_shock(M1, gamma=1.4):
    if M1 < 1:
        raise ValueError("Normal shock relations require M1 >= 1 (supersonic upstream flow).")

    M2 = np.sqrt(
        (1 + (gamma - 1) / 2 * M1 ** 2) / (gamma * M1 ** 2 - (gamma - 1) / 2)
    )
    p2_p1 = 1 + (2 * gamma / (gamma + 1)) * (M1 ** 2 - 1)
    rho2_rho1 = ((gamma + 1) * M1 ** 2) / ((gamma - 1) * M1 ** 2 + 2)
    T2_T1 = p2_p1 / rho2_rho1

    p02_p01 = (
            ((gamma + 1) * M1 ** 2 / ((gamma - 1) * M1 ** 2 + 2)) ** (gamma / (gamma - 1))
            * ((gamma + 1) / (2 * gamma * M1 ** 2 - (gamma - 1))) ** (1 / (gamma - 1))
    )

    return M2, p2_p1, rho2_rho1, T2_T1, p02_p01


def prandtl_meyer(M, gamma=1.4):
    if M < 1:
        raise ValueError("Prandtl-Meyer function requires M >= 1 (supersonic flow).")

    term1 = np.sqrt((gamma + 1) / (gamma - 1))
    term2 = np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (M ** 2 - 1)))
    term3 = np.arctan(np.sqrt(M ** 2 - 1))
    nu = term1 * term2 - term3
    return np.degrees(nu)


def mach_from_pm_angle(nu_target_deg, gamma=1.4, M_min=1.0001, M_max=50):
    func = lambda M: prandtl_meyer(M, gamma) - nu_target_deg
    return brentq(func, M_min, M_max)



def run_validation():
    print("=" * 60)
    print("VALIDATION AGAINST STANDARD TABLES (gamma = 1.4)")
    print("=" * 60)

    T0_T, p0_p, rho0_rho = isentropic_ratios(2.0)
    print(f"\nIsentropic @ M=2.0 (expected: T0/T=1.8000, p0/p=7.8245, rho0/rho=4.3469)")
    print(f"  Computed : T0/T={T0_T:.4f}, p0/p={p0_p:.4f}, rho0/rho={rho0_rho:.4f}")

    M2, p2_p1, rho2_rho1, T2_T1, p02_p01 = normal_shock(2.0)
    print(f"\nNormal Shock @ M1=2.0 (expected: M2=0.5774, p2/p1=4.5000, T2/T1=1.6875)")
    print(f"  Computed : M2={M2:.4f}, p2/p1={p2_p1:.4f}, T2/T1={T2_T1:.4f}")

    nu = prandtl_meyer(2.0)
    print(f"\nPrandtl-Meyer @ M=2.0 (expected: nu=26.38 deg)")
    print(f"  Computed : nu={nu:.2f} deg")

    M_recovered = mach_from_pm_angle(nu)
    print(f"\nInverse PM check: recovering M from nu={nu:.2f} deg -> M={M_recovered:.4f} (expected 2.0000)")
    print("=" * 60 + "\n")


def plot_isentropic_curves(gamma=1.4):
    M = np.linspace(0.01, 5, 300)
    T0_T, p0_p, rho0_rho = isentropic_ratios(M, gamma)

    plt.figure(figsize=(8, 6))
    plt.plot(M, T0_T, label=r'$T_0/T$')
    plt.plot(M, p0_p, label=r'$p_0/p$')
    plt.plot(M, rho0_rho, label=r'$\rho_0/\rho$')
    plt.yscale('log')
    plt.xlabel('Mach Number')
    plt.ylabel('Ratio (log scale)')
    plt.title('Isentropic Flow Property Ratios vs. Mach Number')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('isentropic_curves.png', dpi=150)
    plt.show()


def plot_normal_shock_curves(gamma=1.4):
    M1 = np.linspace(2, 5, 300)
    M2_vals, p2_p1_vals, rho2_rho1_vals, T2_T1_vals = [], [], [], []

    for m in M1:
        M2, p2_p1, rho2_rho1, T2_T1, _ = normal_shock(m, gamma)
        M2_vals.append(M2)
        p2_p1_vals.append(p2_p1)
        rho2_rho1_vals.append(rho2_rho1)
        T2_T1_vals.append(T2_T1)

    plt.figure(figsize=(8, 6))
    plt.plot(M1, M2_vals, label=r'$M_2$ (downstream Mach)')
    plt.plot(M1, p2_p1_vals, label=r'$p_2/p_1$')
    plt.plot(M1, T2_T1_vals, label=r'$T_2/T_1$')
    plt.plot(M1, rho2_rho1_vals, label=r'$\rho_2/\rho_1$')
    plt.xlabel('Upstream Mach Number $M_1$')
    plt.ylabel('Ratio / Downstream Mach Number')
    plt.title('Normal Shock Relations vs. Upstream Mach Number')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('normal_shock_curves.png', dpi=150)
    plt.show()


def plot_prandtl_meyer_curve(gamma=1.4):
    M = np.linspace(1.0001, 5, 300)
    nu = [prandtl_meyer(m, gamma) for m in M]

    plt.figure(figsize=(8, 6))
    plt.plot(M, nu, color='darkorange')
    plt.xlabel('Mach Number')
    plt.ylabel('Prandtl-Meyer Angle, ν (degrees)')
    plt.title('Prandtl-Meyer Expansion Angle vs. Mach Number')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('prandtl_meyer_curve.png', dpi=150)
    plt.show()


def interactive_calculator():
    print("\n" + "=" * 60)
    print("COMPRESSIBLE FLOW CALCULATOR")
    print("=" * 60)

    try:
        M = float(input("\nEnter Mach number: "))
        gamma_input = input("Enter gamma (press Enter for default 1.4): ")
        gamma = float(gamma_input) if gamma_input.strip() else 1.4
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        return

    print(f"\n--- Isentropic Relations (M = {M}) ---")
    T0_T, p0_p, rho0_rho = isentropic_ratios(M, gamma)
    print(f"  T0/T     = {T0_T:.4f}")
    print(f"  p0/p     = {p0_p:.4f}")
    print(f"  rho0/rho = {rho0_rho:.4f}")

    if M >= 1:
        print(f"\n--- Normal Shock Relations (M1 = {M}) ---")
        M2, p2_p1, rho2_rho1, T2_T1, p02_p01 = normal_shock(M, gamma)
        print(f"  M2        = {M2:.4f}")
        print(f"  p2/p1     = {p2_p1:.4f}")
        print(f"  T2/T1     = {T2_T1:.4f}")
        print(f"  rho2/rho1 = {rho2_rho1:.4f}")
        print(f"  p02/p01   = {p02_p01:.4f}  (stagnation pressure loss across shock)")

        print(f"\n--- Prandtl-Meyer Expansion (M = {M}) ---")
        nu = prandtl_meyer(M, gamma)
        print(f"  nu(M) = {nu:.4f} degrees")
    else:
        print("\n(Normal shock and Prandtl-Meyer relations require M >= 1 — skipped.)")

    print("=" * 60 + "\n")



if __name__ == "__main__":
    run_validation()


    print("Generating plots (isentropic, normal shock, Prandtl-Meyer)...\n")
    plot_isentropic_curves()
    plot_normal_shock_curves()
    plot_prandtl_meyer_curve()

    while True:
        interactive_calculator()
        again = input("Calculate another Mach number? (y/n): ").strip().lower()
        if again != 'y':
            print("Exiting. Plots saved as PNG files in the current directory.")
            break