from ..parameters import Parameter
from .derivative_manipulation import invert_function, invert_jacobian, invert_function_interpolation
from .utilities import latex_sqrt, latex_frac, solve_formatter, solve_formatter_tex
import numpy as np
import warnings


class Defaults:
    kT = Parameter(
        value=4.11, lower_bound=0.0, upper_bound=8.0, fixed=True, shared=True, unit="pN*nm"
    )
    Lp = Parameter(value=40.0, lower_bound=0.0, upper_bound=100, unit="nm")
    Lc = Parameter(value=16.0, lower_bound=0.0, upper_bound=np.inf, unit="micron")
    St = Parameter(value=1500.0, lower_bound=0.0, upper_bound=np.inf, unit="pN")
    offset = Parameter(value=0.0, lower_bound=-0.1, upper_bound=0.1, unit="pN")


def offset_equation(x, offset):
    return offset


def offset_equation_tex(x, offset):
    return offset


def distance_offset_model(f, d_offset):
    """Offset on the the model output."""
    return d_offset * np.ones(f.shape)


def force_offset_model(d, f_offset):
    """Offset on the the model output."""
    return f_offset * np.ones(d.shape)


def offset_model_jac(x, offset):
    """Offset on the model output."""
    return np.ones((1, len(x)))


def offset_model_derivative(x, offset):
    """Offset on the model output."""
    return np.zeros(len(x))


def marko_siggia_simplified_equation(d, Lp, Lc, kT):
    return f"({kT}/{Lp}) * ((1/4) * (1-({d}/{Lc}))**(-2) + ({d}/{Lc}) - (1/4))"


def marko_siggia_simplified_equation_tex(d, Lp, Lc, kT):
    return (
        f"\\frac{{{kT}}}{{{Lp}}} \\left(\\frac{{1}}{{4}} \\left(1-\\frac{{{d}}}{{{Lc}}}\\right)^{{-2}} + "
        f"\\frac{{{d}}}{{{Lc}}} - \\frac{{1}}{{4}}\\right)"
    )


def marko_siggia_simplified(d, Lp, Lc, kT):
    """Marko Siggia's Worm-like Chain model based on only entropic contributions. Valid for F < 10 pN).

    References:
        1. J. Marko, E. D. Siggia. Stretching dna., Macromolecules 28.26,
        8759-8770 (1995).
    """
    if np.any(d > Lc):
        warnings.warn(
            "Marko Siggia model is only defined properly up to the contour length (d = Lc)",
            RuntimeWarning,
        )

    d_div_Lc = d / Lc
    return (kT / Lp) * (0.25 * (1.0 - d_div_Lc) ** (-2) + d_div_Lc - 0.25)


def marko_siggia_simplified_jac(d, Lp, Lc, kT):
    return np.vstack(
        (
            -0.25 * Lc ** 2 * kT / (Lp ** 2 * (Lc - d) ** 2)
            + 0.25 * kT / Lp ** 2
            - d * kT / (Lc * Lp ** 2),
            -0.5 * Lc * d * kT / (Lp * (Lc - d) ** 3) - d * kT / (Lc ** 2 * Lp),
            0.25 * Lc ** 2 / (Lp * (Lc - d) ** 2) - 0.25 / Lp + d / (Lc * Lp),
        )
    )


def marko_siggia_simplified_derivative(d, Lp, Lc, kT):
    return 0.5 * Lc ** 2 * kT / (Lp * (Lc - d) ** 3) + kT / (Lc * Lp)


def WLC_equation(f, Lp, Lc, St, kT=4.11):
    return f"{Lc} * (1 - (1/2)*sqrt({kT}/({f}*{Lp})) + {f}/{St})"


def WLC_equation_tex(f, Lp, Lc, St, kT=4.11):
    return f"{Lc} \\left(1 - \\frac{1}{2}\\sqrt{{\\frac{{{kT}}}{{{f} {Lp}}}}} + \\frac{{{f}}}{{{St}}}\\right)"


def WLC(f, Lp, Lc, St, kT=4.11):
    """Odijk's Extensible Worm-like Chain model

    References:
      1. T. Odijk, Stiff Chains and Filaments under Tension, Macromolecules
         28, 7016-7018 (1995).
      2. M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching
         DNA with optical tweezers., Biophysical journal 72, 1335-46 (1997).

    Parameters
    ----------
    f : array_like
        force [pN]
    Lp : float
        persistence length [nm]
    Lc : float
        contour length [um]
    St : float
        stretching modulus [pN]
    kT : float
        Boltzmann's constant times temperature (default = 4.11 [pN nm]) [pN nm]
    """
    return Lc * (1.0 - 1.0 / 2.0 * np.sqrt(kT / (f * Lp)) + f / St)


def WLC_jac(f, Lp, Lc, St, kT=4.11):
    sqrt_term = np.sqrt(kT / (f * Lp))
    return np.vstack(
        (
            0.25 * Lc * sqrt_term / Lp,
            f / St - 0.5 * sqrt_term + 1.0,
            -f * Lc / (St * St),
            -0.25 * Lc * sqrt_term / kT,
        )
    )


def tWLC_equation(f, Lp, Lc, St, C, g0, g1, Fc, kT=4.11):
    g = f"({g0} + clip({g1}, {Fc}, inf))"

    return (
        f"{Lc} * (1 - (1 / 2) * sqrt({kT} / ({f} * {Lp})) + ({C} / (-{g}**2 + {St} * {C})) * {f})"
    )


def tWLC_equation_tex(f, Lp, Lc, St, C, g0, g1, Fc, kT=4.11):
    g = f"\\left({g0} + \\max({g1}, {Fc})\\right)"
    sqrt_term = latex_sqrt(latex_frac(kT, f"{f} {Lp}"))
    stiff_term = latex_frac(C, f"-{g}^2 + {St} {C}")

    return f"{Lc} \\left(1 - \\frac{{1}}{{2}} {sqrt_term} + {stiff_term}{f}\\right)"


def tWLC(f, Lp, Lc, St, C, g0, g1, Fc, kT=4.11):
    """Twistable Worm-like Chain model

    References:
       1. P. Gross et al., Quantifying how DNA stretches, melts and changes
          twist under tension, Nature Physics 7, 731-736 (2011).

    Parameters
    ----------
    f : array_like
        force [pN]
    Lp : float
        persistence length [nm]
    Lc : float
        contour length [um]
    St : float
        stretching modulus [pN]
    C : float
        twist rigidity [pN nm^2]
    g0 : float
        twist stretch coupling [pN Nm]
    g1 : float
        twist stretch coupling [nm]
    Fc : float
        critical force for twist stretch coupling [pN]
    kT : float
        Boltzmann's constant times temperature (default = 4.11 [pN nm]) [pN nm]
    """
    g = np.zeros(np.size(f))
    g[f < Fc] = g0 + g1 * Fc
    g[f >= Fc] = g0 + g1 * f[f >= Fc]

    return Lc * (1.0 - 1.0 / 2.0 * np.sqrt(kT / (f * Lp)) + (C / (-g * g + St * C)) * f)


def tWLC_jac(f, Lp, Lc, St, C, g0, g1, Fc, kT=4.11):
    x0 = 1.0 / Lp
    x1 = np.sqrt(kT * x0 / f)
    x2 = 0.25 * Lc * x1
    x3 = C * f
    x4 = C * St
    x5 = f > Fc
    x6 = f <= Fc
    x7 = f * x5 + Fc * x6
    x8 = g0 + g1 * x7
    x9 = x8 * x8
    x10 = x4 - x9
    x11 = 1.0 / x10
    x12 = x10 ** (-2)
    x13 = 2.0 * Lc * x3
    x14 = 1.0 / x8
    x15 = x11 * x11

    return np.vstack(
        (
            x0 * x2,
            -0.5 * x1 + x11 * x3 + 1.0,
            -C * C * f * Lc * x12,
            Lc * (f * x11 - f * x12 * x4),
            x15 * x13 * x8,
            x15 * x13 * x14 * x7 * x9,
            g1 * x15 * x13 * x14 * x9 * x6,
            -x2 / kT,
        )
    )

    # Not continuous derivatives were removed from the 8th parameter derivative:
    # Original derivative was: g1 * x11 ** 2 * x13 * x14 * x9 * (f * Derivative(x5, Fc) + Fc * Derivative(x6, Fc) + x6)


def coth(x):
    sol = np.ones(x.shape)
    mask = abs(x) < 500
    sol[mask] = np.cosh(x[mask]) / np.sinh(
        x[mask]
    )  # Crude overflow protection, this limit approaches 1.0
    mask = abs(x) < -500
    sol[mask] = -1.0  # Crude overflow protection, this limit approaches -1.0
    return sol


def FJC_equation(f, Lp, Lc, St, kT=4.11):
    return f"{Lc} * (coth(2.0 * {f} * {Lp} / {kT}) - {kT} / (2 * {f} * {Lp})) * (1 + {f}/{St})"


def FJC_equation_tex(f, Lp, Lc, St, kT=4.11):
    frac1 = latex_frac(f"{f} {Lp}", kT)
    frac2 = latex_frac(kT, f"2 {f} {Lp}")
    frac3 = latex_frac(f, St)

    return f"{Lc} \\left(coth\\left(2 {frac1}\\right) - {frac2}\\right) \\left(1 + {frac3}\\right)"


def FJC(f, Lp, Lc, St, kT=4.11):
    """Freely-Jointed Chain

    References:
       1. S. B. Smith, Y. Cui, C. Bustamante, Overstretching B-DNA: The
          Elastic Response of Individual Double-Stranded and Single-Stranded
          DNA Molecules, Science 271, 795-799 (1996).
       2. M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching
          DNA with optical tweezers., Biophysical journal 72, 1335-46 (1997).

    Parameters
    ----------
    F : array_like
        force [pN]
    Lp : float
        persistence length [nm]
    Lc : float
        contour length [um]
    St : float
        elastic modulus [pN]
    kT : float
        Boltzmann's constant times temperature (default = 4.11 [pN nm]) [pN nm]
    """
    return Lc * (coth(2.0 * f * Lp / kT) - kT / (2.0 * f * Lp)) * (1.0 + f / St)


def FJC_jac(f, Lp, Lc, St, kT=4.11):
    x0 = 0.5 / f
    x1 = 2.0 * f / kT
    x2 = Lp * x1
    x3 = np.zeros(x2.shape)
    x3[abs(x2) < 300] = np.sinh(x2[abs(x2) < 300]) ** (-2)
    x4 = f / St + 1.0
    x5 = Lc * x4
    x6 = x0 / Lp
    x7 = -kT * x6 + coth(x2)
    return np.vstack(
        (
            x5 * (-x1 * x3 + kT * x0 / (Lp * Lp)),
            x4 * x7,
            -f * Lc * x7 / (St * St),
            x5 * (2.0 * f * Lp * x3 / (kT * kT) - x6),
        )
    )


def solve_cubic_wlc(a, b, c, selected_root):
    # Convert the equation to a depressed cubic for p and q, which'll allow us to greatly simplify the equations
    p = b - a * a / 3.0
    q = 2 * a * a * a / 27.0 - a * b / 3.0 + c
    det = q * q / 4.0 + p * p * p / 27.0

    # The model changes behaviours when the discriminant equates to zero. From this point we need a different root
    # resolution mechanism.
    sol = np.zeros(det.shape)
    mask = det >= 0

    sqrt_det = np.sqrt(det[mask])
    t1 = -q[mask] * 0.5 + sqrt_det
    t2 = -q[mask] * 0.5 - sqrt_det
    sol[mask] = np.cbrt(t1) + np.cbrt(t2)

    sqrt_minus_p = np.sqrt(-p[np.logical_not(mask)])
    q_masked = q[np.logical_not(mask)]

    asin_argument = 3.0 * np.sqrt(3.0) * q_masked / (2.0 * sqrt_minus_p ** 3)
    asin_argument = np.clip(asin_argument, -1.0, 1.0)

    if selected_root == 0:
        sol[np.logical_not(mask)] = (
            2.0 / np.sqrt(3.0) * sqrt_minus_p * np.sin((1.0 / 3.0) * np.arcsin(asin_argument))
        )
    elif selected_root == 1:
        sol[np.logical_not(mask)] = (
            -2.0
            / np.sqrt(3.0)
            * sqrt_minus_p
            * np.sin((1.0 / 3.0) * np.arcsin(asin_argument) + np.pi / 3.0)
        )
    elif selected_root == 2:
        sol[np.logical_not(mask)] = (
            2.0
            / np.sqrt(3.0)
            * sqrt_minus_p
            * np.cos((1.0 / 3.0) * np.arcsin(asin_argument) + np.pi / 6.0)
        )
    else:
        raise RuntimeError("Invalid root selected. Choose 0, 1 or 2.")

    return sol - a / 3.0


def invWLC_equation(d, Lp, Lc, St, kT=4.11):
    return solve_formatter(WLC_equation("f", Lp, Lc, St, kT), "f", d)


def invWLC_equation_tex(d, Lp, Lc, St, kT=4.11):
    return solve_formatter_tex(WLC_equation_tex("f", Lp, Lc, St, kT), "f", d)


def invWLC(d, Lp, Lc, St, kT=4.11):
    """Inverted Odijk's Worm-like Chain model

    References:
      1. T. Odijk, Stiff Chains and Filaments under Tension, Macromolecules
         28, 7016-7018 (1995).
      2. M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching
         DNA with optical tweezers., Biophysical journal 72, 1335-46 (1997).

    Parameters
    ----------
    d : array_like
        extension [um]
    Lp : float
        persistence length [nm]
    Lc : float
        contour length [um]
    St : float
        stretching modulus [pN]
    kT : float
        Boltzmann's constant times temperature (default = 4.11 [pN nm]) [pN nm]
    """
    #
    # In short, this model was analytically solved, since it is basically a cubic equation. There are some issues
    # with this inverted model. Its derivatives are not defined everywhere, as they contain divisions by zero in
    # specific places due to the cube roots involved. This was worked around by preventing the denominators in these
    # equations to go to zero by clamping them to a lower bound. This results in far better numerical behaviour than
    # even the finite difference approximations.
    #
    # Define:
    #   alpha = (distance/Lc) - 1.0
    #   beta = 1.0 / St
    #   gamma = kT / Lp
    #
    # This allows us to obtain simple polynomial coefficients for the Odijk model. We divide the polynomial by the
    # leading coefficient to make things simpler for us. This leads to the following equations:
    #
    #   denom = beta ** 2.0
    #   a = - (2.0 * alpha * beta) / denom = - 2.0 * alpha / beta
    #   b = alpha * alpha / denom = (alpha * alpha) / (beta * beta)
    #   c = - 0.25 * gamma / denom = - gamma / (4 * beta * beta)
    #
    # We can see now that parameterizing w.r.t. St is easier than b and define:
    alpha = (d / Lc) - 1.0
    gamma = kT / Lp

    a = -2.0 * alpha * St
    b = (alpha * alpha) * (St * St)
    c = -0.25 * gamma * (St * St)

    return solve_cubic_wlc(a, b, c, 2)


def calc_root1_invwlc(det, p, q, dp_da, dq_da, dq_db):
    # Calculate the first root for det < 0
    # Note that dp/dc = 0, dp_db = 1, dq_dc = 1

    sqrt_det = np.sqrt(det)
    term1 = np.abs(sqrt_det - 0.5 * q)  # Technically, the absolute is not part of this solution.
    term2 = np.abs(
        -sqrt_det - 0.5 * q
    )  # But it can cause numerical issues if we raise negative values to a root.
    t1 = term1 ** (2 / 3)
    t2 = term2 ** (2 / 3)

    # Derivatives of cube roots are not defined everywhere.
    #
    # The derivatives go over the bit where the cube roots are non-differentiable in the region of the model where it
    # switches from entropic to enthalpic behaviour. Even the finite difference derivatives look terrible here.
    #
    # When we approach Lc, t2 tends to zero, which causes problems with the division later.
    t1[abs(t1) < 1e-5] = 1e-5
    t2[abs(t2) < 1e-5] = 1e-5

    # When the discriminant goes to zero however, it means there are now repeated roots.
    sqrt_det[abs(sqrt_det) < 1e-5] = 1e-5

    # Compute all the elements required to evaluate the chain rule
    dy_ddet = 1.0 / (6.0 * sqrt_det * t1) - 1.0 / (6.0 * sqrt_det * t2)
    dy_dq = -1.0 / (6.0 * t1) - 1.0 / (6.0 * t2)
    dy_da = -(1.0 / 3.0)

    ddet_dp = p * p / 9.0
    ddet_dq = 0.5 * q

    # Total derivatives, terms multiplied by zero are omitted. Terms that are one are also omitted.
    # dp_db = dq_dc = 1
    total_dy_da = dy_ddet * ddet_dp * dp_da + dy_ddet * ddet_dq * dq_da + dy_dq * dq_da + dy_da
    total_dy_db = dy_ddet * ddet_dp + dy_ddet * ddet_dq * dq_db + dy_dq * dq_db
    total_dy_dc = dy_ddet * ddet_dq + dy_dq

    return total_dy_da, total_dy_db, total_dy_dc


def calc_triple_root_invwlc(p, q, dp_da, dq_da, dq_db, root):
    # If we define:
    #   sqmp = sqrt(-p)
    #   F = 3 * sqrt(3) * q / (2 * sqmp**3 )
    #
    # Then the solution is:
    #   2 /sqrt(3) * sqmp * cos((1/3) * asin(F) + pi/6) - a / 3
    #
    # Note that dp/dc = 0, dp_db = 1, dq_dc = 1
    # The rest of this file is simply applying the chain rule.
    sqmp = np.sqrt(-p)
    F = 3.0 * np.sqrt(3.0) * q / (2.0 * sqmp ** 3)

    dF_dsqmp = -9 * np.sqrt(3) * q / (2 * sqmp ** 4)
    dF_dq = 3.0 * np.sqrt(3) / (2.0 * sqmp ** 3)
    dsqmp_dp = -1.0 / (2.0 * sqmp)
    dy_da = -1.0 / 3.0

    if root == 0:
        arg = np.arcsin(F) / 3.0
        dy_dsqmp = 2.0 * np.sqrt(3.0) * np.sin(arg) / 3.0
        dy_dF = 2.0 * np.sqrt(3.0) * sqmp * np.cos(arg) / (9.0 * np.sqrt(1.0 - F ** 2))
    elif root == 1:
        arg = np.arcsin(F) / 3.0 + np.pi / 3.0
        dy_dsqmp = -2.0 * np.sqrt(3.0) * np.sin(arg) / 3.0
        dy_dF = -2.0 * np.sqrt(3.0) * sqmp * np.cos(arg) / (9.0 * np.sqrt(1.0 - F ** 2))
    elif root == 2:
        arg = np.arcsin(F) / 3.0 + np.pi / 6.0
        dy_dsqmp = 2.0 * np.sqrt(3.0) * np.cos(arg) / 3.0
        dy_dF = -2.0 * np.sqrt(3.0) * sqmp * np.sin(arg) / (9.0 * np.sqrt(1.0 - F * F))

    # Total derivatives
    total_dy_da = (
        dy_dsqmp * dsqmp_dp * dp_da + dy_dF * (dF_dsqmp * dsqmp_dp * dp_da + dF_dq * dq_da) + dy_da
    )
    total_dy_db = dy_dsqmp * dsqmp_dp + dy_dF * (dF_dsqmp * dsqmp_dp + dF_dq * dq_db)
    total_dy_dc = dy_dF * dF_dq

    return total_dy_da, total_dy_db, total_dy_dc


def invwlc_root_derivatives(a, b, c, selected_root):
    """Calculate the root derivatives of a cubic polynomial with respect to the polynomial coefficients.

    For a polynomial of the form:
        x**3 + a * x**2 + b * x + c = 0

    Note that this is not a general root-finding function, but tailored for use with the inverted WLC model. For det < 0
    it returns the derivatives of the first root. For det > 0 it returns the derivatives of the third root.

    Parameters
    ----------
    a, b, c: array_like
        Coefficients of the reduced cubic polynomial.
        x**3 + a x**2 + b x + c = 0
    selected_root: integer
        which root to compute the derivative of
    """
    p = b - a * a / 3.0
    q = 2.0 * a * a * a / 27.0 - a * b / 3.0 + c
    det = q * q / 4.0 + p * p * p / 27.0

    # Determine derivatives of our transformation from polynomial coefficients to helper coordinates p and q
    dp_da = -2.0 * a / 3.0
    dq_da = 2.0 * a ** 2.0 / 9.0 - b / 3.0
    dq_db = -a / 3.0

    total_dy_da = np.zeros(det.shape)
    total_dy_db = np.zeros(det.shape)
    total_dy_dc = np.zeros(det.shape)

    mask = det > 0
    total_dy_da[mask], total_dy_db[mask], total_dy_dc[mask] = calc_root1_invwlc(
        det[mask], p[mask], q[mask], dp_da[mask], dq_da[mask], dq_db[mask]
    )

    nmask = np.logical_not(mask)
    total_dy_da[nmask], total_dy_db[nmask], total_dy_dc[nmask] = calc_triple_root_invwlc(
        p[nmask], q[nmask], dp_da[nmask], dq_da[nmask], dq_db[nmask], selected_root
    )

    return total_dy_da, total_dy_db, total_dy_dc


def invWLC_jac(d, Lp, Lc, St, kT=4.11):
    alpha = (d / Lc) - 1.0
    gamma = kT / Lp

    St_squared = St * St
    a = -2.0 * alpha * St
    b = (alpha * alpha) * St_squared
    c = -0.25 * gamma * St_squared

    total_dy_da, total_dy_db, total_dy_dc = invwlc_root_derivatives(a, b, c, 2)

    # Map back to our output parameters
    da_dLc = 2.0 * St * d / Lc ** 2
    da_dSt = -2.0 * alpha
    db_dLc = -2.0 * St ** 2 * d * alpha / Lc ** 2
    db_dSt = 2.0 * St * alpha ** 2
    dc_dLp = 0.25 * St_squared * kT / Lp ** 2
    dc_dSt = -0.5 * St * gamma
    dc_dkT = -0.25 * St_squared / Lp

    # Terms multiplied by zero are omitted. Terms that are one are also omitted.
    total_dy_dLp = total_dy_dc * dc_dLp
    total_dy_dLc = total_dy_da * da_dLc + total_dy_db * db_dLc
    total_dy_dSt = total_dy_da * da_dSt + total_dy_db * db_dSt + total_dy_dc * dc_dSt
    total_dy_dkT = total_dy_dc * dc_dkT

    return [total_dy_dLp, total_dy_dLc, total_dy_dSt, total_dy_dkT]


def invWLC_derivative(d, Lp, Lc, St, kT=4.11):
    alpha = (d / Lc) - 1.0
    gamma = kT / Lp

    St_squared = St * St
    a = -2.0 * alpha * St
    b = (alpha * alpha) * St_squared
    c = -0.25 * gamma * St_squared

    total_dy_da, total_dy_db, _ = invwlc_root_derivatives(a, b, c, 2)

    # Map back to our output parameters
    da_dd = -2.0 * St / Lc
    db_dd = 2 * St ** 2 * (-1.0 + d / Lc) / Lc

    return total_dy_da * da_dd + total_dy_db * db_dd


def WLC_derivative(f, Lp, Lc, St, kT=4.11):
    x0 = 1.0 / f
    return Lc * (0.25 * x0 * np.sqrt(kT * x0 / Lp) + 1.0 / St)


def tWLC_derivative(f, Lp, Lc, St, C, g0, g1, Fc, kT):
    """Derivative of the tWLC model w.r.t. the independent variable"""
    x0 = 1.0 / f
    x1 = f > Fc
    x2 = f <= Fc
    x3 = g0 + g1 * (f * x1 + Fc * x2)
    x4 = x3 * x3
    x5 = 1.0 / (C * St - x4)

    # The derivative terms were omitted since they are incompatible with a smooth optimization algorithm.
    # Lc * (2.0 * C * f * g1 * x4 * x5 * x5 * (f * Derivative(x1, f) + Fc * Derivative(x2, f) + x1) / x3 + C * x5 + 0.25 * x0 * sqrt(kT * x0 / Lp))]
    return Lc * (
        2.0 * C * f * g1 * x4 * x5 * x5 * x1 / x3 + C * x5 + 0.25 * x0 * np.sqrt(kT * x0 / Lp)
    )


def FJC_derivative(f, Lp, Lc, St, kT=4.11):
    """Derivative of the FJC model w.r.t. the independent variable"""
    x0 = 1.0 / St
    x1 = 2.0 * Lp / kT
    x2 = f * x1
    x3 = 0.5 * kT / Lp

    # Overflow protection
    sinh_term = np.zeros(x2.shape)
    sinh_term[x2 < 300] = 1.0 / np.sinh(x2[x2 < 300]) ** 2

    return Lc * x0 * (coth(x2) - x3 / f) + Lc * (f * x0 + 1.0) * (-x1 * sinh_term + x3 / f ** 2)


def invtWLC_equation(d, Lp, Lc, St, C, g0, g1, Fc, kT=4.11):
    return solve_formatter(tWLC_equation_tex("f", Lp, Lc, St, C, g0, g1, Fc, kT=4.11), "f", d)


def invtWLC_equation_tex(d, Lp, Lc, St, C, g0, g1, Fc, kT=4.11):
    return solve_formatter_tex(tWLC_equation_tex("f", Lp, Lc, St, C, g0, g1, Fc, kT=4.11), "f", d)


def invtWLC(d, Lp, Lc, St, C, g0, g1, Fc, kT=4.11):
    """Inverted Twistable Worm-like Chain model

    References:
       1. P. Gross et al., Quantifying how DNA stretches, melts and changes
          twist under tension, Nature Physics 7, 731-736 (2011).

    Parameters
    ----------
    d : array_like
        distance [um]
    Lp : float
        persistence length [nm]
    Lc : float
        contour length [um]
    St : float
        stretching modulus [pN]
    C : float
        twist rigidity [pN nm^2]
    g0 : float
        twist stretch coupling [pN Nm]
    g1 : float
        twist stretch coupling [nm]
    Fc : float
        critical force for twist stretch coupling [pN]
    kT : float
        Boltzmann's constant times temperature (default = 4.11) [pN nm]
    """
    f_min = 0
    f_max = (-g0 + np.sqrt(St * C)) / g1  # Above this force the model loses its validity

    return invert_function_interpolation(
        d,
        1.0,
        f_min,
        f_max,
        lambda f_trial: tWLC(f_trial, Lp, Lc, St, C, g0, g1, Fc, kT),
        lambda f_trial: tWLC_derivative(f_trial, Lp, Lc, St, C, g0, g1, Fc, kT),
    )


def invtWLC_jac(d, Lp, Lc, St, C, g0, g1, Fc, kT=4.11):
    return invert_jacobian(
        d,
        lambda f_trial: invtWLC(f_trial, Lp, Lc, St, C, g0, g1, Fc, kT),
        lambda f_trial: tWLC_jac(f_trial, Lp, Lc, St, C, g0, g1, Fc, kT),
        lambda f_trial: tWLC_derivative(f_trial, Lp, Lc, St, C, g0, g1, Fc, kT),
    )


def invFJC(d, Lp, Lc, St, kT=4.11):
    """Inverted Freely-Jointed Chain

    References:
       1. S. B. Smith, Y. Cui, C. Bustamante, Overstretching B-DNA: The
          Elastic Response of Individual Double-Stranded and Single-Stranded
          DNA Molecules, Science 271, 795-799 (1996).
       2. M. D. Wang, H. Yin, R. Landick, J. Gelles, S. M. Block, Stretching
          DNA with optical tweezers., Biophysical journal 72, 1335-46 (1997).

    Parameters
    ----------
    d : array_like
        distance [um]
    Lp : float
        persistence length [nm]
    Lc : float
        contour length [um]
    St : float
        elastic modulus [pN]
    kT : float
        Boltzmann's constant times temperature (default = 4.11 [pN nm]) [pN nm]
    """
    f_min = 0
    f_max = np.inf

    return invert_function(
        d,
        1.0,
        f_min,
        f_max,
        lambda f_trial: FJC(f_trial, Lp, Lc, St, kT),
        lambda f_trial: FJC_derivative(f_trial, Lp, Lc, St, kT),
    )


def marko_siggia_ewlc_solve_force_equation(d, Lp, Lc, St, kT=4.11):
    return solve_formatter(
        f"(1/4) * (1 - ({d}/{Lc}) + (f/{St}))**(-2) - (1/4) + ({d}/{Lc}) - (f/{St})",
        "f",
        f"f*{Lp}/{kT}",
    )


def marko_siggia_ewlc_solve_force_equation_tex(d, Lp, Lc, St, kT=4.11):
    dLc = latex_frac(d, Lc)
    FSt = latex_frac("f", St)
    lhs = latex_frac(f"f {Lp}", kT)

    return solve_formatter_tex(
        f"{latex_frac(1, 4)}\\left(1 - {dLc} + {FSt}\\right)^{{-2}} - {latex_frac(1, 4)} + "
        f"{dLc} - {FSt}",
        "f",
        lhs,
    )


def marko_siggia_ewlc_solve_force(d, Lp, Lc, St, kT=4.11):
    """Margo-Siggia's Worm-like Chain model with distance as dependent parameter (useful for F < 10 pN).
    These equations were symbolically derived. The expressions are not pretty, but they work."""
    c = -(St ** 3) * d * kT * (1.5 * Lc ** 2 - 2.25 * Lc * d + d ** 2) / (Lc ** 3 * (Lp * St + kT))
    b = (
        St ** 2
        * (
            Lc ** 2 * Lp * St
            + 1.5 * Lc ** 2 * kT
            - 2 * Lc * Lp * St * d
            - 4.5 * Lc * d * kT
            + Lp * St * d ** 2
            + 3 * d ** 2 * kT
        )
        / (Lc ** 2 * (Lp * St + kT))
    )
    a = (
        St
        * (2 * Lc * Lp * St + 2.25 * Lc * kT - 2 * Lp * St * d - 3 * d * kT)
        / (Lc * (Lp * St + kT))
    )

    return solve_cubic_wlc(a, b, c, 2)


def marko_siggia_ewlc_solve_force_jac(d, Lp, Lc, St, kT=4.11):
    c = -(St ** 3) * d * kT * (1.5 * Lc ** 2 - 2.25 * Lc * d + d ** 2) / (Lc ** 3 * (Lp * St + kT))
    b = (
        St ** 2
        * (
            Lc ** 2 * Lp * St
            + 1.5 * Lc ** 2 * kT
            - 2 * Lc * Lp * St * d
            - 4.5 * Lc * d * kT
            + Lp * St * d ** 2
            + 3 * d ** 2 * kT
        )
        / (Lc ** 2 * (Lp * St + kT))
    )
    a = (
        St
        * (2 * Lc * Lp * St + 2.25 * Lc * kT - 2 * Lp * St * d - 3 * d * kT)
        / (Lc * (Lp * St + kT))
    )

    total_dy_da, total_dy_db, total_dy_dc = invwlc_root_derivatives(a, b, c, 2)

    # Map back to our output parameters
    denom1 = Lc ** 3 * (Lp * St + kT) ** 2
    denom2 = Lc * (Lp ** 2 * St ** 2 + 2.0 * Lp * St * kT + kT ** 2)
    dkT = d * kT
    dc_dLc = (
        St ** 3 * dkT * (1.5 * Lc ** 2 - 4.5 * Lc * d + 3.0 * d ** 2) / (Lc ** 4 * (Lp * St + kT))
    )
    dc_dSt = (
        -(St ** 2)
        * dkT
        * (2 * Lp * St + 3 * kT)
        * (1.5 * Lc ** 2 - 2.25 * Lc * d + d ** 2)
        / denom1
    )
    dc_dLp = St ** 4 * dkT * (1.5 * Lc ** 2 - 2.25 * Lc * d + d ** 2) / denom1
    dc_dkT = -Lp * St ** 4 * d * (1.5 * Lc ** 2 - 2.25 * Lc * d + d ** 2) / denom1
    db_dLc = (
        St ** 2
        * d
        * (2.0 * Lc * Lp * St + 4.5 * Lc * kT - 2.0 * Lp * St * d - 6.0 * d * kT)
        / (Lc ** 3 * (Lp * St + kT))
    )
    db_dSt = (
        St
        * (
            2.0 * Lc ** 2 * Lp ** 2 * St ** 2
            + 4.5 * Lc ** 2 * Lp * St * kT
            + 3.0 * Lc ** 2 * kT ** 2
            - 4.0 * Lc * Lp ** 2 * St ** 2 * d
            - 10.5 * Lc * Lp * St * d * kT
            - 9.0 * Lc * d * kT ** 2
            + 2.0 * Lp ** 2 * St ** 2 * d ** 2
            + 6.0 * Lp * St * d ** 2 * kT
            + 6.0 * d ** 2 * kT ** 2
        )
        / (Lc * denom2)
    )
    db_dLp = -(St ** 3) * kT * (0.5 * Lc ** 2 - 2.5 * Lc * d + 2.0 * d ** 2) / (Lc * denom2)
    db_dkT = Lp * St ** 3 * (0.5 * Lc ** 2 - 2.5 * Lc * d + 2.0 * d ** 2) / (Lc * denom2)
    da_dLc = St * d * (2 * Lp * St + 3 * kT) / (Lc ** 2 * (Lp * St + kT))
    da_dSt = (
        -Lp * St * (2 * Lc * Lp * St + 2.25 * Lc * kT - 2 * Lp * St * d - 3 * d * kT)
        + (Lp * St + kT)
        * (
            2 * Lc * Lp * St
            + 2.25 * Lc * kT
            - 2 * Lp * St * d
            + 2 * Lp * St * (Lc - d)
            - 3 * d * kT
        )
    ) / (Lc * (Lp * St + kT) ** 2)
    da_dLp = -(St ** 2) * kT * (0.25 * Lc - d) / denom2
    da_dkT = Lp * St ** 2 * (0.25 * Lc - d) / denom2

    # Terms multiplied by zero are omitted. Terms that are one are also omitted.
    total_dy_dLp = total_dy_da * da_dLp + total_dy_db * db_dLp + total_dy_dc * dc_dLp
    total_dy_dLc = total_dy_da * da_dLc + total_dy_db * db_dLc + total_dy_dc * dc_dLc
    total_dy_dSt = total_dy_da * da_dSt + total_dy_db * db_dSt + total_dy_dc * dc_dSt
    total_dy_dkT = total_dy_da * da_dkT + total_dy_db * db_dkT + total_dy_dc * dc_dkT

    return [total_dy_dLp, total_dy_dLc, total_dy_dSt, total_dy_dkT]


def marko_siggia_ewlc_solve_force_derivative(d, Lp, Lc, St, kT=4.11):
    c = -(St ** 3) * d * kT * (1.5 * Lc ** 2 - 2.25 * Lc * d + d ** 2) / (Lc ** 3 * (Lp * St + kT))
    b = (
        St ** 2
        * (
            Lc ** 2 * Lp * St
            + 1.5 * Lc ** 2 * kT
            - 2 * Lc * Lp * St * d
            - 4.5 * Lc * d * kT
            + Lp * St * d ** 2
            + 3 * d ** 2 * kT
        )
        / (Lc ** 2 * (Lp * St + kT))
    )
    a = (
        St
        * (2 * Lc * Lp * St + 2.25 * Lc * kT - 2 * Lp * St * d - 3 * d * kT)
        / (Lc * (Lp * St + kT))
    )

    total_dy_da, total_dy_db, total_dy_dc = invwlc_root_derivatives(a, b, c, 2)

    # Map back to our output parameters
    denom = Lc * (Lp * St + kT)
    dc_dd = -(St ** 3) * kT * (1.5 * Lc ** 2 - 4.5 * Lc * d + 3.0 * d ** 2) / (Lc * Lc * denom)
    db_dd = (
        St ** 2 * (-2 * Lc * Lp * St - 4.5 * Lc * kT + 2 * Lp * St * d + 6 * d * kT) / (Lc * denom)
    )
    da_dd = -St * (2 * Lp * St + 3 * kT) / denom

    return total_dy_da * da_dd + total_dy_db * db_dd + total_dy_dc * dc_dd


def marko_siggia_ewlc_solve_distance_equation(f, Lp, Lc, St, kT=4.11):
    return solve_formatter(
        f"(1/4) * (1 - (d/{Lc}) + ({f}/{St}))**(-2) - (1/4) + (d/{Lc}) - ({f}/{St})",
        "d",
        f"{f}*{Lp}/{kT}",
    )


def marko_siggia_ewlc_solve_distance_equation_tex(f, Lp, Lc, St, kT=4.11):
    dLc = latex_frac("d", Lc)
    FSt = latex_frac(f, St)
    lhs = latex_frac(f"{f} {Lp}", kT)

    return solve_formatter_tex(
        f"{latex_frac(1, 4)}\\left(1 - {dLc} + {FSt}\\right)^{{-2}} - {latex_frac(1, 4)} + "
        f"{dLc} - {FSt}",
        "d",
        lhs,
    )


def inverted_marko_siggia_simplified_coefficients(f, Lp, Lc, kT):
    a = -Lc * (f * Lp / kT + 2.25)
    b = Lc ** 2.0 * (2.0 * f * Lp / kT + 1.5)
    c = -f * Lc ** 3.0 * Lp / kT
    return a, b, c


def inverted_marko_siggia_simplified(f, Lp, Lc, kT=4.11):
    a, b, c = inverted_marko_siggia_simplified_coefficients(f, Lp, Lc, kT)

    return solve_cubic_wlc(a, b, c, 1)


def inverted_marko_siggia_simplified_jac(f, Lp, Lc, kT=4.11):
    a, b, c = inverted_marko_siggia_simplified_coefficients(f, Lp, Lc, kT)

    total_dy_da, total_dy_db, total_dy_dc = invwlc_root_derivatives(a, b, c, 1)

    dc_dLc = -3.0 * f * Lc ** 2 * Lp / kT
    dc_dLp = -f * Lc ** 3 / kT
    dc_dkT = f * Lc ** 3 * Lp / kT ** 2
    db_dLc = Lc * (4.0 * f * Lp / kT + 3.0)
    db_dLp = 2.0 * f * Lc ** 2 / kT
    db_dkT = -2.0 * f * Lc ** 2 * Lp / (kT ** 2)
    da_dLc = -f * Lp / kT - 2.25
    da_dLp = -f * Lc / kT
    da_dkT = f * Lc * Lp / kT ** 2

    # Terms multiplied by zero are omitted. Terms that are one are also omitted.
    total_dy_dLp = total_dy_da * da_dLp + total_dy_db * db_dLp + total_dy_dc * dc_dLp
    total_dy_dLc = total_dy_da * da_dLc + total_dy_db * db_dLc + total_dy_dc * dc_dLc
    total_dy_dkT = total_dy_da * da_dkT + total_dy_db * db_dkT + total_dy_dc * dc_dkT

    return [total_dy_dLp, total_dy_dLc, total_dy_dkT]


def inverted_marko_siggia_simplified_derivative(f, Lp, Lc, kT=4.11):
    a, b, c = inverted_marko_siggia_simplified_coefficients(f, Lp, Lc, kT)
    total_dy_da, total_dy_db, total_dy_dc = invwlc_root_derivatives(a, b, c, 1)

    da_df = -Lc * Lp / kT
    db_df = 2.0 * Lc ** 2 * Lp / kT
    dc_df = -(Lc ** 3) * Lp / kT

    return total_dy_da * da_df + total_dy_db * db_df + total_dy_dc * dc_df


def inverted_marko_siggia_simplified_equation(f, Lp, Lc, kT=4.11):
    return solve_formatter(
        f"(1/4) * (1 - (d/{Lc}))**(-2) - (1/4) + (d/{Lc})", "d", f"{f}*{Lp}/{kT}"
    )


def inverted_marko_siggia_simplified_equation_tex(f, Lp, Lc, kT=4.11):
    dLc = latex_frac("d", Lc)
    lhs = latex_frac(f"{f} {Lp}", kT)

    return solve_formatter_tex(
        f"{latex_frac(1, 4)}\\left(1 - {dLc}\\right)^{{-2}} - {latex_frac(1, 4)} + {dLc}", "d", lhs
    )


def marko_siggia_ewlc_solve_distance(f, Lp, Lc, St, kT=4.11):
    c = (
        -f
        * Lc ** 3
        * (
            f ** 2 * Lp * St
            + f ** 2 * kT
            + 2 * f * Lp * St ** 2
            + 2.25 * f * St * kT
            + Lp * St ** 3
            + 1.5 * St ** 2 * kT
        )
        / (St ** 3 * kT)
    )
    b = (
        Lc ** 2
        * (
            2 * f ** 2 * Lp * St
            + 3 * f ** 2 * kT
            + 2 * f * Lp * St ** 2
            + 4.5 * f * St * kT
            + 1.5 * St ** 2 * kT
        )
        / (St ** 2 * kT)
    )
    a = -f * Lc * Lp / kT - 3 * f * Lc / St - 2.25 * Lc

    return solve_cubic_wlc(a, b, c, 1)


def marko_siggia_ewlc_solve_distance_jac(f, Lp, Lc, St, kT=4.11):
    c = (
        -f
        * Lc ** 3
        * (
            f ** 2 * Lp * St
            + f ** 2 * kT
            + 2 * f * Lp * St ** 2
            + 2.25 * f * St * kT
            + Lp * St ** 3
            + 1.5 * St ** 2 * kT
        )
        / (St ** 3 * kT)
    )
    b = (
        Lc ** 2
        * (
            2 * f ** 2 * Lp * St
            + 3 * f ** 2 * kT
            + 2 * f * Lp * St ** 2
            + 4.5 * f * St * kT
            + 1.5 * St ** 2 * kT
        )
        / (St ** 2 * kT)
    )
    a = -f * Lc * Lp / kT - 3 * f * Lc / St - 2.25 * Lc

    total_dy_da, total_dy_db, total_dy_dc = invwlc_root_derivatives(a, b, c, 1)

    # Map back to our output parameters
    dc_dLc = (
        -3
        * f
        * Lc ** 2
        * (
            f ** 2 * Lp * St
            + f ** 2 * kT
            + 2 * f * Lp * St ** 2
            + 2.25 * f * St * kT
            + Lp * St ** 3
            + 1.5 * St ** 2 * kT
        )
        / (St ** 3 * kT)
    )
    dc_dSt = (
        f
        * Lc ** 3
        * (
            2.0 * f ** 2 * Lp * St
            + 3.0 * f ** 2 * kT
            + 2.0 * f * Lp * St ** 2
            + 4.5 * f * St * kT
            + 1.5 * St ** 2 * kT
        )
        / (St ** 4 * kT)
    )
    dc_dLp = -f * Lc ** 3 * (f ** 2 + 2 * f * St + St ** 2) / (St ** 2 * kT)
    dc_dkT = f * Lc ** 3 * Lp * (f ** 2 + 2 * f * St + St ** 2) / (St ** 2 * kT ** 2)
    db_dLc = (
        4 * f ** 2 * Lc * Lp / (St * kT)
        + 6 * f ** 2 * Lc / St ** 2
        + 4 * f * Lc * Lp / kT
        + 9.0 * f * Lc / St
        + 3.0 * Lc
    )
    db_dSt = -f * Lc ** 2 * (2.0 * f * Lp * St + 6.0 * f * kT + 4.5 * St * kT) / (St ** 3 * kT)
    db_dLp = 2 * f * Lc ** 2 * (f + St) / (St * kT)
    db_dkT = -2 * f * Lc ** 2 * Lp * (f + St) / (St * kT ** 2)
    da_dLc = -f * Lp / kT - 3 * f / St - 2.25
    da_dSt = 3 * f * Lc / St ** 2
    da_dLp = -f * Lc / kT
    da_dkT = f * Lc * Lp / kT ** 2

    # Terms multiplied by zero are omitted. Terms that are one are also omitted.
    total_dy_dLp = total_dy_da * da_dLp + total_dy_db * db_dLp + total_dy_dc * dc_dLp
    total_dy_dLc = total_dy_da * da_dLc + total_dy_db * db_dLc + total_dy_dc * dc_dLc
    total_dy_dSt = total_dy_da * da_dSt + total_dy_db * db_dSt + total_dy_dc * dc_dSt
    total_dy_dkT = total_dy_da * da_dkT + total_dy_db * db_dkT + total_dy_dc * dc_dkT

    return [total_dy_dLp, total_dy_dLc, total_dy_dSt, total_dy_dkT]


def marko_siggia_ewlc_solve_distance_derivative(f, Lp, Lc, St, kT=4.11):
    fsq = f * f
    c = (
        -f
        * Lc ** 3
        * (
            f ** 2 * Lp * St
            + fsq * kT
            + 2 * f * Lp * St ** 2
            + 2.25 * f * St * kT
            + Lp * St ** 3
            + 1.5 * St ** 2 * kT
        )
        / (St ** 3 * kT)
    )
    b = (
        Lc ** 2
        * (
            2 * f ** 2 * Lp * St
            + 3 * fsq * kT
            + 2 * f * Lp * St ** 2
            + 4.5 * f * St * kT
            + 1.5 * St ** 2 * kT
        )
        / (St ** 2 * kT)
    )
    a = -f * Lc * Lp / kT - 3 * f * Lc / St - 2.25 * Lc

    total_dy_da, total_dy_db, total_dy_dc = invwlc_root_derivatives(a, b, c, 1)

    # Map back to our output parameters
    dc_df = (
        -(Lc ** 3)
        * (
            3.0 * fsq * Lp * St
            + 3.0 * fsq * kT
            + 4.0 * f * Lp * St ** 2
            + 4.5 * f * St * kT
            + Lp * St ** 3
            + 1.5 * St ** 2 * kT
        )
        / (St ** 3 * kT)
    )
    db_df = (
        Lc ** 2 * (4 * f * Lp * St + 6 * f * kT + 2 * Lp * St ** 2 + 4.5 * St * kT) / (St ** 2 * kT)
    )
    da_df = -Lc * Lp / kT - 3 * Lc / St

    return total_dy_da * da_df + total_dy_db * db_df + total_dy_dc * dc_df
