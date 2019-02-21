import numpy as np
from scipy.optimize import curve_fit

"""
These impulse and step responses are derived from the circuit model of the Braindrop synapses.

@param t: a numpy array of time values to sample the impulse response over, in units of seconds
@param tau_s: the first-order time constant of the synapse, of order 100 ms, in units of seconds
@param tau_p: the second-order time constant of the synapse, of order 2 ms, in units of seconds
@param eps: the pulse-extender's pulse-width, of order 0.5 ms, in units of seconds

@returns resp: a numpy array of the impulse or step response of the synapse circuit
"""

def synapse_imp_response(t, tau_s, tau_p, eps):
	case1 = (t-eps) >= 0
	case2 = ((t-eps) < 0) & (t >= 0)
	
	resp1 = 1/(tau_s - tau_p)*(tau_s*np.exp(-t/tau_s)*(np.exp(eps/tau_s) - 1) - tau_p * np.exp(-t/tau_p)*(np.exp(-eps/tau_p)-1))
	resp2 = 1/(tau_s - tau_p)*(-tau_s * (np.exp(-t/tau_s) -1) + tau_p * (np.exp(-t/tau_p) - 1))
	resp = np.zeros_like(t)
	resp[case1] = resp1[case1]
	resp[case2] = resp2[case2]
	
	return resp

def synapse_step_response(t, tau_s, tau_p, eps):
    t1 = np.min([t, [eps]*len(t)], axis = 0)
    resp1 = 1/(tau_s - tau_p)*(tau_s**2 * (np.exp(-t1/tau_s) -1) - tau_p**2 * (np.exp(-t1/tau_p) - 1) + (tau_s - tau_p)*t1)
    resp2 = 1/(tau_s - tau_p)*(  tau_s**2 * (np.exp(-eps/tau_s) - np.exp(-t/tau_s))*(np.exp(eps/tau_s) - 1) - \
                              tau_p**2 * (np.exp(-eps/tau_p) - np.exp(-t/tau_p))*(np.exp(eps/tau_p) - 1)  )
    case1 = (t-eps) > 0
    case2 = t > eps
    resp = np.zeros_like(t)
    resp[case1] += resp1[case1]
    resp[case2] += resp2[case2]
    return resp

def mismatch(T, g1, g2):
    return np.exp(-g1 * 1/T - g2)

def syn_temp_model(T, kappa, g01, g11, g12, g21, g22):
    L0 = mismatch(T, g01, 0)
    L1 = mismatch(T, g11, g12)
    L2 = mismatch(T, g21, g22)
    return kappa*L0/(1 + L1 - L2)

def first_order_model(t, tau):
    return 1 - np.exp(-t/tau)

def thermal_fo_step(x, kappa, g01, g11, g12, g21, g22):
    t, T = x
    tau = syn_temp_model(T, kappa, g01, g11, g12, g21, g22)
    response = first_order_model(t, tau)
    return response

def simpler_syn(T, kappa, g01, g02, g11, g12):
    return kappa /(1 + mismatch(T, g01, g02) - mismatch(T, g11, g12))

def simpler_fo_step(x, kappa, g01, g02, g11, g12):
    t, T = x
    tau = simpler_syn(T, kappa, g01, g02, g11, g12)
    response = first_order_model(t, tau)
    return response

def draw_simpler_params():
    kappa = 1
    g01 = np.random.randn()*200
    g02 = np.random.randn()*3 - 3
    g11 = np.random.randn()*200
    g12 = np.random.randn()*3 + 3
    return kappa, g01, g02, g11, g12

def simple_thermal_step_fit(step, t, T, N_trial = 10):
    best_fit = []
    best_resid = np.infty
    for _ in range(N_trial):
        try:
            fit, cov = curve_fit(simpler_fo_step, (t,T), step, p0 = draw_simpler_params())
        except RuntimeError:
            pass
        else:
            model = simpler_fo_step((t, T), *fit)
            res = np.linalg.norm(model - step)
            if (res < best_resid):
                best_resid = res
                best_fit = fit
    return best_fit

def draw_params():
    kappa = 1
    g01 = np.random.randn()*200
    g11 = np.random.randn()*200
    g12 = np.random.randn()*3 - 3
    g21 = np.random.randn()*200
    g22 = np.random.randn()*3 + 3
    return kappa, g01, g11, g12, g21, g22

def draw_step(N_T = 10, noise = 0):
	T = np.linspace(300, 350, N_T)
	N = 1000

	params = draw_params()
	t = np.linspace(0, 2, N)
	t = np.array(list(t)*N_T)
	T = np.array([[T[i]]*N for i in range(N_T)]).flatten()

	step = thermal_fo_step((t, T), *params) + np.random.randn(N*N_T)*noise
	return step, T, t