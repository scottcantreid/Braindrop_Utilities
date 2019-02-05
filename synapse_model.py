import numpy as np

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