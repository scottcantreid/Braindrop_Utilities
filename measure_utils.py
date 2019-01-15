import timeit
import os.path
import sys
sys.path.append('/home/scottreid/Calibration')
import TestEquityM106 as tst
import numpy as np
import pylab
import seaborn
seaborn.set_style("white")
import scipy.interpolate
from IPython.display import clear_output
import matplotlib.pyplot as plt
from os import system, name
import time

from pystorm.hal import HAL
from pystorm.PyDriver import bddriver as bd
from pystorm.hal.net_builder import NetBuilder
from pystorm.hal.calibrator import Calibrator, PoolSpec
from pystorm.hal import data_utils
from pystorm.hal.run_control import RunControl
from pystorm.hal.neuromorph import graph

"""
Temperature Control
"""
from pyModbusTCP.client import ModbusClient
from struct import *

def ints_to_float(tup):
    mypack = pack('>HH',tup[0],tup[1])
    return unpack('>f', mypack)

def read_temp(c):
    c.open()
    regs = c.read_holding_registers(27586, 2)
    c.close()

    if regs:
        return ints_to_float(regs)[0]

def write_setpoint(c, setpoint):
    SP = float(setpoint)
    tup = unpack('<HH', pack('<f', setpoint))
    tup = tup[::-1]
    c.open()
    if c.write_multiple_registers(2782, tup):
        return True
    else:
        return False
    c.close()

def read_setpoint(c):
    c.open()
    regs = c.read_holding_registers(2782, 2)
    c.close()
    if regs:
        return ints_to_float(regs)[0]
    else:
        print("read error")

def open_connection(port = 502):
    # Connect to our favorite port
    c = ModbusClient(host='171.65.103.149', port=port, auto_open=True)
    connected = test_connection(c)
    if (not connected):
        port = 0
    while (not connected):
        c = ModbusClient(host='171.65.103.149', port=port, auto_open=True)
        connected = test_connection(c)
        port += 1
        if (port > 3000):
            break
    if(connected):
        print('Connected to TestEquity M106 with port ' + str(port))
    else:
        print('Not Connected')
    if(connected):
        return c

def test_connection(c):
    c.open()
    reg = c.read_holding_registers(0)
    c.close()
    return reg is not None

def initialize_and_calibrate(Y, X, LY, LX, Din=1, twiddle = True):
    hal = HAL()
    print("HAL Initialized")
    net_builder = NetBuilder(hal)
    cal = Calibrator(hal)

    ps = PoolSpec(YX=(Y,X), loc_yx=(LY, LX), D=Din)

    nrn_tap_matrix, syn_tap_matrix = cal.create_optimized_yx_taps(ps)
    ps.TPM = nrn_tap_matrix
    print("Tap Points Optimized")

    ps.fmax = cal.optimize_fmax(ps)
    print("Fmax Optimized")
    dacs = {
            'DAC_SOMA_REF': 1024,
            'DAC_DIFF_G': 1024,
            'DAC_DIFF_R': 1024}
    if(twiddle):
        print("Beginning Twiddle Optimization\n")
        ps, dacs, opt_encs, opt_offsets, dbg = \
            cal.optimize_yield(ps, dacs=dacs,
                               bias_twiddle_policy='greedy_flat', offset_source='calibration_db', validate=True)
        print("\nTwiddle Optimization Complete")
    return ps, dacs, hal

def map_network(ps, decoders, hal, Din = 1, Dout = 1):
    net = graph.Network("net")
    print(str(Din) + ' --> ' + str(Dout))
    i1 = net.create_input("i1", Din)
    p1 = net.create_pool("p1", ps.TPM, biases=ps.biases)
    b1 = net.create_bucket("b1", Dout)
    o1 = net.create_output("o1", Dout)

    net.create_connection("c_i1_to_p1", i1, p1, None)
    decoder_conn = net.create_connection("c_p1_to_b1", p1, b1, decoders)
    net.create_connection("c_b1_to_o1", b1, o1, None)

    print("Calling Map")
    hal.map(net)
    return net, i1, p1, b1, o1

def measure_tuning_curves(hal, ps, sample_pts, dacs={}, training_hold_time = 2):
    """Validate the output of get_encoders_and_offsets
    Samples neuron firing rates at supplied sample_pts, compares to
    est_encs * sample_pts + est_offsets, to directly assess predictive
    quality of est_encs and est_offsets.
    Inputs:
    =======
    est_encs (NxD array) : encoder estimates
    est_offsets (N array) : offset estimates
    ps : (PoolSpec object)
        required pars: YX, loc_yx, TPM, fmax
        relevant pars: gain_divisors, biases, diffusor_cuts_yx,
    sample_pts (SxD array) : points to sample in the input space
    Returns:
    =======
    rmse_err, meas_rates, est_rates
    rmse_err (float) : RMSE firing rate error
    meas_rates (SxN array) : firing rates of each neuron at each sample_pt
    est_rates (SxN array) : what the est_encoders/offsets predicted
    """
    ps.check_specified(['YX', 'loc_yx', 'TPM', 'fmax'])
    total_training_points = sample_pts.size

    HOLD_TIME = training_hold_time # seconds
    LPF_DISCARD_TIME = 0.5 # seconds

    N = ps.X * ps.Y
    decoders = np.zeros((1,N))

    net, i1, p1, b1, o1 = map_network(ps, decoders, hal)

    for dac, value in dacs.items():
        hal.set_DAC_value(dac, value)
    # let the DACs settle down
    time.sleep(.2)

    FUDGE = 2
    curr_time = hal.get_time()
    times = np.arange(0, total_training_points) * training_hold_time * 1e9 + curr_time + FUDGE * 1e9
    times_w_end = np.hstack((times, times[-1] + training_hold_time * 1e9))
    vals = sample_pts * ps.fmax
    input_vals = {i1 : (times, vals)}

    rc = RunControl(hal, net)

    _, spikes_and_bin_times = rc.run_input_sweep(input_vals, get_raw_spikes=True, end_time=times_w_end[-1], rel_time=False)
    spikes, spike_bin_times = spikes_and_bin_times
    discard_frac = LPF_DISCARD_TIME / HOLD_TIME
    A = data_utils.bins_to_rates(spikes[p1], spike_bin_times, times_w_end, init_discard_frac=.2)

    outputs = hal.get_outputs()

    return A, sample_pts*ps.fmax

def measure_output_decoder(hal, ps, sample_pts, decoders, dacs={}, training_hold_time = 2):
    """Validate the output of get_encoders_and_offsets
    Samples neuron firing rates at supplied sample_pts, compares to
    est_encs * sample_pts + est_offsets, to directly assess predictive
    quality of est_encs and est_offsets.
    Inputs:
    =======
    est_encs (NxD array) : encoder estimates
    est_offsets (N array) : offset estimates
    ps : (PoolSpec object)
        required pars: YX, loc_yx, TPM, fmax
        relevant pars: gain_divisors, biases, diffusor_cuts_yx,
    sample_pts (SxD array) : points to sample in the input space
    Returns:
    =======
    rmse_err, meas_rates, est_rates
    rmse_err (float) : RMSE firing rate error
    meas_rates (SxN array) : firing rates of each neuron at each sample_pt
    est_rates (SxN array) : what the est_encoders/offsets predicted
    """
    ps.check_specified(['YX', 'loc_yx', 'TPM', 'fmax'])
    total_training_points = sample_pts.size

    HOLD_TIME = training_hold_time # seconds
    LPF_DISCARD_TIME = np.min([HOLD_TIME/2, 0.5]) # seconds

    net, i1, p1, b1, o1 = map_network(ps, decoders, hal, Dout = decoders.shape[0])

    for dac, value in dacs.items():
        hal.set_DAC_value(dac, value)
    # let the DACs settle down
    time.sleep(.2)

    FUDGE = 2
    curr_time = hal.get_time()
    times = np.arange(0, total_training_points) * training_hold_time * 1e9 + curr_time + FUDGE * 1e9
    times_w_end = np.hstack((times, times[-1] + training_hold_time * 1e9))
    vals = sample_pts * ps.fmax
    input_vals = {i1 : (times, vals)}

    rc = RunControl(hal, net)

    outputs_and_bin_times, _ = rc.run_input_sweep(input_vals, get_raw_spikes=False, end_time=times_w_end[-1], rel_time=False)
    tags, tag_bin_times = outputs_and_bin_times
    yhat = data_utils.bins_to_rates(tags[o1], tag_bin_times, times_w_end, init_discard_frac=.2)
    yhat = yhat
    return yhat, sample_pts*ps.fmax

def set_temp(setT, inst, printo = True, epsilon = 0.15, sit_time = 15, time_out = 240):
    # Write new temperature setpoint
    if( write_setpoint(inst, setT) ):
        print('Wrote new setpoint, now waiting for stability')

        # Moniter the temperature until it has either stabilized to the setpoint or timeout occurs
        t0 = time.time() # Time of initialization
        t1 = time.time() # Most recent time when Tcurr was not within epsilon of setT
        tcurr = time.time() # Current time

        isstable = False
        istimeout = False
        while( not (isstable or istimeout)):
            inst.open()
            currT = read_temp(inst)
            inst.close()
            if (currT is not None):
                if (np.abs(currT - setT) < epsilon):
                    tcurr = time.time()
                else:
                    tcurr = time.time()
                    t1 = time.time()
                time.sleep(0.1)
            isstable = (tcurr - t1) > sit_time
            istimeout = (tcurr - t0) > time_out

        if(printo):
            if(istimeout):
                print("Timeout")
            else:
                print("Temperature stable within " + str(epsilon) + " C of " + str(setT) + ' C')

        # Measure the thermal stability
        t_meas = sit_time
        t_wait = 0.0
        temp = []
        t2 = time.time()
        t1 = time.time()
        while((t2 - t1) < t_meas):
            currT = read_temp(inst)
            if (currT is not None):
                temp.append(currT)
            t2 = time.time()
            time.sleep(t_wait)

        avg_temp = np.mean(temp)
        std_temp = np.std(temp)
        if(printo):
            print("Average Temp is " + str(avg_temp)[:5] + '+-' + str(std_temp)[:5] + ' C')
        return istimeout, avg_temp, std_temp

    else:
        print('Failed to write new setpoint')

def get_intercept(a, fs, plot = False, sm_factor = 5, epsilon = 3, min_fire = 10):
    sm_factor = 5
    epsilon = 3
    min_fire = 10

    always_fire = False
    never_fire = False
    if(np.min(a) > min_fire):
        intercept = None
        always_fire = True
    else:
        b = 0
        f_red = 0
        for i in range(sm_factor):
            b = b + a[i:-sm_factor+i]
            f_red = f_red + fs[i:-sm_factor+i]
        b = b/sm_factor
        f_red = f_red/sm_factor

        deriv = b[1:] - b[:-1]
        f_red = (f_red[1:] + f_red[:-1])/2
        sgn = np.mean(deriv) > 0
        if(sgn > 0):
            f_set = f_red[deriv > epsilon]
            if (len(f_set) == 0):
                intercept = None
                never_fire = True
            else:
                intercept = f_set[0][0]
        else:
            f_set = f_red[deriv < -epsilon]
            if (len(f_set) == 0):
                intercept = None
                never_fire = True
            else:
                intercept = f_set[-1][0]
    if(plot):
        if(always_fire):
            print("Always Fires")
        elif(never_fire):
            print("Never Fires")
        else:
            print("Intercept at " + str(intercept) + " Hz")
        plt.plot(fs, a)
        if(intercept is not None):
            plt.axvline(intercept)

    return intercept, always_fire, never_fire

def interpret_yield(A, x, printo = False):
    Q, N = A.shape
    tercepts = []
    afire = np.ndarray(N)
    nfire = np.ndarray(N)
    for n in range(N):
        intercept, always_fire, never_fire = get_intercept(A[:,n], x)
        if (intercept is not None):
            tercepts.append(intercept)
        afire[n] = always_fire
        nfire[n] = never_fire

    af = np.sum(afire)
    nf = np.sum(nfire)
    neff = N - af - nf
    if (printo):
        print("Yield for " + str(N) + " total neurons:")
        print(str(neff) + " neurons fire with intercepts (" + str(neff*100/N)[:4] + "%)")
        print(str(af) + " neurons fire without intercepts (" + str(af*100/N)[:4] + "%)")
        print("In total, " + str(af + neff) + " neurons fire (" + str((af+neff)*100/N)[:4] + "%)")
        print(str(nf) + " neurons never fire (" + str(nf*100/N)[:4] + "%)")
        plt.hist(tercepts)
        plt.xlabel('Intercept')
        plt.title('Intercept Distribution')
    else:
        return tercepts, af, nf, neff
