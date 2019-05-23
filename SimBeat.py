from math import *
import matplotlib.pyplot as plt
import numpy as np

#main


i_t=0           #time
i_aud_in=1      #T+LL - preprogrammed auditory input
i_aud_out=2     #T - auditory cortex output
i_gate=3        #T - STN/GPi loop gating status
i_put_tempo=4   #LL - tempo-specific output of putamen to GPi
i_thal=5        #T+LL - GPi output represented in thalamic pathway
i_sma_rel=6     #status of SMA relative timing cells -- TO DO: multiple paths
i_sma_abs=7     #status of SMA absolute timing cells -- TO DO: multiple paths
i_sma_mod=8     #LL - output of SMA speed-modulating cells
i_cort_tempo=9  #LL - output of cortical tempo representation
i_sma_antic=10  #T - anticipatory output from SMA
i_confidence=11  #T - anticipatory output from SMA
i_tap_timer=12
i_tap=13
i_tap_mod = 14
n_vars=15


class Params(object):
    def __init__(self):
        self.tmax = 8.
        self.dt = .001

        self.abs_timer_max = .75         # The longest beat it's willing to consider
        self.rel_timer_max_multiple = 2  # Multiple of expected interval after which timing ends
        self.abs_noise = 0.005
        self.rel_noise = 0.005
        self.confidence_decay = .2
        self.confidence_spike = 7
        self.confidence_drop = 1
        self.antic_baseline = .25#0.1 #
        self.antic_goodline = 0.9#
        
        self.skepticism = 1
        self.imagination = 0#0.45
        self.volume = 1
        
        self.rel_tap_sensitivity = .2
        self.abs_tap_sensitivity = .1
        self.focus = 5
        
    def get_n_steps(self):
        return floor(self.tmax/self.dt)
        
    def get_input(self, signal):
        n_steps = self.get_n_steps()
        aud_input=[0]*n_steps
        time_counter = 0
    
        j=0
        index=0
        aud_input[index]=self.volume
        while (j < len(signal) and index < n_steps):

            
            time_counter=0

            while (time_counter<signal[j] and index < n_steps-1):
                index += 1
                time_counter += self.dt
            
            j += 1
            aud_input[index]=self.volume
        return aud_input
        
    

def antic_func(t, focus, confidence):
    #return exp(-(focus*t)**2) + exp(-(focus*(t-1))**2)
    return (1-confidence)*1 + confidence*max(0, min(max(1-focus*t, focus*(t-1)+1), -focus*(t-1)+1))
#    t = 2.33*t-1.10     # The multiplier determines how much beat anticipation is rushed -- 2.27 is slight.
    #return -.5*(1-t**6)*exp(-t**6/2)+1/(t**6+3) + .333
#    return -.5*(1-t**4)*exp(-t**4/2)+1/(t**4+3) + antic_baseline

def prc(x, sensitivity):
    return min(2*x/4, 2**(1-4*(x/(sensitivity**1.3)-5)))
    
def prc2(x, sensitivity):
    return min(1*x/4, 2**(1-4*(x/(sensitivity**1.3)-5)))

def disp_data(testnum,data,params):
        fig = plt.figure(figsize=(20,6))
        
        plt.subplot(3,1,1)
        plt.plot(data[testnum, :,i_t], data[testnum, :,i_sma_antic],'darkblue')
        plt.plot(data[testnum, :,i_t], data[testnum, :,i_aud_in], 'darkgreen')
        plt.plot(data[testnum, :,i_t], data[testnum, :,i_confidence],'indigo')
#        plt.plot(data[testnum, :,i_t], [params.antic_baseline]*len(data[testnum, :,i_t]))
#        plt.plot(data[testnum, :,i_t], [params.antic_goodline]*len(data[testnum, :,i_t]))
        
        plt.subplot(3,1,2)
        plt.plot(data[testnum, :,i_t], data[testnum, :,i_sma_rel],'b')
        plt.plot(data[testnum, :,i_t], data[testnum, :,i_sma_abs],'r')
        plt.plot(data[testnum, :,i_t], data[testnum, :,i_sma_mod],'m')
        
    
        plt.subplot(3,1,3)
        plt.plot(data[testnum, :,i_t], data[testnum, :,i_tap_timer],'k')
        plt.plot(data[testnum, :,i_t], data[testnum, :,i_tap],'b')  
        
        fig.show()

def run_model(signals, params):
    
    tmax = params.tmax
    dt = params.dt
    n_steps = params.get_n_steps()
    abs_timer_max = params.abs_timer_max         # The longest beat it's willing to consider
    rtmm = params.rel_timer_max_multiple  # Multiple of expected interval after which timing ends
    abs_noise = params.abs_noise
    rel_noise = params.rel_noise
    confidence_decay = params.confidence_decay
    confidence_spike = params.confidence_spike
    confidence_drop = params.confidence_drop
    antic_baseline = params.antic_baseline
    antic_goodline = params.antic_goodline

                
    skepticism = params.skepticism
    imagination = params.imagination
    volume = params.volume
    focus = params.focus
    rel_tap_sensitivity = params.rel_tap_sensitivity
    abs_tap_sensitivity = params.abs_tap_sensitivity
    n_signals = len(signals)    
    
    data = np.array([[[0.0]*n_vars]*n_steps]*n_signals)
    
    for signal_num in range(n_signals):
    
        data[signal_num, :,i_t]=np.array([i*dt for i in range(0,n_steps)])
        
        data[signal_num, 0,i_sma_abs] = abs_timer_max
        data[signal_num, 0,i_sma_rel]=0
        data[signal_num, 0,i_sma_mod] = dt/.5
        data[signal_num, 0,i_put_tempo] = 100
        data[signal_num, 0,i_cort_tempo] = 100
        data[signal_num, :,i_aud_in] = params.get_input(signals[signal_num])
        data[signal_num, 0,i_confidence] = 0
    
        step = 1
    
        while(step<n_steps):

            aud_out_0 = data[signal_num, step-1,i_aud_out]
            gate_0 = data[signal_num, step-1,i_gate]
            put_tempo_0 = data[signal_num, step-1,i_put_tempo]
        #    thal_0 = data[step-1,i_thal]
            sma_rel_0 = data[signal_num, step-1,i_sma_rel]
            sma_abs_0 = data[signal_num, step-1,i_sma_abs]
            sma_mod_0 = data[signal_num, step-1,i_sma_mod]
            cort_tempo_0 = data[signal_num, step-1,i_cort_tempo]
            sma_antic_0 = data[signal_num, step-1,i_sma_antic]
            confidence_0 = data[signal_num, step-1,i_confidence]
            tap_timer_0 = data[signal_num, step-1,i_tap_timer]
            tap_mod_0 = data[signal_num, step-1,i_tap_mod]
        
            aud_out = data[signal_num, step-1, i_aud_in]
            
            heard_a_beat = False
            
            if aud_out_0 + sma_antic_0 > 1+antic_baseline:  # If there is a sound in the approximate vicinity of the expected time ### (and confidence is low?)
                heard_a_beat = True
            
            if heard_a_beat: #######
                sma_abs = 0#prc(sma_abs_0, .05) # restart absolute timer
                
                if sma_abs_0<abs_timer_max and sma_abs_0>.2 and confidence_0<.75:           # Unless the absolute timer already maxed out
                    cort_tempo = sma_abs_0                                 # cortex senses a coincidence between a specific absolute timing population and an auditory input, and sets its tempo.
                else:                                                  # Otherwise
                    cort_tempo=cort_tempo_0                                # no tempo change
            else:                                                  # Otherwise
                cort_tempo=cort_tempo_0                                # keep the tempo
                if sma_abs_0 < abs_timer_max:                          # If the absolute timer hasn't maxed out
                    sma_abs = sma_abs_0 + dt + sqrt(dt)*abs_noise*(np.random.rand()-.5)                               # absolute timer advances
                else:                                                  # Otherwise
                    sma_abs = sma_abs_0                                    # timer is stopped
            
            if sma_rel_0 < rtmm:              # If the relative timer hasn't yet maxed out
                sma_rel = sma_rel_0 + dt*sma_mod_0 + sqrt(dt)*rel_noise*(np.random.rand()-.5)     # relative timer advances
                tap_timer = tap_timer_0 + dt*tap_mod_0 + sqrt(dt)*rel_noise*(np.random.rand()-.5)
            else:                                               # Otherwise
                sma_rel = rtmm                    # relative timer is stopped        
            
            
            if (sma_rel_0 > 1 and sma_rel_0 < 1.1) or heard_a_beat:#aud_out_0 + sma_antic_0*(1+imagination*confidence_0) >= skepticism: # If there is a sound in the approximate vicinity of the expected time OR a beat is confidently anticipated
                sma_rel = 0#prc2(sma_rel_0, rel_tap_sensitivity) 
                if confidence_0 >.75:
                    sma_abs = 0#prc(sma_abs_0, .1)# restart abs timer
                if tap_timer<2/3:
                    tap_timer= 0#prc(sma_rel_0, rel_tap_sensitivity)

            if tap_timer_0>1:
                data[signal_num, step, i_tap]=1
                sma_rel = sma_rel#-prc(sma_rel, rel_tap_sensitivity)
                sma_abs = sma_abs#-prc2(sma_abs, abs_tap_sensitivity)
                tap_timer=sma_rel 
                tap_mod = sma_mod_0      
            
            confidence = confidence_0 - dt*confidence_decay*confidence_0
            confidence = confidence + heard_a_beat*confidence_spike*max(sma_antic_0-antic_goodline, 0)*(1-confidence_0)
            confidence = confidence + heard_a_beat*confidence_drop*min(sma_antic_0-antic_goodline, 0)*confidence_0
        
            put_tempo = cort_tempo_0                             # cortex sends tempo representation to putamen
    
            sma_mod = 1/put_tempo_0                                  # SMA modulators stay the course
            if tap_timer <2/3:
                tap_mod = 1/put_tempo_0
            else:
                tap_mod = tap_mod_0 
   
    
            sma_antic = antic_func(sma_rel, focus, confidence)                    # Amount of anticipation is a function of the state of the relative timer
        
            data[signal_num, step,i_aud_out]=aud_out
            data[signal_num, step,i_put_tempo]=put_tempo
            data[signal_num, step,i_sma_rel]=sma_rel
            data[signal_num, step,i_sma_abs]=sma_abs
            data[signal_num, step,i_sma_mod]=sma_mod
            data[signal_num, step,i_cort_tempo]=cort_tempo
            data[signal_num, step,i_sma_antic]=sma_antic
            data[signal_num, step,i_confidence]=confidence
            data[signal_num, step,i_tap_timer]=tap_timer
            data[signal_num, step,i_tap_mod]=tap_mod
            step = step+1
    return data





supplemental_figs = False
verbose = False
alltests = True
tapping = False
phase_period_test = False
test_pics = False

params = Params()

if phase_period_test:
    #phase test
    phase_shifts = []
    rel_phase_shifts = []
    for i in range(-4,5):
        phase_shifts.append(i/50)
        rel_phase_shifts.append(i/50/.5 * 100)
    signals=[]
    for i in range(len(phase_shifts)):
        signals.append( [.5, .5, .5,.5, .5, .5+phase_shifts[i], .5+phase_shifts[i], .5+phase_shifts[i]])
    data = run_model(signals, params)
    ITIs = []
    responses = []
    
    
    for i in range(len(phase_shifts)):
        if test_pics:
            disp_data(i,data,params)
        ITIs.append([])
        lasttap = 0
        for j in range(params.get_n_steps()):
 
            if data[i,j,i_tap]==1 and data[i,j,i_t]-lasttap > .1:
                ITIs[i].append(data[i,j,i_t]-lasttap)
                lasttap = data[i,j,i_t]
        
        if len(ITIs[i])>=6:
            responses.append( (ITIs[i][5]/.5*100)-100)
        else:
            responses.append(-1)
    fig=plt.figure()
    plt.plot(rel_phase_shifts, responses, '.')
    plt.xlabel('phase shift magnitude')
    plt.ylabel('phase correction response')
    fig.show()
    fig=plt.figure()
    for i in range(len(phase_shifts)):
        plt.plot(ITIs[i][3:])
    plt.xlabel('tap')
    plt.ylabel('ITI (s)')
    fig.show()



else:
    if alltests:
        s1 = [.55]*5 + [.5]*8
        s2 = [.5]*4 + [.55]*8
        s3 = [.5]*4 + [1]+ [.5]*8
        s4 = [.5]*3+[3]+[.5]*8
        s5 = [.5]*3 + [1.1]+ [.4]*8
        s6 = [.5,.5]+[.5, .25, .5, .25, .5]*5  
        s7 = [.6]*4 + [.3]*20
        #s2 = [10]*(n_steps//8) + [.5]*(n_steps//8) + [10]*(n_steps//8) + [.3]*(n_steps//8)+ [.35]*(n_steps//8) + [10]*(n_steps//8) + [.6]*(n_steps//8) + [.5]*(n_steps//8) +100*[10]
        signals = [s1,s2,s3, s4, s5, s6, s7]
    else:
        s1 = [.5, .5, .5, .5, .5, .5]
        signals = [s1]

# for testnum in range(len(signals)):
#     fig=plt.figure(1, figsize=(20,5))
#     output = params.get_input(signals[testnum])
#     plt.plot(range(len(output)), output)
#     fig.show()

    data = run_model(signals, params)
        
        
    for testnum in range(len(signals)):
        disp_data(testnum,data,params)