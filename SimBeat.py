from math import *
import matplotlib.pyplot as plt
import numpy as np

#main


i_t=0           #time
i_aud_in=1      #T+LL - preprogrammed auditory input
i_aud_out=2     #T - auditory cortex output
i_put_tempo=4   #LL - tempo-specific output of putamen to GPi
i_thal=5        #T+LL - GPi output represented in thalamic pathway
i_sma_rel=6     #status of SMA relative timing cells -- TO DO: multiple paths
i_sma_abs=7     #status of SMA absolute timing cells -- TO DO: multiple paths
i_sma_mod=8     #LL - output of SMA speed-modulating cells
i_cort_tempo=9  #LL - output of cortical tempo representation
i_sma_antic=10  #T - anticipatory output from SMA
i_IBS=11  #T - anticipatory output from SMA
n_vars=12


class Params(object):
    def __init__(self):
        self.tmax = 8.
        self.dt = .001

        self.abs_timer_max = 2             # Time until abs timer ends
        self.rel_timer_max_multiple = 2    # Multiple of expected interval after which rel timing ends
        
        self.abs_noise = 0.005             # Amplitude of brownian drift
        self.rel_noise = 0.005
        
        self.IBS_decay = 0       # Exponential decay rate of IBS
        self.IBS_spike = 7        # Multiplier for IBS spike on accurate prediction
        self.IBS_drop = 1         # Multiplier for IBS drop on inaccurate prediction
        self.antic_baseline = .25 # Anticipation necessary to hear a beat at volume 1
        self.antic_goodline = 0.9 # Above here, hearing a beat increases IBS; below, it decreases IBS.
        self.skepticism = .75     # If IBS is above here, beats can be imagined
        self.volume = 1           # Amplitude of auditory signal
        self.focus = 5            # Precision of anticipatory windows
        
    # Figure out number of time steps from dt and tmax
    def get_n_steps(self):
        return floor(self.tmax/self.dt)
     
    # Get auditory signal from list of inter-click intervals   
    def get_input(self, signal):
        n_steps = self.get_n_steps()
        aud_input=[0]*n_steps
        time_counter = 0
    
        j=0
        index=0
#        aud_input[0]=self.volume
        while (j < len(signal) and index < n_steps):

            
            time_counter=0

            while (time_counter<=signal[j] and index < n_steps-1):
                index += 1
                time_counter += self.dt
                
            
            print(time_counter)
            print(index)
            j += 1
            aud_input[index]=self.volume
        return aud_input
        
    
# Determines level of beat anticipation (upside-down of hyperdirect drive)
# inputs:
#  t - relative timer
#  focus - precision parameter 
def antic_func(t, focus):
    return max(0, min(max(1-focus*t, focus*(t-1)+1), -focus*(t-1)+1))

# Display data from a specific test
def disp_data(testnum,data,params):
        
        f, (a2, a0, a1) = plt.subplots(3,1, gridspec_kw = {'height_ratios':[1, 3, 3]}, figsize=(7,4))
        
        a2.plot(data[testnum, :,i_t], data[testnum, :,i_aud_in], 'darkgreen')
        a2.set_ylim([0,1.2])
        a2.set_xlim([0,5])        
        a0.plot(data[testnum, :,i_t], data[testnum, :,i_sma_rel],'b')
        a0.plot(data[testnum, :,i_t], data[testnum, :,i_sma_abs],'r')
        a0.plot(data[testnum, :,i_t], 1/np.array(data[testnum, :,i_sma_mod]),"violet")
        a0.set_ylim([0,1.2])        
        a0.set_xlim([0,5]) 
                
        a1.plot(data[testnum, :,i_t], data[testnum, :,i_IBS],'k')
        a1.plot(np.array(data[testnum, :,i_t]), 1-np.array(data[testnum, :,i_sma_antic]),'b')

        a1.plot(data[testnum, :,i_t], [params.skepticism]*len(data[testnum, :,i_t]), "orange")
             
        a1.set_xlim([0,5]) 
        a1.set_ylim([0,1.2])        
        f.tight_layout()
        
        f.show()

# Runs simulation
# inputs:
#  signals - list of lists of inter-click intervals
#  p - parameter object
def run_model(signals, p):
    
    tmax = params.tmax
    dt = params.dt
    n_steps = params.get_n_steps()
    abs_timer_max = params.abs_timer_max 
    rtmm = params.rel_timer_max_multiple
    abs_noise = params.abs_noise
    rel_noise = params.rel_noise
    IBS_decay = params.IBS_decay
    IBS_spike = params.IBS_spike
    IBS_drop = params.IBS_drop
    antic_baseline = params.antic_baseline
    antic_goodline = params.antic_goodline

                
    skepticism = params.skepticism
    volume = params.volume
    focus = params.focus
    n_signals = len(signals)    
    
    data = np.array([[[0.0]*n_vars]*n_steps]*n_signals)
    
    for signal_num in range(n_signals):
    
        data[signal_num, :,i_t]=np.array([i*dt for i in range(0,n_steps)])
        
        data[signal_num, 0,i_sma_abs] = 0#abs_timer_max
        data[signal_num, 0,i_sma_rel]=0
        data[signal_num, 0,i_sma_mod] = 0
        data[signal_num, 0,i_put_tempo] = .5*dt
        data[signal_num, 0,i_cort_tempo] = .5*dt
        data[signal_num, :,i_aud_in] = params.get_input(signals[signal_num])
        data[signal_num, 0,i_IBS] = 0
        data[signal_num, 0,i_sma_antic] = 0
    
        step = 1
    
        while(step<n_steps):

            aud_out_0 = data[signal_num, step-1, i_aud_in]
            put_tempo_0 = data[signal_num, step-1,i_put_tempo]
            sma_rel_0 = data[signal_num, step-1,i_sma_rel]
            sma_abs_0 = data[signal_num, step-1,i_sma_abs]
            sma_mod_0 = data[signal_num, step-1,i_sma_mod]
            cort_tempo_0 = data[signal_num, step-1,i_cort_tempo]
            sma_antic_0 = data[signal_num, step-1,i_sma_antic]
            IBS_0 = data[signal_num, step-1,i_IBS]
        
#            aud_out = data[signal_num, step-1, i_aud_in]
            
            sma_antic = antic_func(sma_rel_0, p.focus)
            augmented_antic_signal = (1-IBS_0)*1 + IBS_0*sma_antic
            
            heard_a_beat = False
            felt_a_beat = False
            
            if aud_out_0 + augmented_antic_signal > 1+p.antic_baseline:  # If there is a sound in the approximate vicinity of the expected time ### (and IBS is low?)
                heard_a_beat = True
            
            if heard_a_beat: #######
                sma_abs = 0 # restart absolute timer
                
                if (not np.isnan(sma_abs_0)) and IBS_0<p.skepticism:           # Unless the absolute timer already maxed out
                    cort_tempo = sma_abs_0                                 # cortex senses a coincidence between a specific absolute timing population and an auditory input, and sets its tempo.
                    put_tempo = cort_tempo                             # cortex sends tempo representation to putamen
                    sma_mod = 1/put_tempo
                else:                                                  # Otherwise
                    cort_tempo=cort_tempo_0                                # no tempo change
                    put_tempo = put_tempo_0                             # cortex sends tempo representation to putamen
                    sma_mod = sma_mod_0                                 # SMA modulators stay the course
            else:                                                  # Otherwise
                cort_tempo=cort_tempo_0                                # keep the tempo
                put_tempo = put_tempo_0                             # cortex sends tempo representation to putamen
    
                sma_mod = sma_mod_0                                 # SMA modulators stay the course
                if sma_abs_0 < p.abs_timer_max:                          # If the absolute timer hasn't maxed out
                    sma_abs = sma_abs_0 + p.dt + sqrt(p.dt)*p.abs_noise*(np.random.rand()-.5)                               # absolute timer advances
                else:                                                  # Otherwise
                    sma_abs = np.nan                                    # timer is stopped
            
            if sma_rel_0 < p.rel_timer_max_multiple:              # If the relative timer hasn't yet maxed out
                sma_rel = sma_rel_0 + p.dt*sma_mod_0 + sqrt(p.dt)*p.rel_noise*(np.random.rand()-.5)     # relative timer advances
            else:                                               # Otherwise
                sma_rel = sma_rel#np.nan                    # relative timer is stopped        
            
            
            
            if (sma_rel_0 > 1 and sma_rel_0 < 1.1 and IBS>p.skepticism) or heard_a_beat: # If there is a sound in the approximate vicinity of the expected time OR a beat is confidently anticipated
                felt_a_beat = True
                sma_rel = 0
#                if IBS_0 >.75:
#                sma_abs = 0  #prc(sma_abs_0, .1)# restart abs timer

            IBS = IBS_0 - p.dt*p.IBS_decay*IBS_0
            
            if (not np.isnan(sma_rel)):
                IBS = IBS + heard_a_beat*p.IBS_spike*max(sma_antic-p.antic_goodline, 0)*(1-IBS_0)
                IBS = IBS + heard_a_beat*p.IBS_drop*min(sma_antic-p.antic_goodline, 0)*IBS_0
   
         
#            data[signal_num, step,i_aud_out]=aud_out
            data[signal_num, step,i_put_tempo]=put_tempo
            data[signal_num, step,i_sma_rel]=sma_rel
            data[signal_num, step,i_sma_abs]=sma_abs
            data[signal_num, step,i_sma_abs]=sma_abs
            data[signal_num, step,i_sma_mod]=sma_mod
            data[signal_num, step,i_cort_tempo]=cort_tempo
            data[signal_num, step,i_sma_antic]=sma_antic
            data[signal_num, step,i_IBS]=IBS
            step = step+1
    return data


supplemental_figs = False
verbose = False
alltests = True
phase_period_test = False
test_pics = False
basicfigs = False

params = Params()


if alltests:
    signals = []
    # signals.append( [.4]*8 )
    # signals.append( [.6]*8 )
    signals.append( [.55]*4 + [.5]*8 )
    signals.append( [.5]*4 + [.55]*8)
    signals.append( [.5]*3+[.55]+[.5]*5)
    signals.append( [.5]*3+[.45]+[.5]*5)
    signals.append( [.5,.5]+[.5, .25, .5, .25, .5]*5 )
    signals.append( [.6]*4 + [.3]*20)
    signals.append( [.5]*3 + [2.5]+ [.5]*8)
    signals.append( [.5]*4 + [1]+ [.5]*8)
   
else:
    s1 = [.5, .5, .5, .5, .5, .5]
    signals = [s1]
data = run_model(signals, params)
    
        
for testnum in range(len(signals)):
    disp_data(testnum,data,params)