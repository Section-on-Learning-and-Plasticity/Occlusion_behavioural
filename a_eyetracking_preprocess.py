import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import butter,filtfilt
from scipy.interpolate import interp1d
import math, mne, glob, os
from eyelinkio import read_edf
#*******************************#
####### HELPER FUNCTIONS ########
#*******************************#
def pix_to_deg(full_size_pix,screenwidth_pix,screenwidth_cm,screendistance_cm):
    pix_per_cm = screenwidth_pix/screenwidth_cm
    size_cm = full_size_pix/pix_per_cm
    dva = math.atan(size_cm/2/screendistance_cm)*2
    return np.rad2deg(dva)

def deviation_calculator(tv,dia,is_valid,t_interp,smooth_filt_a,smooth_filt_b):
    dia_valid                                       = dia[[x and y for x,y in zip(is_valid,~np.isnan(dia))]]
    t_valid                                         = tv[[x and y for x,y in zip(is_valid,~np.isnan(dia))]]
    interp_f_lin                                    = interp1d(t_valid,dia_valid,kind='linear',bounds_error=False)
    interp_f_near                                   = interp1d(t_valid,dia_valid,kind='nearest',fill_value='extrapolate')
    extrapolated                                    = interp_f_near(t_interp)
    uniform_baseline                                = interp_f_lin(t_interp)
    uniform_baseline[np.isnan(uniform_baseline)]    = extrapolated[np.isnan(uniform_baseline)]
    smooth_uniform_baseline                         = filtfilt(smooth_filt_b,smooth_filt_a,uniform_baseline)
    interp_f_baseline                               = interp1d(t_interp,smooth_uniform_baseline,kind='linear',bounds_error=False)
    smooth_baseline                                 = interp_f_baseline(tv)
    dev = np.abs(dia-smooth_baseline)

    return(dev,smooth_baseline)

def expand_gap(tv,is_valid):
    min_gap_width                                   = 75
    max_gap_width                                   = 2000
    pad_back                                        = 100
    pad_forward                                     = 150
    valid_t                                         = tv[is_valid]
    valid_idx                                       = np.where(is_valid)[0]
    gaps                                            = np.diff(valid_t)
    needs_padding                                   = [x and y for x,y in zip(gaps>min_gap_width,gaps<max_gap_width)]
    gap_start_t                                     = valid_t[np.pad(needs_padding,(0,1),constant_values=False)]
    gap_end_t                                       = valid_t[np.pad(needs_padding,(1,0),constant_values=False)]

    remove_idx = []
    for i_start,i_end in zip(gap_start_t,gap_end_t):
        # when the gap is super large, it's most likely a recording artifact (and not an eyeblink), so we should clean around it more
        if i_end -i_start > 500:
            pb                                      = pad_back * 2
            pf                                      = pad_forward * 2
        else:
            pb                                      = pad_back
            pf                                      = pad_forward
        remove_idx.extend(np.where([x and y for x,y in zip(valid_t>(i_start-pb),valid_t<(i_end+pf))])[0])
    remove_idx                                      = np.unique(remove_idx)

    if remove_idx.any():
        is_valid[valid_idx[remove_idx]]             = False
    return is_valid

def remove_loners(is_valid,et_refreshrate):
    valid_out                                       = is_valid.copy()
    
    lonely_sample_max_length                        = 100 #in ms
    time_separation                                 = 40 #in ms
    valid_idx                                       = np.where(is_valid)[0]

    gap_start                                       = valid_idx[np.where(np.pad(np.diff(valid_idx),(0,1),constant_values=1)>1)]
    gap_end                                         = valid_idx[np.where(np.pad(np.diff(valid_idx),(1,0),constant_values=1)>1)]
    start_valid_idx                                 = [valid_idx[0]]
    end_valid_idx                                   = [valid_idx[-1]]
    valid_data_chunks                               = np.reshape(np.sort(np.concatenate([start_valid_idx,gap_start,gap_end,end_valid_idx])),[-1,2])
    size_valid_data_chunks                          = np.diff(valid_data_chunks,axis=1)
    size_idx                                        = np.where((size_valid_data_chunks/et_refreshrate*1000)<lonely_sample_max_length)[0]
    separation                                      = np.squeeze(np.diff(np.reshape(np.sort(np.concatenate([gap_start,gap_end])),[-1,2]),axis=1))
    sep_idx                                         = np.where(np.pad([i>(time_separation*1/et_refreshrate) for i in separation],(1,0)))[0]
    data_chunks_to_delete                           = valid_data_chunks[np.intersect1d(sep_idx,size_idx)]

    for i in data_chunks_to_delete:
        valid_out[np.arange(i[0],i[1]+1)]           = 0

    print('removed ' + str(is_valid.sum()-valid_out.sum()) + ' samples')
        
    return valid_out.astype(bool)

def make_et_dataframe(raw_et,desired_sampling_rate=1000):
    # check sampling rate
    if raw_et.info['sfreq']!=desired_sampling_rate:
        print(f"sampling rate is {raw_et.info['sfreq']}Hz. This does not match the desired sampling rate of {desired_sampling_rate}Hz")
        raw_et_resampled = raw_et.copy()
        raw_et_resampled._data = np.where(np.isnan(raw_et_resampled._data), 0, raw_et_resampled._data)
        raw_et_resampled.resample(desired_sampling_rate)

        nan_idx=np.isnan(raw_et._data)
        sampling = raw_et._data.shape[1]/raw_et_resampled._data.shape[1]
        if sampling<1:#we upsampled
            nan_idx=nan_idx.repeat(raw_et_resampled._data.shape[1]/raw_et._data.shape[1],axis=1)
        else:
            nan_idx=nan_idx[::raw_et_resampled._data.shape[1]/raw_et._data.shape[1]]

        raw_et_resampled._data[nan_idx] = np.nan
        raw_et = raw_et_resampled.copy()

    # get annotations
    event_array, annot_dict  = mne.events_from_annotations(raw_et)
    on_keys = [key for key in annot_dict if key.endswith('_on')]
    # off_keys = [key for key in annot_dict if key.endswith('_off')]
    selected_values = [event_array[np.where(event_array[:,2]==annot_dict[key])[0][0],:] for key in on_keys]
    new_events =np.sort(np.array(selected_values),axis=0)
    print(f'min trial distance {np.min(np.diff(new_events[:,0]))*(1/raw_et.info["sfreq"])} s')
    print(f'max trial distance {np.max(np.diff(new_events[:,0]))*(1/raw_et.info["sfreq"])} s')
    annot = mne.Annotations(onset=new_events[:,0]/raw_et.info['sfreq'],duration=np.zeros(len(new_events)),description=[f'Trial {str(i).zfill(3)}' for i in range(len(new_events))])
    raw_et.set_annotations(annot)

    # make a dataframe with x, y, pupil, trial_onset (1 to mark onset, otherwise 0), time
    df = pd.DataFrame()
    df['x'] = raw_et._data[[i[0:4]=='xpos' for i in raw_et.ch_names],:].flatten()
    df['y'] = raw_et._data[[i[0:4]=='ypos' for i in raw_et.ch_names],:].flatten()
    df['pupil'] = raw_et._data[[i[0:5]=='pupil' for i in raw_et.ch_names],:].flatten()
    df['trial_onset'] = np.zeros(len(raw_et.times))
    df.loc[np.array(selected_values)[:,0],'trial_onset']=1
    df['time'] = raw_et.times*1000

    return df,raw_et

def make_preprocessed_mne(df,raw_et,trial_length,baseline,metadata=[]):
    preprocessed_et = raw_et.copy()
    preprocessed_et._data[[i[0:4]=='xpos' for i in preprocessed_et.ch_names],:] = df['x'].to_numpy()
    preprocessed_et._data[[i[0:4]=='ypos' for i in preprocessed_et.ch_names],:] = df['y'].to_numpy()
    preprocessed_et._data[[i[0:5]=='pupil' for i in preprocessed_et.ch_names],:] = df['pupil'].to_numpy()
    annot_events = mne.events_from_annotations(preprocessed_et)
    preprocessed_et_epoched = mne.Epochs(preprocessed_et, annot_events[0], event_id=annot_events[1], tmin=-baseline/1000, tmax=np.max(trial_length)/1000, preload=True, baseline=None,
                        reject=None , reject_by_annotation=False,metadata=metadata)
    return preprocessed_et_epoched

def get_trial_length(trialmat):
    return trialmat['trial_length'].to_numpy()

def load_trialmat(bids_dir,subjid):
    return pd.read_csv(glob.glob(f'{bids_dir}/sub-{subjid}/*.csv')[0])      

#*******************************#
###### PREPROCESSING STEPS ######
#*******************************#
# Step 1: We are removing all samples that either go beyond the screen size
def remove_invalid_samples(eyes,tv):
    et_refreshrate                                  = 1000/np.diff(tv)[1]
    withinwidth                                     = eyes['x']<(screensize_pix[0])
    withinheight                                    = eyes['y']<(screensize_pix[1])
    valid_pupil_size                                = eyes['pupil']>0

    is_valid                                        = np.array([x and y and z for x,y,z in zip(withinwidth,withinheight,valid_pupil_size)]).astype(bool)
    is_valid                                        = remove_loners(is_valid,et_refreshrate)
    is_valid                                        = expand_gap(tv,is_valid)

    return is_valid.astype(bool)

# Step 2: Checking how much the pupil dliation changes from timepoint to timepoint and exclude timepoints where the dilation is very different
def madspeedfilter(tv,dia,is_valid):
    et_refreshrate                              = 1000/np.diff(tv)[1]
    max_gap                                     = 200
    dilation                                    = dia[is_valid]
    cur_tv                                      = tv[is_valid]
    cur_dia_speed                               = np.diff(dilation)/np.diff(cur_tv)
    cur_dia_speed[np.diff(cur_tv)>max_gap]      = np.nan
  

    back_dilation                               = np.pad(cur_dia_speed,(1,0),constant_values=np.nan)
    fwd_dilation                                = np.pad(cur_dia_speed,(0,1),constant_values=np.nan)
    back_fwd_dilation                           = np.vstack([back_dilation,fwd_dilation])

    max_dilation_speed                          = np.full_like(dia,np.nan)
    max_dilation_speed[is_valid]                = np.nanmax(np.abs(back_fwd_dilation),axis=0)

    mad                                         = np.nanmedian(np.abs(max_dilation_speed-np.nanmedian(max_dilation_speed)))
    mad_multiplier                              = 16 # as defined in Kret et al., 2019

    if mad == 0: 
        print('mad is 0, using dilation speed plus constant as threshold')
        threshold                               = np.nanmedian(max_dilation_speed)+mad_multiplier
    else:
        threshold                               = np.nanmedian(max_dilation_speed)+mad_multiplier*mad
    print('threshold: ' + str(threshold))

    valid_out                                   = is_valid.copy()
    valid_out[max_dilation_speed>=threshold]    = False
    valid_out                                   = remove_loners(valid_out.astype(bool),et_refreshrate)
    valid_out                                   = expand_gap(tv,valid_out)
    valid_out                                   = remove_loners(valid_out.astype(bool),et_refreshrate)

    return valid_out.astype(bool)

# Step 3: Fitting a smooth line and exclude samples that deviate from that fitted line
def mad_deviation(tv,dia,is_valid):
    et_refreshrate                              = 1000/np.diff(tv)[1]
    n_passes                                    = 4
    mad_multiplier                              = 16
    interp_fs                                   = 100
    lowpass_cf                                  = 16
    [smooth_filt_b,smooth_filt_a]               = butter(1,lowpass_cf/(interp_fs/2))
    t_interp                                    = np.arange(tv[0],tv[-1],1000/lowpass_cf)
    dia[~is_valid]                              = np.arange(len(dia[~is_valid]))*np.nan
    is_valid_running                            = is_valid.copy()
    residuals_per_pass                          = np.empty([len(is_valid),n_passes])
    smooth_baseline_per_pass                    = np.empty([len(is_valid),n_passes])

    is_done                                     = False
    for pass_id in range(n_passes):
        if is_done: 
            break
        is_valid_start                          = is_valid_running.copy()

        residuals_per_pass[:,pass_id], smooth_baseline_per_pass[:,pass_id] = deviation_calculator(tv,dia,[x and y for x, y in zip(is_valid_running, is_valid)],t_interp,smooth_filt_a,smooth_filt_b)

        mad                                     = np.nanmedian(np.abs(residuals_per_pass[:,pass_id]-np.nanmedian(residuals_per_pass[:,pass_id])))
        threshold                               = np.nanmedian(residuals_per_pass[:,pass_id])+mad_multiplier*mad

        is_valid_running                        = [x and y for x,y in zip((residuals_per_pass[:,pass_id] <= threshold), is_valid)]
        is_valid_running                        = remove_loners(np.array(is_valid_running).astype(bool),et_refreshrate)
        is_valid_running                        = expand_gap(tv,np.array(is_valid_running).astype(bool))
        
        if (pass_id>0 and np.all(is_valid_start==is_valid_running)):
            is_done                             = True
    valid_out                                   = is_valid_running
    return valid_out.astype(bool)

#*******************************#
####### RUN PREPROCESSING #######
#*******************************#
def run_preproc(eyes,tv,dia):
    # Step 1: remove out of bounds
    isvalid1 = remove_invalid_samples(eyes,tv)

    # Step 2: speed dilation exclusion
    isvalid2 = madspeedfilter(tv,dia,isvalid1)

    # Step 3: deviation from smooth line
    isvalid3 = mad_deviation(tv,dia,isvalid2)

    # # remove invalid samples & transform to degrees (and rel. size for pupil)
    eyes_preproc = eyes.copy()

    eyes_preproc.loc[~isvalid3,['x','y','pupil']]  = eyes_preproc.loc[~isvalid3,['x','y','pupil']]*np.nan
    x = (eyes_preproc['x']-screensize_pix[0]/2)
    eyes_preproc['x'] = [pix_to_deg(i,screenwidth_pix,screenwidth_cm,screendistance_cm) for i in x]
    
    y = (eyes_preproc['y']-screensize_pix[1]/2)
    eyes_preproc['y'] = [pix_to_deg(i,screenwidth_pix,screenwidth_cm,screendistance_cm) for i in y]
    
    eyes_preproc['pupil_rel'] = eyes_preproc['pupil']/eyes_preproc['pupil'].max()

    return eyes_preproc


def run_epoching(trial_length_samples,eyes_preproc,baseline_samples):
    onsets = np.where(eyes_preproc.trial_onset==1)[0]
    trial_idx=[]
    for i,v in enumerate(onsets):
        trial_idx.append(np.arange(v-baseline_samples,v+trial_length_samples[i]))

    eyes_preproc['trial']=0
    for i,v in enumerate(trial_idx):
        eyes_preproc.loc[v,'trial']=i+1

    eyes_epoched = eyes_preproc.copy()
    eyes_epoched.drop(list(np.where(eyes_epoched['trial']==0)[0]),inplace=True)
    eyes_epoched.reset_index(drop=True,inplace=True)

    return eyes_epoched


#*****************************#
### COMMAND LINE INPUTS ###
#*****************************#
if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-bids_dir",
        required=True,
        help='path to bids root directory',
    )

    parser.add_argument(
        "-subjid",
        required=True,
        help='subject id',
    )

    parser.add_argument(
        "-task",
        required=True,
        help='BIDS task tag',
    )

    args = parser.parse_args()
    bids_dir = args.bids_dir
    subjid = args.subjid
    task = args.task

    # parameters 
    screenwidth_cm = 53.5
    screendistance_cm = 57
    screensize_pix = [1920, 1080]
    screenwidth_pix = screensize_pix[0]
    baseline_time = 500
    if task == 'exp1a':
        trial_dur= 1650
    

    # load data
    trialmat                = load_trialmat(bids_dir,subjid)
    edf_fn                  = [f for f in os.listdir(f'{bids_dir}/sub-{subjid}/') if f.endswith('.edf') and not f.startswith('._')][0]
    edf_file                = read_edf(f'{bids_dir}/sub-{subjid}/{edf_fn}')
    dfs                     = edf_file.to_pandas()
    raw_et, calibration     = edf_file.to_mne() 
    rename_dict             = {var: ('xpos' if 'xpos' in var else 'ypos' if 'ypos' in var else 'pupil') for var in raw_et.ch_names}
    raw_et.rename_channels(rename_dict)

    # run preprocecssing
    raw_et_df,raw_et        = make_et_dataframe(raw_et,desired_sampling_rate=1000)
    trial_length            = trial_dur/(1000//raw_et.info['sfreq'])
    eyes_preproc            = run_preproc(raw_et_df,raw_et_df.time.to_numpy(),raw_et_df['pupil'].copy().to_numpy())
    eyes_epoched            = run_epoching(trial_length,eyes_preproc,baseline_time/(1000//raw_et.info['sfreq']))
    preproc_et              = make_preprocessed_mne(eyes_preproc,raw_et,trial_length,baseline_time,trialmat)
    
    # save stuff
    eyes_epoched.to_csv(f'{bids_dir}/derivatives/sub-{subjid}_{task}_eyetracking_epoched.csv')
    preproc_et.save(f'{bids_dir}/derivatives/sub-{subjid}_{task}_eyetracking-epo.fif',overwrite=True)
    eyes_preproc.save(f'{bids_dir}/derivatives/sub-{subjid}_{task}_eyetracking_preprocessed.fif',overwrite=True)