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

def resample_mark_onsets(raw_et,desired_sampling_rate=1000):
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
    
    return raw_et

def run_epoching(et_preprocessed,baseline_time,metadata=[]):
    annot_events = mne.events_from_annotations(et_preprocessed)
    preprocessed_et_epoched = mne.Epochs(et_preprocessed, annot_events[0], event_id=annot_events[1], tmin=-baseline_time/1000, tmax=np.max(trial_length)/raw_et.info['sfreq'], preload=True, baseline=None,
                        reject=None , reject_by_annotation=False,metadata=metadata)
    return preprocessed_et_epoched


#*******************************#
###### PREPROCESSING STEPS ######
#*******************************#
# Step 1: We are removing all samples that either go beyond the screen size
def remove_invalid_samples(raw_et):
    x_idx                                           = np.where([i=='xpos' for i in raw_et.ch_names])[0]
    y_idx                                           = np.where([i=='ypos' for i in raw_et.ch_names])[0]
    pupil_idx                                       = np.where([i=='pupil' for i in raw_et.ch_names])[0]
    # eyes                                            = pd.DataFrame(raw_et._data.T,columns=raw_et.ch_names)
    withinwidth                                     = raw_et._data[x_idx,:]<(screensize_pix[0])
    withinheight                                    = raw_et._data[y_idx,:]<(screensize_pix[1])
    valid_pupil_size                                = raw_et._data[pupil_idx,:]>0

    is_valid                                        = np.array([x & y & z for x,y,z in zip(withinwidth,withinheight,valid_pupil_size)]).astype(bool).flatten()
    is_valid                                        = remove_loners(is_valid,et_refreshrate=1000)
    is_valid                                        = expand_gap(raw_et.times*1000,is_valid)

    return is_valid.astype(bool)

# Step 2: Checking how much the pupil dilation changes from timepoint to timepoint and exclude timepoints where the dilation is very different
def madspeedfilter(raw_et,is_valid):
    dia                                         = raw_et._data[np.where([i=='pupil' for i in raw_et.ch_names])[0],:].flatten()
    max_gap                                     = 200
    dilation                                    = dia[is_valid]
    cur_tv                                      = (raw_et.times*1000)[is_valid]
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
    valid_out                                   = remove_loners(valid_out.astype(bool),raw_et.info['sfreq'])
    valid_out                                   = expand_gap((raw_et.times*1000),valid_out)
    valid_out                                   = remove_loners(valid_out.astype(bool),raw_et.info['sfreq'])

    return valid_out.astype(bool)

# Step 3: Fitting a smooth line and exclude samples that deviate from that fitted line
def mad_deviation(raw_et,is_valid):
    dia                                         = raw_et._data[np.where([i=='pupil' for i in raw_et.ch_names])[0],:].flatten()
    n_passes                                    = 4
    mad_multiplier                              = 16
    interp_fs                                   = 100
    lowpass_cf                                  = 16
    [smooth_filt_b,smooth_filt_a]               = butter(1,lowpass_cf/(interp_fs/2))
    t_interp                                    = np.arange((raw_et.times*1000)[0],(raw_et.times*1000)[-1],1000/lowpass_cf)
    dia[~is_valid]                              = np.arange(len(dia[~is_valid]))*np.nan
    is_valid_running                            = is_valid.copy()
    residuals_per_pass                          = np.empty([len(is_valid),n_passes])
    smooth_baseline_per_pass                    = np.empty([len(is_valid),n_passes])

    is_done                                     = False
    for pass_id in range(n_passes):
        if is_done: 
            break
        is_valid_start                          = is_valid_running.copy()

        residuals_per_pass[:,pass_id], smooth_baseline_per_pass[:,pass_id] = deviation_calculator((raw_et.times*1000),dia,[x and y for x, y in zip(is_valid_running, is_valid)],t_interp,smooth_filt_a,smooth_filt_b)

        mad                                     = np.nanmedian(np.abs(residuals_per_pass[:,pass_id]-np.nanmedian(residuals_per_pass[:,pass_id])))
        threshold                               = np.nanmedian(residuals_per_pass[:,pass_id])+mad_multiplier*mad

        is_valid_running                        = [x and y for x,y in zip((residuals_per_pass[:,pass_id] <= threshold), is_valid)]
        is_valid_running                        = remove_loners(np.array(is_valid_running).astype(bool),raw_et.info['sfreq'])
        is_valid_running                        = expand_gap((raw_et.times*1000),np.array(is_valid_running).astype(bool))
        
        if (pass_id>0 and np.all(is_valid_start==is_valid_running)):
            is_done                             = True
    valid_out                                   = is_valid_running
    return valid_out.astype(bool)

#*******************************#
####### RUN PREPROCESSING #######
#*******************************#
def run_preproc(raw_et):
    # Step 1: remove out of bounds
    isvalid1 = remove_invalid_samples(raw_et)

    # Step 2: speed dilation exclusion
    isvalid2 = madspeedfilter(raw_et,isvalid1)

    # Step 3: deviation from smooth line
    isvalid3 = mad_deviation(raw_et,isvalid2)

    # # remove invalid samples & transform to degrees (and rel. size for pupil)
    et_preproc = raw_et.copy()
    et_preproc._data[:,~isvalid3] = np.nan
    
    x_idx = np.where([i=='xpos' for i in et_preproc.ch_names])[0]
    x = et_preproc._data[x_idx,:].flatten()-screensize_pix[0]/2
    et_preproc._data[x_idx,:] = [pix_to_deg(i,screenwidth_pix,screenwidth_cm,screendistance_cm) for i in x]
    
    y_idx = np.where([i=='ypos' for i in et_preproc.ch_names])[0]
    y = et_preproc._data[y_idx,:].flatten()-screensize_pix[1]/2
    et_preproc._data[y_idx,:] = [pix_to_deg(i,screenwidth_pix,screenwidth_cm,screendistance_cm) for i in y]
    
    pupil_idx = np.where([i=='pupil' for i in et_preproc.ch_names])[0]
    et_preproc._data[pupil_idx,:] = et_preproc._data[pupil_idx,:]/np.nanmax(et_preproc._data[pupil_idx,:])
    

    return et_preproc


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
    if task == 'exp1b':
        trial_dur= 3300
    if task == 'exp2':
        trial_dur = 2500
    if task == 'exp3':
        trial_dur = 3500

    ##############
    #### RUN #####
    ##############
    if task != 'exp3':
        # load data
        trialmat                = pd.read_csv(f'{bids_dir}/{task}/sub-{subjid}/sub-{subjid}_trialmat.csv')
        edf_fn                  = [f for f in os.listdir(f'{bids_dir}/{task}/sub-{subjid}/') if f.endswith('.edf') and not f.startswith('._')][0]
        edf_file                = read_edf(f'{bids_dir}/{task}/sub-{subjid}/{edf_fn}')
        dfs                     = edf_file.to_pandas()
        raw_et, calibration     = edf_file.to_mne() 
        rename_dict             = {var: ('xpos' if 'xpos' in var else 'ypos' if 'ypos' in var else 'pupil') for var in raw_et.ch_names}
        raw_et.rename_channels(rename_dict)

        # run preprocecssing 
        raw_et                  = resample_mark_onsets(raw_et,desired_sampling_rate=1000)
        trial_length            = trial_dur/(1000//raw_et.info['sfreq'])
        eyes_preproc            = run_preproc(raw_et)
        eyes_epoched            = run_epoching(eyes_preproc,baseline_time,trialmat)
        
        # save stuff
        raw_et.save(f'{bids_dir}/{task}/derivatives/preprocessed/sub-{subjid}_{task}_eyetracking-raw.fif',overwrite=True)
        eyes_preproc.save(f'{bids_dir}/{task}/derivatives/preprocessed/sub-{subjid}_{task}_eyetracking-preproc-raw.fif',overwrite=True)
        eyes_epoched.save(f'{bids_dir}/{task}/derivatives/preprocessed/sub-{subjid}_{task}_eyetracking-preproc-epo.fif',overwrite=True)
    
    # for experiment 3 the two conditions were separately collected back to back -- this stacks the epoched data so we have an easier time analysing
    else: 
        # load data
        trialmat                = pd.read_csv(f'{bids_dir}/{task}/sub-{subjid}/sub-{subjid}_trialmat.csv')

        all_epochs = []
        for condition in ['occluded','transparent']:
            edf_fn                  = [f for f in os.listdir(f'{bids_dir}/{task}/sub-{subjid}/') if f.endswith(f'{condition}.edf') and not f.startswith('._')][0]
            edf_file                = read_edf(f'{bids_dir}/{task}/sub-{subjid}/{edf_fn}')
            dfs                     = edf_file.to_pandas()
            raw_et, calibration     = edf_file.to_mne() 
            rename_dict             = {var: ('xpos' if 'xpos' in var else 'ypos' if 'ypos' in var else 'pupil') for var in raw_et.ch_names}
            raw_et.rename_channels(rename_dict)

            # run preprocecssing 
            raw_et                  = resample_mark_onsets(raw_et,desired_sampling_rate=1000)
            trial_length            = trial_dur/(1000//raw_et.info['sfreq'])
            eyes_preproc            = run_preproc(raw_et)
            eyes_epoched            = run_epoching(eyes_preproc,baseline_time,trialmat[trialmat.condition==condition])
            all_epochs.append(eyes_epoched)

            # save stuff
            raw_et.save(f'{bids_dir}/{task}/derivatives/preprocessed/sub-{subjid}_{task}{condition}_eyetracking-raw.fif',overwrite=True)
            eyes_preproc.save(f'{bids_dir}/{task}/derivatives/preprocessed/sub-{subjid}_{task}{condition}_eyetracking-preproc-raw.fif',overwrite=True)
            eyes_epoched.save(f'{bids_dir}/{task}/derivatives/preprocessed/sub-{subjid}_{task}{condition}_eyetracking-preproc-epo.fif',overwrite=True)

        mne.concatenate_epochs(all_epochs).save(f'{bids_dir}/{task}/derivatives/preprocessed/sub-{subjid}_{task}_eyetracking-preproc-epo.fif',overwrite=True)